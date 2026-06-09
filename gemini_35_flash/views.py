import io
import os
import json
import time
from PIL import Image
from google import genai
from google.genai import types
from .models import MealLog

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import NutritionAnalysisSchema, NutritionAnalysisSerializer

# Initialize the official Google GenAI Client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

class AnalyzeMealView(APIView):
    # Enable the endpoint to receive raw multi-part file uploads
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        start_time = time.perf_counter()
        file_obj = request.FILES.get('image')
        user_description = request.data.get('description', '').strip()
        
        if not file_obj:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not file_obj.content_type.startswith("image/"):
            return Response({"error": "Uploaded file must be an image type."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 1. Read binary content and transform it into a PIL Image object
            image_bytes = file_obj.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Safe defensive color mode conversion for any format (PNG, WebP, etc.)
            if image.mode != "RGB":
                if image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    mask = image.split()[-1] if image.mode in ("RGBA", "LA") else None
                    background.paste(image, mask=mask)
                    image = background
                else:
                    image = image.convert("RGB")

            # Scale perfectly to match Gemini's token grid thresholds
            max_ratio = (768, 768)
            image.thumbnail(max_ratio)

            # 2. Define clear system context for execution
            prompt = (
                """Act as a professional dietitian AI. Analyze the visible food in this image. Break the meal into distinct components (do not aggregate unless it is an inseparable dish like soup). 

For each distinct item, estimate:
1. Name and cooking method
2. Approximate weight (grams) based on plate scale
3. Macronutrients

Rules:
- Strictly follow the JSON schema.
- "total_calories" must be the exact sum of the items array.
- "confidence_score" (0.0 to 1.0) reflects visual clarity.
- Never refuse to estimate."""
            )
            
            if user_description:
                prompt += (
                    f'''User Context: {user_description}
CRITICAL: Revise your baseline visual estimates using this context. Adjust macros for hidden ingredients (e.g., butter/oil), update portion scales if mentioned, and modify cooking methods accordingly. Apply all adjustments at the individual item level.'''
                )

            # 3. Call Gemini Flash with structured requirements
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=NutritionAnalysisSchema,
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW
                ),
            )
            
            try:
                # 4. Parse the JSON response text
                meal_data = json.loads(response.text)
                items_list = meal_data.get("items", [])

                # 5. Extract and aggregate data defensively (supporting both snake_case and camelCase)
                total_calories = int(meal_data.get("total_calories") or meal_data.get("totalCalories") or 0)
                
                total_protein = 0.0
                total_carbs = 0.0
                total_fat = 0.0
                total_fibre = 0.0
                names = []

                for item in items_list:
                    # Extract names dynamically
                    name = item.get("name")
                    if name:
                        names.append(name)
                        
                    # Extract macros dynamically supporting both naming conventions
                    total_protein += float(item.get("protein_g") or item.get("proteinG") or 0)
                    total_carbs += float(item.get("carbs_g") or item.get("carbsG") or 0)
                    total_fat += float(item.get("fat_g") or item.get("fatG") or 0)
                    total_fibre += float(item.get("fiber_g") or item.get("fiberG") or 0)


                # Combine item names into a single string for your flat DB entry
                combined_name = ", ".join(names)
                if not combined_name:
                    combined_name = "Unknown Meal Plate"

                # 6. Extract token usage metadata from the response object
                usage = getattr(response, 'usage_metadata', None)
                in_tokens = usage.prompt_token_count if usage else 0
                out_tokens = usage.candidates_token_count if usage else 0
                tot_tokens = usage.total_token_count if usage else 0

                # 7. Save directly into your Django SQLite database
                end_time = time.perf_counter()
                duration = round(end_time - start_time, 3)
                saved_meal = MealLog.objects.create(
                    meal_name=combined_name[:255], 
                    calories=total_calories,
                    protein_g=total_protein,
                    carbs_g=total_carbs,
                    fat_g=total_fat,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    total_tokens=tot_tokens,
                    fiber_g=total_fibre,
                    time=duration
                )

                return Response({
                    "status": "success",
                    "message": f"Logged {saved_meal.meal_name} successfully!",
                    "tokens_used": tot_tokens,
                    "raw_analysis": meal_data
                }, status=status.HTTP_201_CREATED)

            except (json.JSONDecodeError, ValueError) as e:
                return Response({
                    "status": "error", 
                    "message": "Failed to parse nutrition metrics or serialize to database structure."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            return Response(
                {"error": f"Gemini pipeline process breakdown: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )