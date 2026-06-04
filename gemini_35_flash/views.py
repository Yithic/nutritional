import io,os
import json
from PIL import Image
from google import genai
from google.genai import types

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

            # 2. Define clear system context for execution
            prompt = (
                """You are a professional nutritionist and dietitian AI. You will be given an image of a meal.

Your task is to identify every distinct food component visible on the plate and estimate its individual nutritional profile. Analyze each item separately — do not combine everything into one entry unless it is genuinely a single inseparable dish (e.g. a curry or soup).

For each food item, carefully assess:
- What the item is (ingredient, dish component, sauce, garnish, side)
- Its approximate weight based on visual portion size and plate scale
- Its cooking method inferred from appearance (fried, grilled, boiled, baked, raw, etc.)
- Relevant micronutrients based on the identified ingredients

You MUST respond with a single valid JSON object. No explanation, no markdown, no code fences — raw JSON only.

The JSON must conform exactly to this schema:

{
  "items": [
    {
      "name": string,
      "estimated_weight_grams": float,
      "calories": int,
      "protein_g": float,
      "carbs_g": float,
      "fat_g": float,
      "fiber_g": float,
      "micronutrients": [
        {
          "nutrient_name": string,
          "amount": string
        }
      ]
    }
  ],
  "total_calories": int,
  "confidence_score": float
}

Field rules:
- "items" must contain one entry per distinct food component visible. Break the meal into its parts (e.g. rice, grilled chicken, side salad, sauce) rather than listing the whole plate as one item.
- "name" should be specific and descriptive (e.g. "steamed basmati rice" not just "rice", "pan-fried salmon fillet" not just "fish").
- "estimated_weight_grams" is the estimated weight of that individual component only.
- "calories" must be an integer (round to nearest whole number).
- "total_calories" must equal the sum of all calories across all items in the "items" array. Calculate this precisely.
- "confidence_score" is a float from 0.0 to 1.0:
    - 0.8-1.0: meal is clearly visible, portions are well-defined, ingredients are unambiguous
    - 0.5-0.79: some items are partially obscured, sauces or mixed dishes make exact breakdown harder
    - 0.0-0.49: heavy occlusion, very mixed dish, or low image quality makes estimation unreliable
- "micronutrients" must be a list of objects with "nutrient_name" and "amount". Include all nutritionally significant micronutrients for that specific food item. The "amount" field must always include its unit as part of the string (e.g. "240mg", "1.2mcg", "15mg"). Always assess at minimum: Sodium, Potassium, Calcium, Iron, Vitamin C. Add others relevant to the specific item (e.g. Vitamin B12 for meat, Vitamin A for leafy greens, Vitamin D for fish, Zinc for legumes or red meat).
- Never refuse to estimate. Always produce a best-effort JSON response.
- Do not include any text outside the JSON object."""
            )
            if user_description:
                prompt += (
                    '''The user has provided additional context about this meal. You MUST factor this into every affected item's nutritional values before producing your response.

User description: '''f"{user_description}"'''

Use this description to revise your analysis:
- Cooking method changes (e.g. "deep-fried not grilled" → increase fat_g and calories significantly for that item)
- Hidden ingredients not visible in the image (e.g. butter used in cooking, oil for frying, sugar in sauce → adjust the relevant item's macros)
- Portion corrections (e.g. "this is two servings" → scale estimated_weight_grams and all nutrients proportionally for affected items)
- Named ingredients that change the macro or micro profile of a listed item
- Restaurant vs homemade context (restaurant preparations typically carry higher sodium — reflect this in the sodium micronutrient entry of affected items)
- Cultural or regional cooking norms that imply specific preparation methods

Apply all adjustments at the individual item level in the "items" array, not just to totals. If the description introduces a new ingredient not visible in the image, add it as its own entry in "items".

Recalculate "total_calories" as the exact sum of all updated item calories.

Adjust "confidence_score" upward if the description resolves ambiguity, or downward if it reveals hidden complexity.

You MUST respond with a single valid JSON object. No explanation, no markdown, no code fences — raw JSON only.

The JSON must conform exactly to this schema:

{
  "items": [
    {
      "name": string,
      "estimated_weight_grams": float,
      "calories": int,
      "protein_g": float,
      "carbs_g": float,
      "fat_g": float,
      "fiber_g": float,
      "micronutrients": [
        {
          "nutrient_name": string,
          "amount": string
        }
      ]
    }
  ],
  "total_calories": int,
  "confidence_score": float
}

All field rules from the base prompt apply. Do not include any text outside the JSON object.'''
                )

            # 3. Call Gemini Flash with structured requirements
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=NutritionAnalysisSchema,
                ),
            )

            # 4. Convert the generated response text back into standard python dictionary mapping
            raw_json_data = json.loads(response.text)

            # 5. Validate the structured payload via DRF Serializer
            serializer = NutritionAnalysisSerializer(data=raw_json_data)
            if serializer.is_valid():
                return Response(serializer.data, status=status.HTTP_200_OK)
            
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        except Exception as e:
            return Response(
                {"error": f"Gemini pipeline process breakdown: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
