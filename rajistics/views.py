import io
import os
import torch
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import requests
from transformers import pipeline

# 1. Initialize Hugging Face Pipeline globally (loads once when server starts)
MODEL_NAME = "rajistics/finetuned-indian-food"
print(f"Loading Hugging Face model: {MODEL_NAME}...")
food_classifier = pipeline("image-classification", model=MODEL_NAME)

# 2. USDA API Configurations
USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


def get_calories_from_usda(food_name):
    """Helper function to query the USDA API for calorie data."""
    params = {
        "query": food_name,
        "pageSize": 1,
        "api_key": USDA_API_KEY
    }
    
    try:
        response = requests.get(USDA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("foods"):
            return {"error": f"No matching foods found in USDA database for '{food_name}'"}
            
        top_food = data["foods"][0]
        description = top_food.get("description", food_name)
        
        calories = "Not found"
        for nutrient in top_food.get("foodNutrients", []):
            if nutrient.get("nutrientName") == "Energy" and nutrient.get("unitName") == "KCAL":
                calories = nutrient.get("value")
                break
                
        return {
            "usda_matched_item": description,
            "calories_per_100g": calories
        }
    except Exception as e:
        return {"error": f"USDA API communication failed: {str(e)}"}


@method_decorator(csrf_exempt, name='dispatch')
class TrackCaloriesView(View):
    """Django View to accept an image, classify it, and fetch USDA metrics."""
    
    def post(self, request, *args, **kwargs):
        # Ensure a file was actually uploaded
        if 'file' not in request.FILES:
            return JsonResponse({"success": False, "error": "No file uploaded under the key 'file'."}, status=400)
            
        uploaded_file = request.FILES['file']
        
        # Simple content type validation
        if not uploaded_file.content_type.startswith("image/"):
            return JsonResponse({"success": False, "error": "Uploaded file must be an image."}, status=400)
            
        try:
            # Process image in-memory using PIL
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Predict the food item using Hugging Face
            predictions = food_classifier(image)
            if not predictions:
                return JsonResponse({"success": False, "error": "Model failed to analyze image."}, status=500)
                
            top_prediction = predictions[0]
            detected_label = top_prediction["label"]
            confidence_score = top_prediction["score"]
            
            # Format the string for USDA searching (e.g., "french_fries" -> "french fries")
            cleaned_food_query = detected_label.replace("_", " ")
            
            # Query USDA
            usda_nutrition = get_calories_from_usda(cleaned_food_query)
            
            return JsonResponse({
                "success": True,
                "classification": {
                    "detected_item": cleaned_food_query,
                    "confidence": round(confidence_score, 4)
                },
                "nutrition_data": usda_nutrition
            })
            
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Internal Server Error: {str(e)}"}, status=500)