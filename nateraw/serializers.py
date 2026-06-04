from typing import List
from pydantic import BaseModel, Field
from rest_framework import serializers

# =====================================================================
# 1. PYDANTIC SCHEMAS (Passed to OpenRouter to shape the AI's response)
# =====================================================================

class FoodItemSchema(BaseModel):
    name: str = Field(
        description="The common name of the individual food element discovered on the plate."
    )
    estimated_weight_grams: float = Field(
        description="Estimated weight of this specific item in grams based on visual portion sizing."
    )
    calories: int = Field(
        description="Caloric content calculated for this item's estimated weight."
    )
    protein_g: float = Field(
        description="Protein macronutrient value in grams."
    )
    carbs_g: float = Field(
        description="Total carbohydrate macronutrient value in grams."
    )
    fat_g: float = Field(
        description="Total fat macronutrient value in grams."
    )


class NutritionAnalysisSchema(BaseModel):
    items: List[FoodItemSchema] = Field(
        description="A list containing all the individual components detected in the food image."
    )
    total_calories: int = Field(
        description="The calculated sum of all calories from all items on the plate."
    )
    confidence_score: float = Field(
        description="A value strictly between 0.0 and 1.0 indicating how visually clear and identifiable the food is."
    )


# =====================================================================
# 2. DRF SERIALIZERS (Used by Django to validate data and return clean HTTP responses)
# =====================================================================

class FoodItemSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    estimated_weight_grams = serializers.FloatField()
    calories = serializers.IntegerField()
    protein_g = serializers.FloatField()
    carbs_g = serializers.FloatField()
    fat_g = serializers.FloatField()


class NutritionAnalysisSerializer(serializers.Serializer):
    items = FoodItemSerializer(many=True)
    total_calories = serializers.IntegerField()
    confidence_score = serializers.FloatField()