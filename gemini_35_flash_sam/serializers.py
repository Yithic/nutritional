from pydantic import BaseModel, Field
from typing import List
from rest_framework import serializers


class MicroNutrientSchema(BaseModel):
    nutrient_name: str = Field(description="The name of the micro or vitamin (e.g., Sodium, Vitamin C)")
    amount: str = Field(description="The value with unit string (e.g., 200mg, 15mcg)")

class FoodItemSchema(BaseModel):
    name: str = Field(description="Name of the individual food element")
    estimated_weight_grams: float = Field(description="Estimated weight of the item in grams")
    calories: int = Field(description="Caloric content in kcal")
    protein_g: float = Field(description="Protein content in grams")
    carbs_g: float = Field(description="Total carbohydrate content in grams")
    fat_g: float = Field(description="Total fat content in grams")
    fiber_g: float = Field(description="Total fiber content in grams")
    micronutrients: List[MicroNutrientSchema] = Field(
        description="A dynamic list of key-value tracking pairs for elements not covered in standard macros."
    )


class NutritionAnalysisSchema(BaseModel):
    items: List[FoodItemSchema] = Field(description="List of detected food components")
    total_calories: int = Field(description="Sum of all calculated calories on the plate")
    confidence_score: float = Field(description="A value from 0.0 to 1.0 indicating model clarity")



class FoodItemSerializer(serializers.Serializer):
    name = serializers.CharField()
    estimated_weight_grams = serializers.FloatField()
    calories = serializers.IntegerField()
    protein_g = serializers.FloatField()
    carbs_g = serializers.FloatField()
    fat_g = serializers.FloatField()
    fiber_g = serializers.FloatField()
    micronutrients = serializers.ListField()


class NutritionAnalysisSerializer(serializers.Serializer):
    items = FoodItemSerializer(many=True)
    total_calories = serializers.IntegerField()
    confidence_score = serializers.FloatField()