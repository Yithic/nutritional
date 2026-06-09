from pydantic import BaseModel, Field
from typing import List
from rest_framework import serializers



class FoodItemSchema(BaseModel):
    name: str = Field(description="Name of the individual food element")
    estimated_weight_grams: float = Field(description="Estimated weight of the item in grams")
    calories: int = Field(description="Caloric content in kcal")
    protg: float = Field(description="Protein content in grams")
    carbsg: float = Field(description="Total carbohydrate content in grams")
    fatg: float = Field(description="Total fat content in grams")
    fiberg: float = Field(description="Total fiber content in grams")


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
    


class NutritionAnalysisSerializer(serializers.Serializer):
    items = FoodItemSerializer(many=True)
    total_calories = serializers.IntegerField()
    confidence_score = serializers.FloatField()