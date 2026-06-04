from django.urls import path
from gemini_35_flash_sam.views import AnalyzeMealView

urlpatterns = [
    path('track-calories', AnalyzeMealView.as_view(), name='analyze-meal'),
]