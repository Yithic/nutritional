from django.urls import path
from .views import AnalyzeMealView

urlpatterns = [
    path('analyze-meal/', AnalyzeMealView.as_view(), name='analyze-meal'),
]