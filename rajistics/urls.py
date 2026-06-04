from django.urls import path
from .views import TrackCaloriesView

urlpatterns = [
    path('track-calories/', TrackCaloriesView.as_view(), name='track_calories'),
]