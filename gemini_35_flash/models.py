from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

class MealLog(models.Model):
    meal_name = models.CharField(max_length=255)
    calories = models.IntegerField()
    protein_g = models.FloatField(default=0.0)
    carbs_g = models.FloatField(default=0.0)
    fat_g = models.FloatField(default=0.0)
    fiber_g=models.FloatField(default=0.0)
    logged_at = models.DateTimeField(default=timezone.now)
    time=models.FloatField(default=0.0)
    
    # New Token Tracking Fields
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.meal_name} - {self.calories} kcal"