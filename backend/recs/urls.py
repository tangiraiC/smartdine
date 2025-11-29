from django.urls import path
from .views import RecommendationView, HealthView

urlpatterns = [
    path("recommendations/", RecommendationView.as_view(), name="recommendations"),
    path("health/", HealthView.as_view(), name="health"),
]
