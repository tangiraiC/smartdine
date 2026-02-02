from django.urls import path
from .views import RecommendationView, HealthView, restaurant_detail

urlpatterns = [
    path("recommendations/", RecommendationView.as_view(), name="recommendations"),
    path("health/", HealthView.as_view(), name="health"),
    path("business/<str:business_id>/", restaurant_detail, name="restaurant_detail"),
]
