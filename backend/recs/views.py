from django.shortcuts import render

# Create your views here.

# backend/recs/views.py
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_http_methods

@require_GET
def health(_):
    return JsonResponse({"status": "ok"})

@require_http_methods(["POST"])
def recommendations(request):
    # Week-1 stub: return an empty list in the agreed shape
    return JsonResponse({"k": 10, "model_version": "stub-0.1", "results": []})

@require_GET
def restaurant_detail(request, business_id: str):
    # Week-1 stub: not seeded yet
    return JsonResponse({"detail": "Not found"}, status=404)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, throttling

from .serializers import RecommendationRequestSerializer
from .services import rank_candidates

"""
class RecommendationRateThrottle(throttling.UserRateThrottle):
    scope = "recommendations"

"""
class RecommendationView(APIView):
    permission_classes = [permissions.AllowAny]
    #throttle_classes = [RecommendationRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = RecommendationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prefs = serializer.validated_data

        results = rank_candidates(prefs, k=prefs.get("k", 10))
        return Response({"results": results}, status=status.HTTP_200_OK)
    


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

class HealthView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)



