from rest_framework import serializers

class RecommendationRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField(
        required=False, allow_blank=True,
        help_text="Free-text preferences, e.g., 'vegan tacos'"
    )
    k = serializers.IntegerField(
        required=False, min_value=1, max_value=50, default=10
    )
    weights = serializers.DictField(
        required=False,
        child=serializers.FloatField(),
        help_text="Optional weights, e.g. {'joint_score': 1.0, 'text_similarity': 0.5}"
    )
