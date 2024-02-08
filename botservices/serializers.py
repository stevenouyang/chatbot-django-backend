from rest_framework import serializers
from drf_spectacular.utils import extend_schema

class ChatBotSerializer(serializers.Serializer):
    user_input = serializers.CharField()

    class Meta:
        fields = ['user_input']

    def to_internal_value(self, data):
        return super().to_internal_value(data)
