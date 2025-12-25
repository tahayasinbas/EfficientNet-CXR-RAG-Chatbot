from rest_framework import serializers
from .models import ChatSession, ChatMessage


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model"""

    class Meta:
        model = ChatMessage
        fields = ['id', 'sender', 'content', 'rag_source', 'rag_confidence', 'created_at']
        read_only_fields = ['id', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model"""

    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['id', 'xray', 'created_at', 'updated_at', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ChatMessageCreateSerializer(serializers.Serializer):
    """Serializer for creating chat messages"""

    session_id = serializers.IntegerField(required=False, allow_null=True)
    xray_id = serializers.IntegerField(required=False, allow_null=True)
    message = serializers.CharField(max_length=2000)

    def validate_message(self, value):
        """Validate message is not empty"""
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value
