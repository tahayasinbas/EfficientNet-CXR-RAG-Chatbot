from django.contrib import admin
from .models import ChatSession, ChatMessage


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    """Admin interface for ChatSession model"""

    list_display = ['id', 'xray', 'created_at', 'updated_at']
    list_filter = ['created_at']
    search_fields = ['id', 'xray__id']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """Admin interface for ChatMessage model"""

    list_display = ['id', 'session', 'sender', 'content_preview', 'created_at']
    list_filter = ['sender', 'created_at']
    search_fields = ['content', 'session__id']
    readonly_fields = ['created_at']

    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content

    content_preview.short_description = 'Content'
