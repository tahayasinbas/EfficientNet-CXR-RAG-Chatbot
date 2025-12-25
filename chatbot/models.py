from django.db import models
from xray.models import XRay


class ChatSession(models.Model):
    """Model for chat sessions"""

    xray = models.ForeignKey(
        XRay,
        on_delete=models.CASCADE,
        related_name='chat_sessions',
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        xray_id = self.xray.id if self.xray else 'No X-Ray'
        return f"ChatSession #{self.id} - {xray_id}"


class ChatMessage(models.Model):
    """Model for chat messages"""

    SENDER_CHOICES = [
        ('user', 'User'),
        ('ai', 'AI'),
    ]

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )

    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    content = models.TextField()

    # RAG metadata
    rag_source = models.CharField(max_length=255, blank=True, null=True)
    rag_confidence = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.get_sender_display()}: {self.content[:50]}..."
