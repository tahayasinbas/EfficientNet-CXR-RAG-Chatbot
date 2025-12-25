from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatViewSet

router = DefaultRouter()
router.register(r'sessions', ChatViewSet, basename='chat')

urlpatterns = [
    path('', include(router.urls)),
    path('send/', ChatViewSet.as_view({'post': 'send'}), name='chat-send'),
]
