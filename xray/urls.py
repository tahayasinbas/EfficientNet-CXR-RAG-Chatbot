from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import XRayViewSet

router = DefaultRouter()
router.register(r'xrays', XRayViewSet, basename='xray')

urlpatterns = [
    path('', include(router.urls)),
]
