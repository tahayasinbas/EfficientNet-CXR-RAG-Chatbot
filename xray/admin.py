from django.contrib import admin
from .models import XRay, Diagnosis


@admin.register(XRay)
class XRayAdmin(admin.ModelAdmin):
    """Admin interface for XRay model"""

    list_display = ['id', 'age', 'gender', 'position', 'is_analyzed', 'uploaded_at']
    list_filter = ['gender', 'position', 'is_analyzed']
    search_fields = ['id']
    readonly_fields = ['uploaded_at', 'analyzed_at']


@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    """Admin interface for Diagnosis model"""

    list_display = ['id', 'xray', 'disease_name', 'confidence', 'risk_level', 'created_at']
    list_filter = ['risk_level', 'disease_name']
    search_fields = ['disease_name', 'xray__id']
    readonly_fields = ['created_at']
