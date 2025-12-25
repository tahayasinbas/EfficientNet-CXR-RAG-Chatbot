from rest_framework import serializers
from .models import XRay, Diagnosis


class DiagnosisSerializer(serializers.ModelSerializer):
    """Serializer for Diagnosis model"""

    percentage = serializers.ReadOnlyField()

    class Meta:
        model = Diagnosis
        fields = ['id', 'disease_name', 'confidence', 'percentage', 'risk_level', 'created_at']
        read_only_fields = ['id', 'created_at']


class XRaySerializer(serializers.ModelSerializer):
    """Serializer for XRay model"""

    diagnoses = DiagnosisSerializer(many=True, read_only=True)

    class Meta:
        model = XRay
        fields = [
            'id',
            'age',
            'gender',
            'position',
            'image',
            'uploaded_at',
            'analyzed_at',
            'is_analyzed',
            'diagnoses'
        ]
        read_only_fields = ['id', 'uploaded_at', 'analyzed_at', 'is_analyzed']

    def validate_age(self, value):
        """Validate age is within reasonable range"""
        if value < 0 or value > 120:
            raise serializers.ValidationError("Age must be between 0 and 120")
        return value

    def validate_image(self, value):
        """Validate image file"""
        # Check file size (max 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image file size must be less than 10MB")

        # Check file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
        if value.content_type not in allowed_types:
            raise serializers.ValidationError(
                f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )

        return value


class XRayCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating XRay (without diagnoses)"""

    class Meta:
        model = XRay
        fields = ['age', 'gender', 'position', 'image']
