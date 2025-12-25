from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone

from .models import XRay, Diagnosis
from .serializers import XRaySerializer, XRayCreateSerializer
from .services import get_prediction_service


class XRayViewSet(viewsets.ModelViewSet):
    """
    ViewSet for X-Ray operations

    Endpoints:
    - GET /api/xrays/ - List all X-Rays
    - POST /api/xrays/ - Upload new X-Ray
    - GET /api/xrays/{id}/ - Get specific X-Ray
    - POST /api/xrays/{id}/analyze/ - Analyze X-Ray
    """

    queryset = XRay.objects.all()
    serializer_class = XRaySerializer

    def get_serializer_class(self):
        """Use different serializer for creation"""
        if self.action == 'create':
            return XRayCreateSerializer
        return XRaySerializer

    def create(self, request, *args, **kwargs):
        """Upload new X-Ray image"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        xray = serializer.save()

        # Return full serializer with all fields
        output_serializer = XRaySerializer(xray)

        return Response(
            output_serializer.data,
            status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=['post'])
    def analyze(self, request, pk=None):
        """
        Analyze X-Ray image using the AI model

        POST /api/xrays/{id}/analyze/
        """
        xray = self.get_object()

        # Check if already analyzed
        if xray.is_analyzed:
            return Response(
                {
                    'message': 'X-Ray already analyzed',
                    'xray': XRaySerializer(xray).data
                },
                status=status.HTTP_200_OK
            )

        try:
            # Get prediction service
            predictor = get_prediction_service()

            # Run prediction
            predictions = predictor.predict(
                image_path=xray.image.path,
                age=xray.age,
                gender=xray.gender,
                position=xray.position
            )

            # Save all predictions to database
            for pred in predictions:
                Diagnosis.objects.create(
                    xray=xray,
                    disease_name=pred['disease_name'],
                    confidence=pred['confidence'],
                    risk_level=pred['risk_level']
                )

            # Update X-Ray status
            xray.is_analyzed = True
            xray.analyzed_at = timezone.now()
            xray.save()

            # Return results
            return Response(
                {
                    'message': 'Analysis completed successfully',
                    'xray': XRaySerializer(xray).data,
                    'all_predictions': predictions
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    'error': f'Analysis failed: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
