from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import ChatSession, ChatMessage
from .serializers import (
    ChatSessionSerializer,
    ChatMessageSerializer,
    ChatMessageCreateSerializer,
    AutomaticRAGQuerySerializer
)
from .services import get_chatbot_service
from xray.models import XRay
from xray.serializers import DiagnosisSerializer


class ChatViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Chat operations

    Endpoints:
    - GET /api/chat/sessions/ - List all chat sessions
    - POST /api/chat/sessions/ - Create new chat session
    - GET /api/chat/sessions/{id}/ - Get specific chat session
    - POST /api/chat/send/ - Send message and get AI response
    """

    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer

    @action(detail=False, methods=['post'])
    def send(self, request):
        """
        Send a message and get AI response

        POST /api/chat/send/
        Body: {
            "session_id": 1 (optional),
            "xray_id": 1 (optional),
            "message": "What is the treatment?"
        }
        """
        serializer = ChatMessageCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        message_text = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id')
        xray_id = serializer.validated_data.get('xray_id')

        # Get or create session
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id)
            except ChatSession.DoesNotExist:
                return Response(
                    {'error': 'Chat session not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            # Create new session
            xray = None
            if xray_id:
                try:
                    xray = XRay.objects.get(id=xray_id)
                except XRay.DoesNotExist:
                    pass

            session = ChatSession.objects.create(xray=xray)

        # Save user message
        user_message = ChatMessage.objects.create(
            session=session,
            sender='user',
            content=message_text
        )

        # Prepare context
        context = self._prepare_context(session)

        # Get AI response from RAG system
        chatbot = get_chatbot_service()
        ai_response = chatbot.get_response(
            question=message_text,
            context=context,
            session_id=session.id
        )

        # Save AI message
        ai_message = ChatMessage.objects.create(
            session=session,
            sender='ai',
            content=ai_response['content'],
            rag_source=ai_response.get('source'),
            rag_confidence=ai_response.get('confidence'),
            rag_query=ai_response.get('rag_query'),
            rag_query_source=ai_response.get('rag_query_source'),
            rag_query_metadata=ai_response.get('rag_query_metadata')
        )

        # Return response
        return Response({
            'session_id': session.id,
            'user_message': ChatMessageSerializer(user_message).data,
            'ai_message': ChatMessageSerializer(ai_message).data,
            'success': ai_response.get('success', True)
        }, status=status.HTTP_200_OK)

    @action(detail=False, methods=['post'])
    def automatic(self, request):
        """
        Build a deterministic classifier-guided RAG query and get AI response.

        POST /api/chat/automatic/
        Body: {
            "session_id": 1 (optional),
            "xray_id": 1
        }
        """
        serializer = AutomaticRAGQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data.get('session_id')
        xray_id = serializer.validated_data['xray_id']

        try:
            xray = XRay.objects.get(id=xray_id)
        except XRay.DoesNotExist:
            return Response(
                {'error': 'X-Ray not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        if not xray.is_analyzed:
            return Response(
                {'error': 'X-Ray must be analyzed before automatic RAG query construction'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id)
            except ChatSession.DoesNotExist:
                return Response(
                    {'error': 'Chat session not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            if session.xray and session.xray_id != xray.id:
                return Response(
                    {'error': 'Chat session belongs to a different X-Ray'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if not session.xray:
                session.xray = xray
                session.save(update_fields=['xray', 'updated_at'])
        else:
            session = ChatSession.objects.create(xray=xray)

        context = self._prepare_context(session)
        chatbot = get_chatbot_service()

        try:
            query_metadata = chatbot.build_automatic_query(context=context)
        except ValueError as exc:
            return Response(
                {'error': str(exc)},
                status=status.HTTP_400_BAD_REQUEST
            )

        generated_query_message = ChatMessage.objects.create(
            session=session,
            sender='user',
            content=query_metadata['query'],
            rag_query=query_metadata['query'],
            rag_query_source=query_metadata['query_source'],
            rag_query_metadata=query_metadata
        )

        ai_response = chatbot.get_response(
            question=query_metadata['query'],
            context=context,
            session_id=session.id,
            query_metadata=query_metadata
        )

        ai_message = ChatMessage.objects.create(
            session=session,
            sender='ai',
            content=ai_response['content'],
            rag_source=ai_response.get('source'),
            rag_confidence=ai_response.get('confidence'),
            rag_query=ai_response.get('rag_query'),
            rag_query_source=ai_response.get('rag_query_source'),
            rag_query_metadata=ai_response.get('rag_query_metadata')
        )

        return Response({
            'session_id': session.id,
            'generated_query_message': ChatMessageSerializer(generated_query_message).data,
            'ai_message': ChatMessageSerializer(ai_message).data,
            'rag_query_metadata': query_metadata,
            'success': ai_response.get('success', True)
        }, status=status.HTTP_200_OK)

    def _prepare_context(self, session):
        """Prepare context for RAG system"""
        context = {}

        if session.xray:
            xray = session.xray
            context['patient'] = {
                'age': xray.age,
                'gender': xray.get_gender_display(),
                'position': xray.get_position_display()
            }

            if xray.is_analyzed:
                diagnoses = xray.diagnoses.all()
                context['diagnoses'] = DiagnosisSerializer(diagnoses, many=True).data

        return context
