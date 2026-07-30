from rest_framework import generics, permissions
from rest_framework.exceptions import PermissionDenied, AuthenticationFailed
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer, MessageCreateSerializer, ConversationCreateSerializer



def get_django_user(request):
  
    user = request.user
    if hasattr(user, 'django_user') and user.django_user:
        return user.django_user
    raise AuthenticationFailed("No Django user associated with this token")


class ConversationCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = get_django_user(request)
        serializer = ConversationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        conversation = Conversation.objects.create(
            title=serializer.validated_data['title'],
            type=serializer.validated_data.get('type', 'direct'),
        )
        conversation.participants.add(user)

        # أضف participants إضافيين إذا أُرسلوا
        participant_ids = serializer.validated_data.get('participant_ids', [])
        for pid in participant_ids:
            try:
                from authApp.models import User
                conversation.participants.add(User.objects.get(id=pid))
            except Exception:
                pass

        return Response(ConversationSerializer(conversation).data, status=201)




class ConversationListView(generics.ListAPIView):
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = get_django_user(self.request)
        return Conversation.objects.filter(participants=user).distinct()


class ConversationMessagesView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return MessageCreateSerializer
        return MessageSerializer

    def _get_conversation_or_403(self):
        user = get_django_user(self.request)
        conversation_id = self.kwargs['conversation_id']

        try:
            conversation = Conversation.objects.get(id=conversation_id)
        except Conversation.DoesNotExist:
            raise PermissionDenied("Conversation not found")

        if not conversation.participants.filter(id=user.id).exists():
            raise PermissionDenied("You are not a participant of this conversation")

        return conversation

    def list(self, request, *args, **kwargs):
        conversation = self._get_conversation_or_403()
        
        # ← pagination
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))
        offset = (page - 1) * page_size

        queryset = Message.objects.filter(
            conversation=conversation
        ).order_by('-created_at')[offset:offset + page_size]  

        messages = list(reversed(queryset))  
        serializer = MessageSerializer(messages, many=True)
        
        total = Message.objects.filter(conversation=conversation).count()
        
        return Response({
            'results': serializer.data,
            'total': total,
            'page': page,
            'has_more': offset + page_size < total,
        })

    def create(self, request, *args, **kwargs):
        user = get_django_user(self.request)
        conversation = self._get_conversation_or_403()

        serializer = MessageCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        message = Message.objects.create(
            conversation=conversation,
            sender=user,
            content=serializer.validated_data.get('content'),
            priority=serializer.validated_data.get('priority', 'normal'),
        )
        return Response(MessageSerializer(message).data, status=201)
    
class UserListView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        from authApp.models import User
        current_user = get_django_user(request)
        users = User.objects.exclude(id=current_user.id)
        data = [{'id': u.id, 'name': u.name, 'email': u.email} for u in users]
        return Response(data)    