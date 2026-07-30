from rest_framework import serializers
from .models import Conversation, Message

class ParticipantSerializer(serializers.Serializer): 
    id = serializers.IntegerField()
    name = serializers.CharField()
    email = serializers.EmailField()
    
class ConversationSerializer(serializers.ModelSerializer):
    last_message_preview = serializers.SerializerMethodField()
    last_message_at = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()
    participants = ParticipantSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id',
            'title',
            'type',
            'last_message_preview',
            'last_message_at',
            'unread_count',
            'participants',
        ]

    def get_last_message_preview(self, obj):
        last_message = obj.messages.order_by('-created_at').first()
        if not last_message:
            return None
        return (last_message.content or '')[:120]

    def get_last_message_at(self, obj):
        last_message = obj.messages.order_by('-created_at').first()
        if not last_message:
            return None
        return last_message.created_at

    def get_unread_count(self, obj):
        # Placeholder for now (can be improved with read receipts table)
        return 0


class MessageSerializer(serializers.ModelSerializer):
    conversation_id = serializers.IntegerField(source='conversation.id', read_only=True)
    sender_id = serializers.IntegerField(source='sender.id', read_only=True)
    sender_name = serializers.SerializerMethodField()  
    
    class Meta:
        model = Message
        fields = [
            'id',
            'conversation_id',
            'sender_id',
            'sender_name',
            'content',
            'priority',
            'created_at',
        ]

    def get_sender_name(self, obj):  
        sender = obj.sender
        if hasattr(sender, 'name') and sender.name:
            return sender.name
        if hasattr(sender, 'username') and sender.username:
            return sender.username
        return sender.email or 'Unknown'


class MessageCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['content', 'priority']

    def validate_priority(self, value):
        if value not in ['normal', 'urgent']:
            raise serializers.ValidationError("priority must be 'normal' or 'urgent'")
        return value

class ConversationCreateSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    type = serializers.ChoiceField(choices=['team', 'mission', 'direct'], default='direct')
    participant_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        default=list,
    )    