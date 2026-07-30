from django.urls import path
from .views import ConversationCreateView, ConversationListView, ConversationMessagesView, UserListView

urlpatterns = [
    path('conversations/', ConversationListView.as_view(), name='conversations-list'),
    path('conversations/create/', ConversationCreateView.as_view()),
    path(
        'conversations/<int:conversation_id>/messages/',
        ConversationMessagesView.as_view(),
        name='conversation-messages',
    ),
    path('users/', UserListView.as_view()),
]