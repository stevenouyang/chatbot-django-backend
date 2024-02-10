from django.urls import path
from .views import ChatBotView, Home, ChatBotViewTemplate

urlpatterns = [
    path('bot/chatbot/', ChatBotView.as_view(), name='chatbot'),
    path('', Home, name='home'),
    path('bot/chat/', ChatBotViewTemplate.as_view(), name='chatbot-view'),
]
