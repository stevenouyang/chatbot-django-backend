from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import random
import json
import torch
from .bot.model import NeuralNet
from .bot.nltk_utils import bag_of_words, tokenize
from django.conf import settings
import os
from drf_spectacular.utils import OpenApiParameter
from drf_spectacular.utils import extend_schema_view
from drf_spectacular.utils import extend_schema
from .serializers import ChatBotSerializer
from django.db import transaction
from .models import ChatLog  

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load model and intents data
FILE_PATH = os.path.join(settings.BASE_DIR,"botservices", "bot", "data.pth")
data = torch.load(FILE_PATH)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


INTENTS_FILE_PATH = os.path.join(settings.BASE_DIR, "botservices", "bot", "train-intents.json")

with open(INTENTS_FILE_PATH, 'r') as f:
    intents = json.load(f)

bot_name = "Astralis Bot"


class ChatBotView(APIView):
    """
    ChatBot API

    ---
    parameters:
      - name: user_input
        description: User input for the chatbot
        required: true
        type: string
        paramType: form
    """

    @extend_schema(request=ChatBotSerializer)
    def post(self, request, format=None):
        serializer = ChatBotSerializer(data=request.data)
        if serializer.is_valid():
            user_input = serializer.validated_data['user_input']

            sentence = tokenize(user_input)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X)
            X = X.to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            with transaction.atomic():
                chat_log = ChatLog.objects.create(
                    user_chat=user_input,
                    bot_response="",
                    prob=prob.item()
                )

            if prob.item() > 0.7:
                for intent in intents["intents"]:
                    if tag == intent["tag"]:
                        bot_response = random.choice(intent['responses'])
                        chat_log.bot_response = bot_response
                        chat_log.save()
                        return Response({'response': f"{bot_name}: {bot_response}"}, status=status.HTTP_200_OK)

            else:
                bot_response = "Maaf saya tidak bisa menjawab pertanyaan ini, anda dapat langsung menghubungi kami melalui Whatsapp 0855-8830-051"
                chat_log.bot_response = bot_response
                chat_log.save()
                return Response({'response': f"{bot_name}: {bot_response}"}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)