from django.db import models
from wagtail.admin.panels import FieldPanel, MultiFieldPanel, InlinePanel
from modelcluster.models import ClusterableModel
from modelcluster.fields import ParentalKey

# Create your models here.
class ChatLog(models.Model):
    user_chat = models.CharField(max_length=255, blank=True, null=True)
    bot_response = models.CharField(max_length=255, blank=True, null=True)
    prob = models.FloatField(null=True, blank=True)
    date_time = models.DateTimeField(auto_now_add=True)
    
    panels = [
        FieldPanel("user_chat"),
        FieldPanel("bot_response"),
        FieldPanel("prob"),
    ]
    
    def __str__(self):
        return self.user_chat
    
    
class ChatIntents(ClusterableModel):
    tag =models.CharField(max_length=255, blank=True, null=True)
    
    panels = [
        FieldPanel("tag"),
        InlinePanel("intents_pattern", label="itents patterns"),
        InlinePanel("intents_responses", label="intents responses"),
    ]
    
    def __str__(self):
        return self.tag


class IntentsPattern(models.Model):
    parent      = ParentalKey(ChatIntents, related_name="intents_pattern", blank=True, null=True)
    pattern     = models.CharField(max_length=255)
    
class IntentsResponses(models.Model):
    parent      = ParentalKey(ChatIntents, related_name="intents_responses", blank=True, null=True)
    responses   = models.CharField(max_length=255)