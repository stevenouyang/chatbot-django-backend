from django.db import models
from wagtail.admin.panels import FieldPanel, MultiFieldPanel, InlinePanel

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
    
    