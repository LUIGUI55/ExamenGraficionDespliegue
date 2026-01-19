from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/emotion', views.predict_emotion, name='predict_emotion'),
    path('predict/tumor', views.predict_tumor, name='predict_tumor'),
]
