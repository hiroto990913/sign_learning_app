from django.urls import path

from . import views

app_name = 'syuwa'
urlpatterns = [
    path('', views.index, name='index'),
    path('answer/', views.gazou, name='gazou'),
    path('predict/', views.answer, name='answer'),
    ]
    