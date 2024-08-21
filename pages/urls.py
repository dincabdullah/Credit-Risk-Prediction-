from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('index/', views.index),
    path('checkcreditrisk/', views.checkcreditrisk),
    path('blog/', views.blog),
    path('about/', views.about),
]
