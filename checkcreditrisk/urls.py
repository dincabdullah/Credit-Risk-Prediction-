from django.urls import path
from . import views


urlpatterns = [
    path('index/', views.index, name="check_credit_risk_index_page_link")
]
