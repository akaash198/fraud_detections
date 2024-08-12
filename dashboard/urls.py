from django.urls import path
from . import views

urlpatterns = [
    path("", views.kpi_metrics, name="kpi_metrics"),
    path('search/', views.search_customer, name='search_customer'),
    path('predict/', views.predict_fraud, name='predict_fraud'),  # Add this line
]
