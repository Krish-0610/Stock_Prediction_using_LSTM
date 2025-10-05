from django.urls import path
from .views import PredictView, LatestPredictionView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='make-prediction'),
    path('predictions/latest/', LatestPredictionView.as_view(), name='latest-prediction'),
]