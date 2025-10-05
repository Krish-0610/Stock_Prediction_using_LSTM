from django.urls import path
from .views import ChartDataAPI, ModelEvaluationAPI, PredictionAPI

urlpatterns = [
    path('predict/', PredictionAPI.as_view(), name='predict-price'),
    path('chart/<str:period>/', ChartDataAPI.as_view(), name='chart-data'),
    path('evaluate/', ModelEvaluationAPI.as_view(), name='evaluate-model'),
]