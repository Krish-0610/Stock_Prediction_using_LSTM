from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from .models import Prediction
from .serializers import PredictionSerializer
from . import predictor

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        ticker = request.data.get('ticker', '^NSEI')
        if not ticker:
            return Response({"error": "Ticker not provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            prediction_results = predictor.make_prediction(ticker)
            
            prediction = Prediction.objects.create(
                ticker=ticker,
                predicted_close_price=prediction_results['predicted_close'],
                predicted_open_price=prediction_results['predicted_open']
            )
            
            serializer = PredictionSerializer(prediction)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:

            print(f"Error during prediction: {e}") 
            return Response({"error": "An error occurred during prediction."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LatestPredictionView(generics.ListAPIView):
    serializer_class = PredictionSerializer

    def get_queryset(self):
        ticker = self.request.query_params.get('ticker', '^NSEI')
        queryset = Prediction.objects.filter(ticker=ticker).order_by('-created_at')[:1]
        return queryset