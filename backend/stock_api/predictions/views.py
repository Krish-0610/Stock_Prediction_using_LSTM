# views.py

from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import yfinance as yf
from .predictor import predict_stock_prices, get_model_evaluation


class PredictionAPI(APIView):
    """
    API to get the next day's predicted stock prices (Open and Close).
    """

    def get(self, request, *args, **kwargs):
        predictions = predict_stock_prices()
        if "error" in predictions:
            return Response(predictions, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(predictions, status=status.HTTP_200_OK)


class ChartDataAPI(APIView):
    """
    API to get historical stock data for charting.
    Accepts a 'period' parameter: 1d, 5d, 1m, 6m, 1y.
    """

    def get(self, request, period, *args, **kwargs):
        ticker = "^NSEI"
        valid_periods = ["1d", "5d", "1m", "6m", "1y"]

        if period not in valid_periods:
            return Response(
                {"error": "Invalid period. Choose from: 1d, 5d, 1m, 6m, 1y"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        # Calculate start and end dates
        end_date = datetime.now()

        if period.endswith("d"):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        elif period.endswith("m"):
            months = int(period[:-1])
            start_date = end_date - timedelta(days=30 * months)
        elif period.endswith("y"):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        else:
            return Response(
                {"error": "Invalid period format"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )

        if data.empty:
            return Response(
                {"error": "Could not fetch data for the given period."},
                status=status.HTTP_404_NOT_FOUND,
            )

        data.columns = [
            col[0] if isinstance(col, tuple) else col for col in data.columns
        ]

        chart_data = data.reset_index()[["Date", "Close"]].to_dict("records")
        return Response(chart_data, status=status.HTTP_200_OK)


class ModelEvaluationAPI(APIView):
    """
    API to get the R-squared scores for the model's performance.
    """

    def get(self, request, *args, **kwargs):
        evaluation = get_model_evaluation()
        if "error" in evaluation:
            return Response(evaluation, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(evaluation, status=status.HTTP_200_OK)
