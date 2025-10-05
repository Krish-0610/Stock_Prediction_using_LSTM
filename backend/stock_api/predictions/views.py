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
    Accepts a 'ticker' query parameter.
    """

    def get(self, request, *args, **kwargs):
        ticker = request.query_params.get('ticker', '^NSEI')
        predictions = predict_stock_prices(ticker=ticker)
        if "error" in predictions:
            return Response(predictions, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Fetch latest data for current price and change
        try:
            latest_data = yf.download(ticker, period="2d", interval="1d")
            if not latest_data.empty and len(latest_data) >= 2:
                current_price = latest_data['Close'].iloc[-1]
                previous_price = latest_data['Close'].iloc[-2]
                change = ((current_price - previous_price) / previous_price) * 100
                predictions['current_price'] = current_price
                predictions['change'] = change
        except Exception as e:
            # If fetching latest data fails, proceed without it
            print(f"Could not fetch latest data for ticker {ticker}: {e}")

        return Response(predictions, status=status.HTTP_200_OK)


class ChartDataAPI(APIView):
    """
    API to get historical stock data for charting.
    Accepts a 'period' parameter: 1d, 5d, 1m, 6m, 1y.
    """

    def get(self, request, period, *args, **kwargs):
        ticker = request.query_params.get('ticker', '^NSEI')
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
        ticker = request.query_params.get('ticker', '^NSEI')
        evaluation = get_model_evaluation(ticker=ticker)
        if "error" in evaluation:
            return Response(evaluation, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(evaluation, status=status.HTTP_200_OK)
