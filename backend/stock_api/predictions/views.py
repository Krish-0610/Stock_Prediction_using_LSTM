# views.py

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import yfinance as yf
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .predictor import predict_stock_prices, get_model_evaluation

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=3)


# ---- Cache Layer ----
@lru_cache(maxsize=20)
def get_cached_data(ticker: str, period: str = "1mo"):
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        return data.reset_index()[["Date", "Close"]]
    except Exception as e:
        logger.error(f"Data fetch failed for {ticker}: {e}")
        return None


# ---- APIs ----
class PredictionAPI(APIView):
    """
    Predict next day's open/close prices for a given ticker.
    """

    def get(self, request):
        ticker = request.query_params.get("ticker", "^NSEI").upper()

        # Predict asynchronously
        future = executor.submit(predict_stock_prices, ticker)
        predictions = future.result()

        if not predictions or "error" in predictions:
            return Response(predictions, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            # Fetch recent 2-day close data
            data = yf.download(ticker, period="2d", interval="1d", progress=False)
            if len(data) >= 2:
                current_price = data["Close"].iloc[-1]
                prev_price = data["Close"].iloc[-2]
                predictions.update(
                    {
                        "current_price": float(current_price),
                        "change_percent": round(
                            ((current_price - prev_price) / prev_price) * 100, 2
                        ),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to fetch current price for {ticker}: {e}")

        return Response(predictions, status=status.HTTP_200_OK)


class ChartDataAPI(APIView):
    """
    Return historical stock data for the specified period, including both Open and Close prices.
    """

    VALID_PERIODS = {"1d": "1d", "5d": "5d", "1m": "1mo", "6m": "6mo", "1y": "1y"}

    def get(self, request, period):
        ticker = request.query_params.get("ticker", "^NSEI").upper()
        if period not in self.VALID_PERIODS:
            return Response(
                {
                    "error": f"Invalid period. Use one of: {', '.join(self.VALID_PERIODS.keys())}"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        yf_period = self.VALID_PERIODS[period]

        try:
            data = yf.download(ticker, period=yf_period, interval="1d", progress=False)
        except Exception as e:
            return Response(
                {"error": f"Failed to fetch data for {ticker}: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if data.empty:
            return Response(
                {"error": "No data available for the given period."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Prepare combined Open and Close dataset
        # Flatten MultiIndex column names (if any)
        data.columns = [
            col[0] if isinstance(col, tuple) else col for col in data.columns
        ]

        # Keep only relevant columns
        data = data.reset_index()[["Date", "Open", "Close"]]
        records = data.to_dict("records")
        return Response(records, status=status.HTTP_200_OK)


class ModelEvaluationAPI(APIView):
    """
    Return R² metrics for the ML model performance.
    """

    def get(self, request):
        ticker = request.query_params.get("ticker", "^NSEI").upper()
        evaluation = get_model_evaluation(ticker=ticker)
        if not evaluation or "error" in evaluation:
            return Response(evaluation, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(evaluation, status=status.HTTP_200_OK)
