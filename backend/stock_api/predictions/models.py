from django.db import models

# Create your models here.

class Prediction(models.Model):
    ticker = models.CharField(max_length=10)
    predicted_close_price = models.FloatField()
    predicted_open_price = models.FloatField()
    prediction_date = models.DateField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.ticker} on {self.prediction_date}"