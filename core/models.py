from django.db import models

class FinancialData(models.Model):
    date = models.DateTimeField()
    symbol = models.CharField(max_length=10)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.IntegerField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"