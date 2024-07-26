from django.shortcuts import render
from .models import FinancialData

def home(request):
    data = FinancialData.objects.all()[:10]  # Merr 10 të dhënat e fundit
    return render(request, 'core/home.html', {'data': data})


