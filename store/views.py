# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),
# ]

# views.py
from django.shortcuts import render

def home(request):
    # Your view logic here
    return render(request, 'home.html')

