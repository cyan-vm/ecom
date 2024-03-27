from django.shortcuts import render
from .models import Product

def home(request):
    products = Product.objects.all()
    # print(type(Product))
    # products = Product.objects.all()
    # Your view logic here
    return render(request, 'home.html', {'products': products})

def about(request):
    return render(request, 'about.html')

