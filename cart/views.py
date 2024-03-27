# from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
# from .cart import Cart
# from store.models import Product
# from django.http import JsonResponse
# from django.contrib import messages

def cart_summary(request):
	# # Get the cart
	# cart = Cart(request)
	# cart_products = cart.get_prods
	# quantities = cart.get_quants
	# totals = cart.cart_total()
	return render(request, "cart_summary.html", {})



# Create your views here.
