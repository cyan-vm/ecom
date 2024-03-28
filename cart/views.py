# from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
from .cart import Cart
from store.models import Product
from django.http import JsonResponse
# from django.contrib import messages

def cart_summary(request):
  # # Get the cart
  cart = Cart(request)
  cart_products = cart.get_prods
  quantities = cart.get_quants
  # totals = cart.cart_total()
  return render(request, "cart_summary.html", {'cart_products': cart_products, 'quantities': quantities})

def cart_add(request):
  # Get the cart
  cart = Cart(request)
  # test for POST
  # print(f"Cart session : {cart.session} Cart request {cart.request} Cart session key : {cart.cart}")
  if request.POST.get('action') == 'post':
    # Get stuff
    product_id = int(request.POST.get('product_id'))
    product_qty = int(request.POST.get('product_qty'))
    # lookup product in DB
    product = get_object_or_404(Product, id=product_id)
    # Save to session
    cart.add(product=product, quantity=product_qty)
    # Get Cart Quantity
    cart_quantity = cart.__len__()
    # Return resonse
    # response = JsonResponse({'Product Name: ': product.name})
    # response = JsonResponse({'Product Name': product.name})
    response = JsonResponse({'quantity : ': cart_quantity})
    # messages.success(request, ("Product Added To Cart..."))
    return response 

# session_k = Session.objects.get(pk='oydztu6mriz8mtd95nd9sfga7v0071ng')

# session_k.get_decoded()

# Create your views here.
