from django.urls import path
from .views import register, login, logout, verify_token
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('register/', register),
    path('login/', login),
    path('logout/', logout),
    path('verify/', verify_token),

    # refresh token
    path('token/refresh/', TokenRefreshView.as_view()),
]