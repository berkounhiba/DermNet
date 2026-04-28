from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import RegisterSerializer, LoginSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication

# REGISTER
@api_view(['POST'])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "User created"})
    return Response(serializer.errors, status=400)


# LOGIN
@api_view(['POST'])
def login(request):
    serializer = LoginSerializer(data=request.data)

    if serializer.is_valid():
        user = serializer.validated_data

        refresh = RefreshToken.for_user(user)

        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "role": user.role
        })

    return Response(serializer.errors, status=400)


# LOGOUT (BLACKLIST)
@api_view(['POST'])
def logout(request):
    refresh_token = request.data.get("refresh")

    if not refresh_token:
        return Response({"error": "Refresh token required"}, status=400)

    try:
        token = RefreshToken(refresh_token)
        token.blacklist()

        return Response({"message": "Logout successful"})
    except Exception as e:
        return Response({"error": "Invalid token"}, status=400)


# VERIFY TOKEN (OPTIONNEL)
@api_view(['POST'])
def verify_token(request):
    try:
        token = request.data.get("token")

        jwt_auth = JWTAuthentication()
        jwt_auth.get_validated_token(token)

        return Response({"valid": True})
    except Exception:
        return Response({"valid": False}, status=400)
