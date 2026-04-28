from rest_framework import serializers
from .models import Compte
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.tokens import RefreshToken


# =========================
# REGISTER
# =========================
class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    email = serializers.EmailField()

    class Meta:
        model = Compte
        fields = ['username', 'email', 'password', 'role']

    # ✅ email unique
    def validate_email(self, value):
        if Compte.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value

    # ✅ password fort (Django validator)
    def validate_password(self, value):
        validate_password(value)
        return value

    # ✅ role valide
    def validate_role(self, value):
        roles = ['patient', 'dermatologue', 'admin']
        if value not in roles:
            raise serializers.ValidationError("Invalid role")
        return value

    def create(self, validated_data):
        user = Compte.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            role=validated_data['role']
        )
        return user


# =========================
# LOGIN
# =========================


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        try:
            user = Compte.objects.get(email=data['email'])
        except Compte.DoesNotExist:
            raise serializers.ValidationError("Invalid credentials")

        if not user.check_password(data['password']):
            raise serializers.ValidationError("Invalid credentials")

        return user  # ✅ IMPORTANT

