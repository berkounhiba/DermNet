from django.db import models
from django.contrib.auth.models import AbstractUser

class Compte(AbstractUser):

    ROLE_CHOICES = (
        ('patient', 'Patient'),
        ('dermatologue', 'Dermatologue'),
        ('admin', 'Admin'),
    )
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    def __str__(self):
        return self.username
