from rest_framework.permissions import BasePermission

class IsPatient(BasePermission):
    def has_permission(self, request, view):
        return request.user.role == "patient"


class IsDermatologue(BasePermission):
    def has_permission(self, request, view):
        return request.user.role == "dermatologue"


class IsAdmin(BasePermission):
    def has_permission(self, request, view):
        return request.user.role == "admin"