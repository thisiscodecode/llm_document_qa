from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from documents.views import (
    index, upload_document, list_documents, delete_document, document_status,
    health_live, health_ready,
)


urlpatterns = [
    path('', index, name='index'),
    path('health/live/', health_live, name='health-live'),
    path('health/ready/', health_ready, name='health-ready'),
    path('upload/', upload_document, name='upload-document'),
    path('documents/', list_documents, name='list-documents'),
    path('documents/<int:doc_id>/delete/', delete_document, name='delete-document'),
    path('documents/<int:doc_id>/status/', document_status, name='document-status'),
    path('admin/', admin.site.urls),
    path('accounts/login/', auth_views.LoginView.as_view(
        template_name='registration/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('api/', include('documents.urls')),
]
