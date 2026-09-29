from django.contrib import admin
from django.urls import path, include
from documents.views import index, upload_document, list_documents, delete_document, document_status


urlpatterns = [
    path('', index, name='index'),
    path('upload/', upload_document, name='upload-document'),
    path('documents/', list_documents, name='list-documents'),
    path('documents/<int:doc_id>/delete/', delete_document, name='delete-document'),
    path('documents/<int:doc_id>/status/', document_status, name='document-status'),
    path('admin/', admin.site.urls),
    path('api/', include('documents.urls')),
]
