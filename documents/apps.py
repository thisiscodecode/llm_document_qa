from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    name = 'documents'

    def ready(self):
        from .ingestion.document_processor import get_task_queue
        get_task_queue().start()
