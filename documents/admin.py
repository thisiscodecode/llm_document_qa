from django.contrib import admin
from django.core.exceptions import ValidationError
from .models import Document, DocumentChunk, QuestionHistory, ChatSession


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    can_delete = False
    readonly_fields = ('content', 'chunk_index', 'page_number')

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'owner', 'created_at', 'processed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('title', 'full_text')
    readonly_fields = ('full_text', 'created_at', 'updated_at', 'processed_at')
    inlines = [DocumentChunkInline]
    actions = ['reprocess_documents']

    def get_actions(self, request):
        actions = super().get_actions(request)
        actions.pop('delete_selected', None)
        return actions

    def reprocess_documents(self, request, queryset):
        from .ingestion.document_processor import process_document
        for doc in queryset:
            process_document(doc.id)
        self.message_user(request, f"Reprocessing {queryset.count()} documents.")
    reprocess_documents.short_description = "Reprocess selected documents"

    def delete_model(self, request, obj):
        from .indexing.vector_index import delete_vectors_for_document
        from .ingestion.document_processor import document_lock

        with document_lock(obj.id):
            if not delete_vectors_for_document(obj.id):
                raise ValidationError('Chroma cleanup failed; the document was not deleted')
            super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        from .indexing.vector_index import delete_vectors_for_document
        from .ingestion.document_processor import document_lock

        # Delete one complete Chroma/SQL pair at a time. If a later cleanup
        # fails, earlier documents are already gone and the remaining ones
        # still have their vectors.
        for obj in list(queryset):
            with document_lock(obj.id):
                if not delete_vectors_for_document(obj.id):
                    raise ValidationError('Chroma cleanup failed; remaining documents were not deleted')
                obj.delete()


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('document', 'chunk_index')
    search_fields = ('content',)
    list_filter = ('document',)
    readonly_fields = ('document', 'content', 'chunk_index', 'page_number')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class QuestionHistoryInline(admin.TabularInline):
    model = QuestionHistory
    extra = 0
    readonly_fields = ('question', 'answer', 'created_at')
    fields = ('question', 'answer', 'created_at')


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('title', 'owner', 'created_at', 'updated_at')
    search_fields = ('title',)
    list_filter = ('created_at',)
    inlines = [QuestionHistoryInline]


@admin.register(QuestionHistory)
class QuestionHistoryAdmin(admin.ModelAdmin):
    list_display = ('question', 'session', 'owner', 'created_at')
    search_fields = ('question', 'answer', 'retrieved_context')
    readonly_fields = ('created_at',)
    list_filter = ('created_at', 'session')
