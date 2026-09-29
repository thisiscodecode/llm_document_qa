from django.contrib import admin
from .models import Document, DocumentChunk, QuestionHistory, ChatSession


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ('content', 'chunk_index')


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'owner', 'created_at', 'processed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('title', 'full_text')
    readonly_fields = ('full_text', 'created_at', 'updated_at', 'processed_at')
    inlines = [DocumentChunkInline]
    actions = ['reprocess_documents']

    def reprocess_documents(self, request, queryset):
        from .ingestion.document_processor import process_document
        for doc in queryset:
            process_document(doc.id)
        self.message_user(request, f"Reprocessing {queryset.count()} documents.")
    reprocess_documents.short_description = "Reprocess selected documents"

    def delete_model(self, request, obj):
        from .indexing.vector_index import delete_vectors_for_document
        delete_vectors_for_document(obj.id)
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        from .indexing.vector_index import delete_vectors_for_document
        for document_id in queryset.values_list('id', flat=True):
            delete_vectors_for_document(document_id)
        super().delete_queryset(request, queryset)


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('document', 'chunk_index')
    search_fields = ('content',)
    list_filter = ('document',)


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
