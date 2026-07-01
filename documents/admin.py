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
        from .processing import process_document
        for doc in queryset:
            process_document(doc.id)
        self.message_user(request, f"Reprocessing {queryset.count()} documents.")
    reprocess_documents.short_description = "Reprocess selected documents"


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
