from rest_framework import serializers
from .models import Document, DocumentChunk, QuestionHistory, ChatSession


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            'id',
            'title',
            'status',
            'error_message',
            'processed_at',
            'chunk_count',
            'created_at',
        ]

    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentChunkSerializer(serializers.ModelSerializer):
    document_title = serializers.CharField(source='document.title', read_only=True)

    class Meta:
        model = DocumentChunk
        fields = [
            'id',
            'document',
            'document_title',
            'content',
            'chunk_index',
            'page_number',
        ]


class ChatSessionSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            'id',
            'title',
            'message_count',
            'last_message',
            'created_at',
            'updated_at',
        ]

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        if last:
            return {
                'question': last.question[:100],
                'created_at': last.created_at,
            }
        return None


class AskQuestionSerializer(serializers.Serializer):
    SEARCH_METHOD_CHOICES = [
        ("simple", "Simple Search"),
        ("bm25", "BM25 Search"),
        ("vector", "Vector Search"),
        ("hybrid", "Hybrid Search (BM25 + Vector)"),
    ]

    question = serializers.CharField(
        required=True,
        max_length=5000,
        help_text="The question to ask about the documents"
    )

    search_method = serializers.ChoiceField(
        choices=SEARCH_METHOD_CHOICES,
        default="simple",
        required=False,
        help_text="Search method to use for retrieval"
    )

    document_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        default=list,
        help_text="List of document IDs to search within (empty = all documents)"
    )

    session_id = serializers.IntegerField(
        required=False,
        allow_null=True,
        help_text="Chat session ID (creates new session if not provided)"
    )


class QuestionHistorySerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source='owner.username', read_only=True, default=None)

    class Meta:
        model = QuestionHistory
        fields = [
            "id",
            "session",
            "question",
            "answer",
            "retrieved_context",
            "source_chunks",
            "owner",
            "owner_username",
            "created_at",
        ]
        read_only_fields = ['owner']
