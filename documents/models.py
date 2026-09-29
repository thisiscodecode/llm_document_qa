from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('pending_processing', 'Pending Processing'),
        ('processing', 'Processing'),
        ('extracting_text', 'Extracting Text'),
        ('chunking', 'Chunking'),
        ('indexing', 'Indexing Vectors'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    ]

    VALID_TRANSITIONS = {
        'uploaded': ['pending_processing', 'processing', 'failed'],
        'pending_processing': ['processing', 'extracting_text', 'failed'],
        'processing': ['extracting_text', 'chunking', 'indexing', 'ready', 'failed'],
        'extracting_text': ['chunking', 'failed'],
        'chunking': ['indexing', 'failed'],
        'indexing': ['ready', 'failed'],
        'ready': ['pending_processing', 'processing'],
        'failed': ['pending_processing', 'processing'],
    }

    CHUNK_CONFIGS = {
        'default': {'chunk_size': 1000, 'overlap': 150},
        'code': {'chunk_size': 500, 'overlap': 100},
        'legal': {'chunk_size': 1500, 'overlap': 200},
        'academic': {'chunk_size': 1200, 'overlap': 180},
    }

    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    full_text = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    error_message = models.TextField(blank=True, null=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='documents')
    document_type = models.CharField(max_length=20, default='default')
    chunk_size = models.PositiveIntegerField(default=1000)
    chunk_overlap = models.PositiveIntegerField(default=150)
    processing_metadata = models.JSONField(default=dict, blank=True)
    retry_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def can_transition_to(self, new_status: str) -> bool:
        return new_status in self.VALID_TRANSITIONS.get(self.status, [])

    def transition_to(self, new_status: str) -> bool:
        if not self.can_transition_to(new_status):
            return False
        self.status = new_status
        self.save(update_fields=['status', 'updated_at'])
        return True

    def get_chunk_config(self) -> dict:
        return self.CHUNK_CONFIGS.get(self.document_type, self.CHUNK_CONFIGS['default'])

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']


class DocumentChunk(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    content = models.TextField()
    chunk_index = models.PositiveIntegerField()
    page_number = models.PositiveIntegerField(default=1)
    embedding = models.BinaryField(null=True, blank=True)

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"

    class Meta:
        ordering = ['chunk_index']


class ChatSession(models.Model):
    title = models.CharField(max_length=255, blank=True, default='New Chat')
    owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or f"Chat {self.id}"

    class Meta:
        ordering = ['-updated_at']


class QuestionHistory(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages', null=True, blank=True)
    question = models.TextField()
    answer = models.TextField(blank=True)
    retrieved_context = models.TextField(blank=True)
    source_chunks = models.JSONField(default=list, blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='history')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:80]

    class Meta:
        ordering = ['created_at']
        verbose_name_plural = 'Question histories'
