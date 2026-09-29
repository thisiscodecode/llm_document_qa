from django.core.management.base import BaseCommand, CommandError

from documents.indexing.index_sync import rebuild_global_index, upsert_document_chunks
from documents.models import Document


class Command(BaseCommand):
    help = "Rebuild the Chroma vector index from chunks stored in Django"

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=int,
            help='Reindex one document instead of rebuilding the full collection.',
        )

    def handle(self, *args, **options):
        document_id = options.get('document_id')
        if document_id is not None:
            if not Document.objects.filter(id=document_id).exists():
                raise CommandError(f'Document {document_id} does not exist')
            success = upsert_document_chunks(document_id)
        else:
            success = rebuild_global_index()

        if not success:
            raise CommandError('Chroma index rebuild failed; check logs and embedding credentials')
        self.stdout.write(self.style.SUCCESS('Chroma index rebuilt successfully'))
