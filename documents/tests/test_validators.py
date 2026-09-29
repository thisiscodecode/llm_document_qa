from io import BytesIO
from zipfile import ZipFile

from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.ingestion.validators import validate_upload


class ValidatorTests(TestCase):
    def test_validate_upload_rejects_large_file(self):
        file = SimpleUploadedFile("test.pdf", b"x" * (51 * 1024 * 1024))
        errors = validate_upload(file)
        self.assertTrue(any("too large" in e.lower() for e in errors))

    def test_validate_upload_rejects_unsupported_extension(self):
        file = SimpleUploadedFile("test.exe", b"content")
        errors = validate_upload(file)
        self.assertTrue(any("invalid file type" in e.lower() for e in errors))

    def test_validate_upload_accepts_pdf(self):
        file = SimpleUploadedFile("test.pdf", b"%PDF-1.4\n1 0 obj\n")
        errors = validate_upload(file)
        self.assertEqual(len(errors), 0)

    def test_validate_upload_accepts_docx(self):
        buffer = BytesIO()
        with ZipFile(buffer, 'w') as package:
            package.writestr('[Content_Types].xml', '<Types/>')
            package.writestr('word/document.xml', '<document/>')
        file = SimpleUploadedFile("test.docx", buffer.getvalue())
        errors = validate_upload(file)
        self.assertEqual(len(errors), 0)
