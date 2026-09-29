import os
import re
import zipfile

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DOCX_UNCOMPRESSED_SIZE = 200 * 1024 * 1024
MAX_DOCX_XML_PART_SIZE = 32 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000
ALLOWED_EXTENSIONS = {'.docx', '.pdf'}


def validate_upload(file):
    errors = []

    if file.size > MAX_FILE_SIZE:
        errors.append(f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.')

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        errors.append(f'Invalid file type: {ext}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}')

    safe_name = re.sub(r'[^\w\s\-.]', '', file.name)
    if not safe_name:
        errors.append('Invalid filename.')

    if ext in ALLOWED_EXTENSIONS and file.size <= MAX_FILE_SIZE:
        position = file.tell()
        try:
            file.seek(0)
            if ext == '.pdf':
                if file.read(5) != b'%PDF-':
                    errors.append('Invalid PDF file signature.')
            else:
                try:
                    with zipfile.ZipFile(file) as archive:
                        entries = archive.infolist()
                    parts = {entry.filename: entry for entry in entries}
                    if not {'[Content_Types].xml', 'word/document.xml'} <= parts.keys():
                        errors.append('Invalid DOCX file structure.')
                    elif (
                        len(entries) > MAX_DOCX_ENTRIES
                        or sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_SIZE
                        or any(
                            entry.filename.lower().endswith(('.xml', '.rels'))
                            and entry.file_size > MAX_DOCX_XML_PART_SIZE
                            for entry in entries
                        )
                    ):
                        errors.append('DOCX contents exceed the allowed extraction size.')
                except (zipfile.BadZipFile, OSError):
                    errors.append('Invalid DOCX file signature.')
        finally:
            file.seek(position)

    return errors
