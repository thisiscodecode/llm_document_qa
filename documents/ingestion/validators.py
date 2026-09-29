import os
import re

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
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

    return errors