from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0007_alter_document_status'),
    ]

    operations = [
        migrations.DeleteModel(name='VectorIndexMapping'),
    ]
