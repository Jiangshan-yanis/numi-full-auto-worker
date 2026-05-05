# Numi Full-Automatic Worker

This FastAPI service receives Supabase Database Webhooks from `storage.objects` and refreshes the corresponding Silver tables.

Tracked folders inside the `numi-bronze` bucket:
- `donor_pdf`
- `protocol_pdf`
- `experiment_excel`

Optional folder for file-id mapping:
- `metadata_bronze/bronze_file_catalog.xlsx`

Required environment variables:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `BRONZE_BUCKET`
- `WEBHOOK_SECRET`
