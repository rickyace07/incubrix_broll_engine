import os
import shutil
import boto3
from botocore.client import Config
from app.config import settings
class StorageAdapter:
    def __init__(self):
        os.makedirs(settings.LOCAL_S3_DIR, exist_ok=True)
        self.s3 = None
        if settings.STORAGE_BACKEND == "s3":
            self.s3 = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1"
            )
            try:
                self.s3.head_bucket(Bucket=settings.S3_BUCKET_NAME)
            except Exception:
                try:
                    self.s3.create_bucket(Bucket=settings.S3_BUCKET_NAME)
                except Exception:
                    pass
    def upload_file(self, local_path: str, object_key: str) -> str:
        dest_path = os.path.join(settings.LOCAL_S3_DIR, object_key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(local_path, dest_path)
        if settings.STORAGE_BACKEND == "s3" and self.s3:
            self.s3.upload_file(local_path, settings.S3_BUCKET_NAME, object_key)
        base = settings.S3_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/{object_key}"
    def check_health(self) -> bool:
        if settings.STORAGE_BACKEND == "s3" and self.s3:
            try:
                self.s3.list_buckets()
                return True
            except Exception:
                return False
        return os.path.isdir(settings.LOCAL_S3_DIR)
storage = StorageAdapter()
