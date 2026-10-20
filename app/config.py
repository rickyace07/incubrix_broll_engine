from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    PROVIDER_MODE: str = "live"  # "live" or "mock"
    PEXELS_API_KEY: str = ""
    PIXABAY_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    STORAGE_BACKEND: str = "local"
    S3_ENDPOINT_URL: str = "http://minio:9000"
    S3_PUBLIC_BASE_URL: str = "http://localhost:8000/files"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "broll-outputs"
    DATABASE_PATH: str = "./jobs.db"
    WORK_DIR: str = "./tmp_work"
    LOCAL_S3_DIR: str = "./local_s3"
    WEBHOOK_SECRET: str = "incubrix-secret-key-2026"
    SIMULATE_PRIMARY_SOURCE_DOWN: bool = False
    MIN_SCENE_DURATION_SEC: float = 2.5
    MAX_AUDIO_DURATION_SEC: float = 600.0
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
settings = Settings()
