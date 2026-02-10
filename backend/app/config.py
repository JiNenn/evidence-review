from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+psycopg://diffui:diffui@postgres:5432/diffui"
    redis_url: str = "redis://redis:6379/0"

    s3_endpoint: str = "http://minio:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "diffui"
    s3_presign_expires: int = 900
    source_doc_orphan_ttl_hours: int = 24

    jwt_secret: str = "replace-with-strong-secret"
    jwt_expires_in: int = 3600
    auth_enabled: bool = False
    auth_dev_user: str = "admin"
    auth_dev_password: str = "admin"

    llm_provider: str = "stub"
    llm_api_key: str = ""
    model_high: str = "high-stub-v1"
    model_low: str = "low-stub-v1"

    cors_origins: str = "http://localhost:3000"

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
