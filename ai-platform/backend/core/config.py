"""
Core configuration settings for GuidedPATH AI Platform
"""

import secrets
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Project Information
    PROJECT_NAME: str = "GuidedPATH AI Platform"
    PROJECT_DESCRIPTION: str = "Comprehensive AI-powered healthcare platform for cancer and inflammatory disease patients"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # React Native dev server
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(
        cls, v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Trusted Hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://guidedpath:password@localhost:5432/guidedpath"
    MONGODB_URL: str = "mongodb://localhost:27017/guidedpath"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Vector Database
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    WEAVIATE_URL: Optional[str] = None

    # AI Model API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None

    # OAuth Configuration
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    # Healthcare APIs
    FHIR_SERVER_URL: Optional[str] = None
    DRUGBANK_API_KEY: Optional[str] = None

    # File Storage
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None

    # Feature Flags
    ENABLE_VOICE_FEATURES: bool = True
    ENABLE_VIDEO_FEATURES: bool = False
    ENABLE_ADVANCED_ANALYTICS: bool = False

    # Model Configuration
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL: str = "anthropic/claude-3-5-sonnet-20241022"
    MAX_TOKENS_RESPONSE: int = 4096

    # Healthcare Settings
    DEFAULT_TRIAGE_URGENCY: str = "medium"
    ENABLE_EMERGENCY_ESCALATION: bool = True
    CLINICIAN_REVIEW_THRESHOLD: float = 0.8

    # Privacy and Compliance
    ENABLE_ANONYMIZATION: bool = True
    DATA_RETENTION_DAYS: int = 2555  # 7 years
    CONSENT_REQUIRED: bool = True


# Global settings instance
settings = Settings()
