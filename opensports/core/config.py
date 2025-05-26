"""
Configuration management for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Settings(PydanticBaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="OpenSports", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    database_url: str = Field(default="sqlite:///opensports.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Sports Data APIs
    espn_api_key: Optional[str] = Field(default=None, env="ESPN_API_KEY")
    sportradar_api_key: Optional[str] = Field(default=None, env="SPORTRADAR_API_KEY")
    odds_api_key: Optional[str] = Field(default=None, env="ODDS_API_KEY")
    nba_api_key: Optional[str] = Field(default=None, env="NBA_API_KEY")
    nfl_api_key: Optional[str] = Field(default=None, env="NFL_API_KEY")
    
    # Monitoring and Observability
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Machine Learning
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Real-time Processing
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_prefix: str = Field(default="opensports", env="KAFKA_TOPIC_PREFIX")
    
    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Data Storage
    data_storage_path: str = Field(default="./data", env="DATA_STORAGE_PATH")
    backup_storage_path: str = Field(default="./backups", env="BACKUP_STORAGE_PATH")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # CORS Settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Feature Flags
    enable_ai_agents: bool = Field(default=True, env="ENABLE_AI_AGENTS")
    enable_real_time: bool = Field(default=True, env="ENABLE_REAL_TIME")
    enable_experiments: bool = Field(default=True, env="ENABLE_EXPERIMENTS")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "echo": self.database_echo and not self.is_production,
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.redis_url,
            "decode_responses": True,
        }
    
    @property
    def openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
        }
    
    @property
    def sports_apis_config(self) -> Dict[str, Optional[str]]:
        """Get sports APIs configuration."""
        return {
            "espn": self.espn_api_key,
            "sportradar": self.sportradar_api_key,
            "odds": self.odds_api_key,
            "nba": self.nba_api_key,
            "nfl": self.nfl_api_key,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        settings.model_cache_dir,
        settings.data_storage_path,
        settings.backup_storage_path,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Create directories on import
create_directories() 