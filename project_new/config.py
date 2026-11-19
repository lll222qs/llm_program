
# config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App
    LOG_LEVEL: str = "INFO"
    AGENT_PORT: int = 8000

    # LLM
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b-instruct"

    # Vector DB (Qdrant)
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    QDRANT_COLLECTION: str = "langchain"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Paths
    HF_HOME: str = "./.cache/huggingface"
    TRANSFORMERS_CACHE: str = "./.cache/huggingface"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # 允许 .env 里有配置类没定义的变量，不报错
    )

settings = Settings()

# 说明：删除了原先手动写 /etc/hosts 注入 host.docker.internal 的 hack。
# 该段在 Windows 本机必报 WinError 2（用了 Linux 的 ip route 命令）。
# 现在改由 docker-compose.yml 的 extra_hosts: host.docker.internal:host-gateway
# 统一处理，容器内能解析，本机也不会再刷这个警告。