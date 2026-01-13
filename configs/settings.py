from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Agentic Knowledge Database"
    debug: bool = False

    # MinerU - 文档解析
    mineru_base_url: str = "http://localhost:8003"

    # LLM 服务
    llm_base_url: str = "http://localhost:8000"
    llm_model: str = "Qwen/Qwen3-VL-8B"
    llm_api_key: str = "EMPTY"

    # Embedding 服务
    embedding_base_url: str = "http://localhost:8001"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_api_key: str = "EMPTY"
    embedding_dimension: int = 1024  # 0.6B=1024, 8B=4096

    # Rerank 服务
    rerank_base_url: str = "http://localhost:8002"
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"
    rerank_api_key: str = "EMPTY"
    rerank_top_k: int = 5

    # Milvus 向量数据库
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"

    class Config:
        env_file = ".env"


settings = Settings()
