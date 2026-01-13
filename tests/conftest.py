"""全局测试配置和 Fixtures"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.parsers.chunker import Chunk


# ============ 通用 Fixtures ============

@pytest.fixture
def sample_text():
    """测试用文本"""
    return """这是第一段内容。这是第一段的第二句话。

这是第二段内容。包含一些重要信息。

这是第三段。最后一段内容。"""


@pytest.fixture
def sample_markdown():
    """测试用 Markdown 文本"""
    return """# 标题一

这是标题一下的内容。

## 标题二

这是标题二下的内容。

### 标题三

这是标题三下的内容。"""


@pytest.fixture
def sample_chunks():
    """测试用分块数据"""
    return [
        Chunk(content="第一段内容", index=0, metadata={"doc_id": "doc_001"}),
        Chunk(content="第二段内容", index=1, metadata={"doc_id": "doc_001"}),
        Chunk(content="第三段内容", index=2, metadata={"doc_id": "doc_001"}),
    ]


@pytest.fixture
def sample_documents():
    """测试用文档数据"""
    return [
        {
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "content": "人工智能是计算机科学的一个分支",
            "metadata": {"filename": "ai.pdf"},
        },
        {
            "doc_id": "doc_001",
            "chunk_id": "chunk_002",
            "content": "机器学习是人工智能的核心技术",
            "metadata": {"filename": "ai.pdf"},
        },
        {
            "doc_id": "doc_002",
            "chunk_id": "chunk_003",
            "content": "深度学习使用神经网络进行学习",
            "metadata": {"filename": "dl.pdf"},
        },
    ]


# ============ Embedding Fixtures ============

@pytest.fixture
def mock_embedding_response():
    """Mock Embedding API 响应"""
    return {
        "object": "list",
        "data": [{"embedding": [0.1] * 4096, "index": 0}],
        "model": "Qwen/Qwen3-Embedding-8B",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def sample_embedding():
    """测试用向量"""
    return [0.1] * 4096


# ============ Milvus Fixtures ============

@pytest.fixture
def mock_milvus_connection():
    """Mock Milvus 连接"""
    with patch("pymilvus.connections.connect") as mock_connect:
        with patch("pymilvus.connections.disconnect") as mock_disconnect:
            yield {
                "connect": mock_connect,
                "disconnect": mock_disconnect,
            }


@pytest.fixture
def mock_collection():
    """Mock Milvus Collection"""
    collection = MagicMock()
    collection.name = "test_collection"
    collection.num_entities = 100
    collection.insert.return_value = MagicMock(primary_keys=[1, 2, 3])
    collection.search.return_value = [[]]
    collection.delete.return_value = MagicMock(delete_count=1)
    return collection


# ============ Rerank Fixtures ============

@pytest.fixture
def mock_rerank_response():
    """Mock Rerank API 响应"""
    return {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.85},
            {"index": 2, "relevance_score": 0.75},
        ]
    }


# ============ LLM Fixtures ============

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI Client"""
    client = MagicMock()

    # Mock chat completion
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "这是 AI 生成的回答"
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 50
    completion.usage.total_tokens = 150

    client.chat.completions.create.return_value = completion
    return client


@pytest.fixture
def mock_async_openai_client():
    """Mock AsyncOpenAI Client"""
    client = MagicMock()

    # Mock async chat completion
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "这是 AI 生成的回答"
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 50
    completion.usage.total_tokens = 150

    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


# ============ MinerU Fixtures ============

@pytest.fixture
def mock_mineru_response():
    """Mock MinerU API 响应"""
    return {
        "content": "# 文档标题\n\n这是解析后的文档内容。\n\n## 第一章\n\n章节内容...",
        "images": [],
        "tables": [],
    }


# ============ HTTP Mock Helpers ============

@pytest.fixture
def httpx_mock_embedding(mock_embedding_response):
    """Mock httpx embedding 请求"""
    import respx
    import httpx

    with respx.mock:
        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_embedding_response)
        )
        yield


@pytest.fixture
def httpx_mock_rerank(mock_rerank_response):
    """Mock httpx rerank 请求"""
    import respx
    import httpx

    with respx.mock:
        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_rerank_response)
        )
        yield
