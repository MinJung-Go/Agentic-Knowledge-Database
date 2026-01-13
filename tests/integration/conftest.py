"""集成测试配置 - 使用真实服务

集成测试与单元测试的区别：
- 单元测试：Mock 所有外部依赖，快速、隔离
- 集成测试：使用真实服务，验证组件间协作

运行方式：
    pytest tests/integration/ -v -m integration
"""
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from configs.settings import settings


# ============ 服务可用性检测 ============

def parse_host_port(url: str, default_port: int = 80) -> tuple[str, int]:
    """从 URL 解析 host 和 port"""
    # 处理纯 host:port 格式
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    return host, port


def is_service_available(host: str, port: int, timeout: float = 2.0) -> bool:
    """检测服务是否可用"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_http_service_healthy(url: str, timeout: float = 5.0) -> bool:
    """检测 HTTP 服务健康状态"""
    try:
        import httpx
        response = httpx.get(url, timeout=timeout)
        return response.status_code < 500
    except Exception:
        return False


# ============ 从 settings 获取服务地址 ============

def _check_milvus() -> bool:
    host, port = parse_host_port(settings.milvus_host, settings.milvus_port)
    return is_service_available(host, settings.milvus_port)


def _check_embedding() -> bool:
    host, port = parse_host_port(settings.embedding_base_url)
    return is_service_available(host, port)


def _check_rerank() -> bool:
    host, port = parse_host_port(settings.rerank_base_url)
    return is_service_available(host, port)


def _check_llm() -> bool:
    host, port = parse_host_port(settings.llm_base_url)
    return is_service_available(host, port)


def _check_mineru() -> bool:
    host, port = parse_host_port(settings.mineru_base_url)
    return is_service_available(host, port)


# ============ 跳过条件标记 ============

requires_milvus = pytest.mark.skipif(
    not _check_milvus(),
    reason=f"Milvus 服务不可用 ({settings.milvus_host}:{settings.milvus_port})"
)

requires_embedding = pytest.mark.skipif(
    not _check_embedding(),
    reason=f"Embedding 服务不可用 ({settings.embedding_base_url})"
)

requires_rerank = pytest.mark.skipif(
    not _check_rerank(),
    reason=f"Rerank 服务不可用 ({settings.rerank_base_url})"
)

requires_llm = pytest.mark.skipif(
    not _check_llm(),
    reason=f"LLM 服务不可用 ({settings.llm_base_url})"
)

requires_mineru = pytest.mark.skipif(
    not _check_mineru(),
    reason=f"MinerU 解析服务不可用 ({settings.mineru_base_url})"
)

requires_all_services = pytest.mark.skipif(
    not all([_check_milvus(), _check_embedding(), _check_llm()]),
    reason="部分核心服务不可用"
)


# ============ Milvus Fixtures ============

@pytest.fixture(scope="function")
def milvus_client(request):
    """真实 Milvus 连接（每个测试函数独立）"""
    # 先检查 Milvus 是否可用，不可用则跳过
    if not _check_milvus():
        pytest.skip(f"Milvus 服务不可用 ({settings.milvus_host}:{settings.milvus_port})")

    from core.milvus.client import MilvusClient

    # 使用 settings 中的配置（新 API 使用 uri）
    client = MilvusClient()  # 自动从 settings 读取配置
    try:
        # 触发连接（懒加载）
        _ = client.client
    except Exception as e:
        pytest.skip(f"Milvus 连接失败: {e}")

    yield client

    try:
        client.close()
    except Exception:
        pass


@pytest.fixture(scope="function")
def test_collection(milvus_client):
    """集成测试专用 Collection（每个测试函数独立）"""
    # milvus_client fixture 会在不可用时 skip，此处 milvus_client 已经验证过可用性
    import uuid
    from core.milvus.collection import CollectionManager

    # 使用 UUID 避免并发测试冲突
    collection_name = f"test_integration_{uuid.uuid4().hex[:8]}"

    collection = CollectionManager(
        collection_name=collection_name,
        dimension=1024,  # qwen3-0.6b-embedding outputs 1024 dimensions
        client=milvus_client,
    )
    collection.create()

    yield collection

    # 清理测试数据
    try:
        collection.drop()
    except Exception:
        pass


@pytest.fixture(scope="function")
def shared_collection(milvus_client):
    """共享 Collection（用于需要预置数据的测试）"""
    import uuid
    from core.milvus.collection import CollectionManager

    collection_name = f"test_shared_{uuid.uuid4().hex[:8]}"

    collection = CollectionManager(
        collection_name=collection_name,
        dimension=1024,  # qwen3-0.6b-embedding outputs 1024 dimensions
        client=milvus_client,
    )
    collection.create()

    yield collection

    try:
        collection.drop()
    except Exception:
        pass


# ============ Embedding Fixtures ============

@pytest.fixture(scope="module")
def embedding_client():
    """真实 Embedding 客户端（使用 settings 配置）"""
    from core.embedding.client import EmbeddingClient

    # 使用 settings 中的配置，自动检测端点类型
    return EmbeddingClient()


@pytest.fixture(scope="function")
async def async_embedding_client():
    """异步 Embedding 客户端（用于性能测试）"""
    # 先检查服务是否可用
    if not _check_embedding():
        pytest.skip(f"Embedding 服务不可用 ({settings.embedding_base_url})")

    from core.embedding.client import AsyncEmbeddingClient

    client = AsyncEmbeddingClient()
    yield client
    await client.close()


# ============ Rerank Fixtures ============

@pytest.fixture(scope="module")
def rerank_client():
    """真实 Rerank 客户端（使用 settings 配置）"""
    from core.rerank.client import RerankClient

    # 使用 settings 中的配置
    return RerankClient()


@pytest.fixture(scope="function")
async def async_rerank_client():
    """异步 Rerank 客户端（用于性能测试）"""
    # 先检查服务是否可用
    if not _check_rerank():
        pytest.skip(f"Rerank 服务不可用 ({settings.rerank_base_url})")

    from core.rerank.client import AsyncRerankClient

    client = AsyncRerankClient()
    yield client
    await client.close()


# ============ LLM Fixtures ============

@pytest.fixture(scope="module")
def llm_client():
    """真实 LLM 客户端（使用 settings 配置）"""
    from core.llm.client import LLMClient

    # 使用 settings 中的配置
    return LLMClient()


@pytest.fixture(scope="function")
async def async_llm_client():
    """异步 LLM 客户端（用于性能测试）"""
    # 先检查服务是否可用
    if not _check_llm():
        pytest.skip(f"LLM 服务不可用 ({settings.llm_base_url})")

    from core.llm.client import AsyncLLMClient

    # AsyncLLMClient 使用 openai.AsyncOpenAI，不需要手动关闭
    client = AsyncLLMClient()
    yield client


# ============ 文档解析 Fixtures ============

@pytest.fixture(scope="module")
def mineru_client():
    """真实 MinerU 解析客户端（使用 settings 配置）"""
    from core.parsers.mineru import MinerUParser

    # 使用 settings 中的配置
    return MinerUParser()


# ============ 测试数据 Fixtures ============

@pytest.fixture
def fixtures_dir() -> Path:
    """测试数据目录"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf(fixtures_dir) -> Path:
    """测试 PDF 文件"""
    pdf_path = fixtures_dir / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"测试文件不存在: {pdf_path}")
    return pdf_path


@pytest.fixture
def sample_texts() -> list[str]:
    """测试文本列表"""
    return [
        "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支。",
        "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。",
        "自然语言处理（NLP）让计算机能够理解和生成人类语言。",
        "计算机视觉使机器能够从图像和视频中提取有意义的信息。",
        "强化学习通过与环境交互来学习最优策略。",
        "知识图谱是一种结构化的知识表示方法，用于描述实体及其关系。",
        "RAG（检索增强生成）结合了检索系统和生成模型的优势。",
    ]


@pytest.fixture
def sample_documents(sample_texts) -> list[dict]:
    """测试文档数据"""
    return [
        {
            "doc_id": f"doc_{i:03d}",
            "chunk_id": f"chunk_{i:03d}",
            "content": text,
            "metadata": {
                "filename": f"test_{i}.txt",
                "page": 1,
                "source": "integration_test",
            },
        }
        for i, text in enumerate(sample_texts)
    ]


# ============ 辅助函数 ============

@pytest.fixture
def performance_timer():
    """性能计时器"""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()
            return self

        def stop(self):
            self.end_time = time.perf_counter()
            return self

        @property
        def elapsed_ms(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return (self.end_time - self.start_time) * 1000

        @property
        def elapsed_s(self) -> float:
            return self.elapsed_ms / 1000

    return Timer


# ============ Pytest Hooks ============

def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line(
        "markers", "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试标记"
    )
    config.addinivalue_line(
        "markers", "e2e: 端到端测试标记"
    )
