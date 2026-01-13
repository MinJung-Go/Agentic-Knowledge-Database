"""EmbeddingClient 单元测试"""
import pytest
import respx
import httpx

from core.embedding.client import EmbeddingClient, AsyncEmbeddingClient, EmbeddingResult


class TestEmbeddingResult:
    """EmbeddingResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
        assert len(result.embeddings) == 1
        assert result.usage["prompt_tokens"] == 10


class TestEmbeddingClient:
    """EmbeddingClient 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        client = EmbeddingClient()
        assert client.base_url is not None
        assert client.model is not None

    def test_init_custom(self):
        """测试自定义初始化"""
        client = EmbeddingClient(
            base_url="http://custom:8001",
            model="custom-model",
            api_key="test-key",
            timeout=30.0,
        )
        assert client.base_url == "http://custom:8001"
        assert client.model == "custom-model"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0

    @respx.mock
    def test_embed_single_text(self):
        """测试单条文本向量化"""
        mock_response = {
            "object": "list",
            "data": [{"embedding": [0.1] * 4096, "index": 0}],
            "model": "Qwen/Qwen3-Embedding-8B",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        embedding = client.embed("测试文本")

        assert isinstance(embedding, list)
        assert len(embedding) == 4096

    @respx.mock
    def test_embed_batch(self):
        """测试批量向量化"""
        mock_response = {
            "object": "list",
            "data": [
                {"embedding": [0.1] * 4096, "index": 0},
                {"embedding": [0.2] * 4096, "index": 1},
            ],
            "model": "Qwen/Qwen3-Embedding-8B",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        result = client.embed_batch(["文本1", "文本2"])

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 4096

    @respx.mock
    def test_embed_batch_sorted_by_index(self):
        """测试批量结果按 index 排序"""
        # 响应中的顺序与请求顺序不一致
        mock_response = {
            "object": "list",
            "data": [
                {"embedding": [0.2] * 10, "index": 1},
                {"embedding": [0.1] * 10, "index": 0},
            ],
            "model": "test",
            "usage": {},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        result = client.embed_batch(["文本0", "文本1"])

        # 应该按 index 排序，第一个是 0.1
        assert result.embeddings[0][0] == 0.1
        assert result.embeddings[1][0] == 0.2

    @respx.mock
    def test_embed_documents_batching(self):
        """测试大批量分批处理"""
        call_count = 0

        def mock_handler(request):
            nonlocal call_count
            call_count += 1
            data = request.content.decode()
            # 简单解析 input 长度
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"embedding": [0.1] * 100, "index": i} for i in range(2)],
                "usage": {},
            })

        respx.post("http://localhost:8001/v1/embeddings").mock(side_effect=mock_handler)

        client = EmbeddingClient(base_url="http://localhost:8001")
        # 4 个文档，batch_size=2，应该调用 2 次
        texts = ["文本1", "文本2", "文本3", "文本4"]
        embeddings = client.embed_documents(texts, batch_size=2)

        assert len(embeddings) == 4
        assert call_count == 2

    @respx.mock
    def test_embed_documents_single_batch(self):
        """测试小批量无需分批"""
        mock_response = {
            "object": "list",
            "data": [{"embedding": [0.1] * 100, "index": i} for i in range(3)],
            "usage": {},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        texts = ["文本1", "文本2", "文本3"]
        embeddings = client.embed_documents(texts, batch_size=10)

        assert len(embeddings) == 3

    @respx.mock
    def test_embedding_dimension(self):
        """测试向量维度"""
        mock_response = {
            "object": "list",
            "data": [{"embedding": [0.1] * 4096, "index": 0}],
            "usage": {},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        embedding = client.embed("测试")

        assert len(embedding) == 4096

    @respx.mock
    def test_health_check_success(self):
        """测试健康检查成功"""
        respx.get("http://localhost:8001/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        assert client.health_check() is True

    @respx.mock
    def test_health_check_failure(self):
        """测试健康检查失败"""
        respx.get("http://localhost:8001/health").mock(
            return_value=httpx.Response(500)
        )

        client = EmbeddingClient(base_url="http://localhost:8001")
        assert client.health_check() is False

    @respx.mock
    def test_http_error(self):
        """测试 HTTP 错误"""
        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        client = EmbeddingClient(base_url="http://localhost:8001")

        with pytest.raises(httpx.HTTPStatusError):
            client.embed("测试")

    def test_context_manager(self):
        """测试上下文管理器"""
        with EmbeddingClient(base_url="http://localhost:8001") as client:
            assert client.client is not None


class TestAsyncEmbeddingClient:
    """AsyncEmbeddingClient 测试"""

    def test_init(self):
        """测试初始化"""
        client = AsyncEmbeddingClient(
            base_url="http://localhost:8001",
            model="test-model",
        )
        assert client.base_url == "http://localhost:8001"
        assert client.model == "test-model"

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_embed(self):
        """测试异步单条向量化"""
        mock_response = {
            "object": "list",
            "data": [{"embedding": [0.1] * 100, "index": 0}],
            "usage": {},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = AsyncEmbeddingClient(base_url="http://localhost:8001")
        embedding = await client.embed("测试")

        assert len(embedding) == 100
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_embed_batch(self):
        """测试异步批量向量化"""
        mock_response = {
            "object": "list",
            "data": [
                {"embedding": [0.1] * 100, "index": 0},
                {"embedding": [0.2] * 100, "index": 1},
            ],
            "usage": {"prompt_tokens": 20},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = AsyncEmbeddingClient(base_url="http://localhost:8001")
        result = await client.embed_batch(["文本1", "文本2"])

        assert len(result.embeddings) == 2
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_embed_documents(self):
        """测试异步批量文档向量化"""
        mock_response = {
            "object": "list",
            "data": [{"embedding": [0.1] * 100, "index": i} for i in range(3)],
            "usage": {},
        }

        respx.post("http://localhost:8001/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = AsyncEmbeddingClient(base_url="http://localhost:8001")
        embeddings = await client.embed_documents(["文本1", "文本2", "文本3"])

        assert len(embeddings) == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """测试异步上下文管理器"""
        async with AsyncEmbeddingClient(base_url="http://localhost:8001") as client:
            assert client.client is not None
