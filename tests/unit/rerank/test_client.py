"""RerankClient 单元测试"""
import pytest
import respx
import httpx

from core.rerank.client import RerankClient, AsyncRerankClient, RerankResult, RerankItem


class TestRerankItem:
    """RerankItem 数据类测试"""

    def test_creation(self):
        """测试创建"""
        item = RerankItem(index=0, score=0.95, text="测试文本")
        assert item.index == 0
        assert item.score == 0.95
        assert item.text == "测试文本"


class TestRerankResult:
    """RerankResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        items = [RerankItem(index=0, score=0.95, text="文本")]
        result = RerankResult(results=items, usage={"tokens": 10})
        assert len(result.results) == 1
        assert result.usage["tokens"] == 10


class TestRerankClient:
    """RerankClient 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        client = RerankClient()
        assert client.base_url is not None
        assert client.model is not None
        assert client.top_k > 0

    def test_init_custom(self):
        """测试自定义初始化"""
        client = RerankClient(
            base_url="http://custom:8002",
            model="custom-model",
            api_key="test-key",
            top_k=10,
            timeout=30.0,
        )
        assert client.base_url == "http://custom:8002"
        assert client.model == "custom-model"
        assert client.top_k == 10

    @respx.mock
    def test_rerank_basic(self):
        """测试基本重排序"""
        mock_response = {
            "id": "rerank-123",
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.75},
            ],
            "meta": {"billed_units": {"tokens": 100}},
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        documents = ["文档A", "文档B", "文档C"]
        result = client.rerank("查询", documents)

        assert isinstance(result, RerankResult)
        assert len(result.results) == 3
        # 检查排序后的顺序
        assert result.results[0].score == 0.95
        assert result.results[0].index == 1

    @respx.mock
    def test_rerank_top_k(self):
        """测试 top_k 参数"""
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002", top_k=5)
        result = client.rerank("查询", ["doc1", "doc2", "doc3"], top_k=2)

        assert len(result.results) == 2

    @respx.mock
    def test_rerank_preserves_text(self):
        """测试结果包含原始文本"""
        mock_response = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
            ],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        documents = ["文档A", "文档B"]
        result = client.rerank("查询", documents)

        # text 应该是原始文档内容
        assert result.results[0].text == "文档B"

    @respx.mock
    def test_rerank_with_metadata(self):
        """测试带元数据的重排序"""
        mock_response = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
            ],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        documents = [
            {"content": "文档A", "doc_id": "1"},
            {"content": "文档B", "doc_id": "2"},
        ]
        result = client.rerank_with_metadata("查询", documents)

        assert len(result) == 2
        # 第一个应该是文档B（index=1）
        assert result[0]["doc_id"] == "2"
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] == 0.95

    @respx.mock
    def test_rerank_with_metadata_custom_key(self):
        """测试自定义文本字段"""
        mock_response = {
            "results": [{"index": 0, "relevance_score": 0.9}],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        documents = [{"text": "文档内容", "id": "1"}]
        result = client.rerank_with_metadata("查询", documents, text_key="text")

        assert len(result) == 1

    @respx.mock
    def test_rerank_empty_documents(self):
        """测试空文档列表"""
        mock_response = {"results": []}

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        result = client.rerank("查询", [])

        assert len(result.results) == 0

    @respx.mock
    def test_rerank_score_field_compatibility(self):
        """测试 score 字段兼容性（relevance_score 或 score）"""
        # 有些实现用 score 而不是 relevance_score
        mock_response = {
            "results": [{"index": 0, "score": 0.88}],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = RerankClient(base_url="http://localhost:8002")
        result = client.rerank("查询", ["文档"])

        assert result.results[0].score == 0.88

    @respx.mock
    def test_health_check_success(self):
        """测试健康检查成功"""
        respx.get("http://localhost:8002/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        client = RerankClient(base_url="http://localhost:8002")
        assert client.health_check() is True

    @respx.mock
    def test_health_check_failure(self):
        """测试健康检查失败"""
        respx.get("http://localhost:8002/health").mock(
            return_value=httpx.Response(500)
        )

        client = RerankClient(base_url="http://localhost:8002")
        assert client.health_check() is False

    @respx.mock
    def test_http_error(self):
        """测试 HTTP 错误"""
        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        client = RerankClient(base_url="http://localhost:8002")

        with pytest.raises(httpx.HTTPStatusError):
            client.rerank("查询", ["文档"])

    def test_context_manager(self):
        """测试上下文管理器"""
        with RerankClient(base_url="http://localhost:8002") as client:
            assert client.client is not None


class TestAsyncRerankClient:
    """AsyncRerankClient 测试"""

    def test_init(self):
        """测试初始化"""
        client = AsyncRerankClient(
            base_url="http://localhost:8002",
            model="test-model",
            top_k=10,
        )
        assert client.base_url == "http://localhost:8002"
        assert client.top_k == 10

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_rerank(self):
        """测试异步重排序"""
        mock_response = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
            ],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = AsyncRerankClient(base_url="http://localhost:8002")
        result = await client.rerank("查询", ["文档A", "文档B"])

        assert len(result.results) == 2
        assert result.results[0].score == 0.95
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_rerank_with_metadata(self):
        """测试异步带元数据重排序"""
        mock_response = {
            "results": [{"index": 0, "relevance_score": 0.9}],
        }

        respx.post("http://localhost:8002/v1/rerank").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        client = AsyncRerankClient(base_url="http://localhost:8002")
        documents = [{"content": "文档", "id": "1"}]
        result = await client.rerank_with_metadata("查询", documents)

        assert len(result) == 1
        assert "rerank_score" in result[0]
        await client.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """测试异步上下文管理器"""
        async with AsyncRerankClient(base_url="http://localhost:8002") as client:
            assert client.client is not None
