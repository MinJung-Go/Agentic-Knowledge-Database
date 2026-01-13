"""MilvusClient 单元测试"""
import pytest
from unittest.mock import patch, MagicMock

from core.milvus.client import MilvusClient, AsyncMilvusClient, SearchResult


class TestSearchResult:
    """SearchResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = SearchResult(
            id=1,
            score=0.95,
            data={"content": "测试内容"},
        )
        assert result.id == 1
        assert result.score == 0.95
        assert result.data["content"] == "测试内容"

    def test_default_data(self):
        """测试默认 data 为空字典"""
        result = SearchResult(id=1, score=0.9)
        assert result.data == {}


class TestMilvusClient:
    """MilvusClient 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch("core.milvus.client.settings") as mock_settings:
            mock_settings.milvus_host = "192.168.25.177"
            mock_settings.milvus_port = 19530

            client = MilvusClient()
            assert client.uri == "http://192.168.25.177:19530"
            assert client._client is None  # 懒加载

    def test_init_custom_uri(self):
        """测试自定义 URI"""
        client = MilvusClient(uri="http://custom-host:19531")
        assert client.uri == "http://custom-host:19531"

    def test_init_with_token(self):
        """测试带 token 初始化"""
        client = MilvusClient(uri="http://localhost:19530", token="test_token")
        assert client.token == "test_token"

    @patch("core.milvus.client.PyMilvusClient")
    def test_client_property_lazy_loading(self, mock_pymilvus):
        """测试客户端懒加载"""
        mock_instance = MagicMock()
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")

        # 首次访问前不应该创建
        assert client._client is None
        mock_pymilvus.assert_not_called()

        # 访问 client 属性触发创建
        _ = client.client
        mock_pymilvus.assert_called_once_with(uri="http://localhost:19530", token="")

    @patch("core.milvus.client.PyMilvusClient")
    def test_close(self, mock_pymilvus):
        """测试关闭连接"""
        mock_instance = MagicMock()
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        _ = client.client  # 触发连接
        client.close()

        mock_instance.close.assert_called_once()
        assert client._client is None

    def test_close_not_connected(self):
        """测试未连接时关闭"""
        client = MilvusClient(uri="http://localhost:19530")
        # 不应该抛出异常
        client.close()

    @patch("core.milvus.client.PyMilvusClient")
    def test_list_collections(self, mock_pymilvus):
        """测试列出 Collections"""
        mock_instance = MagicMock()
        mock_instance.list_collections.return_value = ["collection1", "collection2"]
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        collections = client.list_collections()

        assert len(collections) == 2
        assert "collection1" in collections

    @patch("core.milvus.client.PyMilvusClient")
    def test_has_collection(self, mock_pymilvus):
        """测试检查 Collection 是否存在"""
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = True
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        result = client.has_collection("test_collection")

        assert result is True
        mock_instance.has_collection.assert_called_with("test_collection")

    @patch("core.milvus.client.PyMilvusClient")
    def test_create_collection(self, mock_pymilvus):
        """测试创建 Collection"""
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = False
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        client.create_collection("test_collection", dimension=1024)

        mock_instance.create_collection.assert_called_once()

    @patch("core.milvus.client.PyMilvusClient")
    def test_create_collection_exists(self, mock_pymilvus):
        """测试创建已存在的 Collection"""
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = True
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        client.create_collection("test_collection", dimension=1024)

        mock_instance.create_collection.assert_not_called()

    @patch("core.milvus.client.PyMilvusClient")
    def test_drop_collection(self, mock_pymilvus):
        """测试删除 Collection"""
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = True
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        client.drop_collection("test_collection")

        mock_instance.drop_collection.assert_called_with("test_collection")

    @patch("core.milvus.client.PyMilvusClient")
    def test_drop_collection_not_exists(self, mock_pymilvus):
        """测试删除不存在的 Collection"""
        mock_instance = MagicMock()
        mock_instance.has_collection.return_value = False
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        client.drop_collection("test_collection")

        mock_instance.drop_collection.assert_not_called()

    @patch("core.milvus.client.PyMilvusClient")
    def test_get_collection_stats(self, mock_pymilvus):
        """测试获取 Collection 统计信息"""
        mock_instance = MagicMock()
        mock_instance.get_collection_stats.return_value = {"row_count": 1000}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        stats = client.get_collection_stats("test_collection")

        assert stats["name"] == "test_collection"
        assert stats["row_count"] == 1000

    @patch("core.milvus.client.PyMilvusClient")
    def test_describe_collection(self, mock_pymilvus):
        """测试获取 Collection 详情"""
        mock_instance = MagicMock()
        mock_instance.describe_collection.return_value = {"description": "test"}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        desc = client.describe_collection("test_collection")

        assert desc["description"] == "test"

    @patch("core.milvus.client.PyMilvusClient")
    def test_insert(self, mock_pymilvus):
        """测试插入数据"""
        mock_instance = MagicMock()
        mock_instance.insert.return_value = {"insert_count": 2}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        data = [
            {"doc_id": "d1", "embedding": [0.1] * 1024},
            {"doc_id": "d2", "embedding": [0.2] * 1024},
        ]
        result = client.insert("test_collection", data)

        assert result["insert_count"] == 2
        mock_instance.flush.assert_called_once_with("test_collection")

    @patch("core.milvus.client.PyMilvusClient")
    def test_upsert(self, mock_pymilvus):
        """测试 upsert 数据"""
        mock_instance = MagicMock()
        mock_instance.upsert.return_value = {"upsert_count": 1}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        data = [{"doc_id": "d1", "embedding": [0.1] * 1024}]
        result = client.upsert("test_collection", data)

        assert result["upsert_count"] == 1

    @patch("core.milvus.client.PyMilvusClient")
    def test_delete_by_ids(self, mock_pymilvus):
        """测试按 ID 删除"""
        mock_instance = MagicMock()
        mock_instance.delete.return_value = {"delete_count": 3}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        result = client.delete("test_collection", ids=[1, 2, 3])

        assert result["delete_count"] == 3

    @patch("core.milvus.client.PyMilvusClient")
    def test_delete_by_filter(self, mock_pymilvus):
        """测试按过滤条件删除"""
        mock_instance = MagicMock()
        mock_instance.delete.return_value = {"delete_count": 5}
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        result = client.delete("test_collection", filter_expr='doc_id == "d1"')

        mock_instance.delete.assert_called_with(
            collection_name="test_collection",
            filter='doc_id == "d1"',
        )

    def test_delete_requires_ids_or_filter(self):
        """测试删除必须提供 ids 或 filter_expr"""
        client = MilvusClient(uri="http://localhost:19530")

        with pytest.raises(ValueError, match="必须指定"):
            client.delete("test_collection")

    @patch("core.milvus.client.PyMilvusClient")
    def test_search(self, mock_pymilvus):
        """测试向量搜索"""
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            [
                {"id": 1, "distance": 0.1, "entity": {"content": "内容1"}},
                {"id": 2, "distance": 0.2, "entity": {"content": "内容2"}},
            ]
        ]
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        results = client.search(
            "test_collection",
            query_vectors=[[0.1] * 1024],
            limit=10,
        )

        assert len(results) == 1
        assert len(results[0]) == 2
        assert isinstance(results[0][0], SearchResult)
        assert results[0][0].score == 0.1

    @patch("core.milvus.client.PyMilvusClient")
    def test_search_single(self, mock_pymilvus):
        """测试单向量搜索"""
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            [{"id": 1, "distance": 0.1, "entity": {"content": "内容1"}}]
        ]
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        results = client.search_single(
            "test_collection",
            query_vector=[0.1] * 1024,
            limit=10,
        )

        assert len(results) == 1
        assert results[0].id == 1

    @patch("core.milvus.client.PyMilvusClient")
    def test_query(self, mock_pymilvus):
        """测试标量查询"""
        mock_instance = MagicMock()
        mock_instance.query.return_value = [
            {"id": 1, "doc_id": "d1", "content": "内容1"},
        ]
        mock_pymilvus.return_value = mock_instance

        client = MilvusClient(uri="http://localhost:19530")
        results = client.query(
            "test_collection",
            filter_expr='doc_id == "d1"',
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"

    @patch("core.milvus.client.PyMilvusClient")
    def test_context_manager(self, mock_pymilvus):
        """测试上下文管理器"""
        mock_instance = MagicMock()
        mock_pymilvus.return_value = mock_instance

        with MilvusClient(uri="http://localhost:19530") as client:
            _ = client.client  # 触发连接

        mock_instance.close.assert_called_once()


class TestAsyncMilvusClient:
    """AsyncMilvusClient 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch("core.milvus.client.settings") as mock_settings:
            mock_settings.milvus_host = "192.168.25.177"
            mock_settings.milvus_port = 19530

            client = AsyncMilvusClient()
            assert client.uri == "http://192.168.25.177:19530"
            assert client._client is None

    def test_init_custom_uri(self):
        """测试自定义 URI"""
        client = AsyncMilvusClient(uri="http://custom-host:19531")
        assert client.uri == "http://custom-host:19531"

    @patch("core.milvus.client.PyAsyncMilvusClient")
    def test_client_property_lazy_loading(self, mock_pymilvus):
        """测试客户端懒加载"""
        mock_instance = MagicMock()
        mock_pymilvus.return_value = mock_instance

        client = AsyncMilvusClient(uri="http://localhost:19530")

        assert client._client is None
        _ = client.client
        mock_pymilvus.assert_called_once()


class TestGlobalClient:
    """全局客户端测试"""

    @patch("core.milvus.client._default_client", None)
    @patch("core.milvus.client.MilvusClient")
    def test_get_milvus_client(self, mock_client_class):
        """测试获取全局客户端"""
        from core.milvus.client import get_milvus_client

        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client1 = get_milvus_client()
        client2 = get_milvus_client()

        # 应该返回同一个实例
        assert client1 is client2
        mock_client_class.assert_called_once()

    @patch("core.milvus.client._default_async_client", None)
    @patch("core.milvus.client.AsyncMilvusClient")
    def test_get_async_milvus_client(self, mock_client_class):
        """测试获取全局异步客户端"""
        from core.milvus.client import get_async_milvus_client

        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client1 = get_async_milvus_client()
        client2 = get_async_milvus_client()

        assert client1 is client2
        mock_client_class.assert_called_once()
