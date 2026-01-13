"""CollectionManager 单元测试"""
import pytest
from unittest.mock import patch, MagicMock

from configs.settings import settings
from core.milvus.collection import CollectionManager, CollectionConfig
from core.milvus.client import SearchResult


class TestCollectionConfig:
    """CollectionConfig 数据类测试"""

    def test_creation_default(self):
        """测试默认值（从配置读取）"""
        config = CollectionConfig(name="test")
        assert config.name == "test"
        assert config.dimension == settings.embedding_dimension
        assert config.metric_type == "L2"  # Milvus 2.2.x 只支持 L2 和 IP

    def test_creation_custom(self):
        """测试自定义值"""
        config = CollectionConfig(
            name="custom",
            dimension=768,
            metric_type="IP",
        )
        assert config.dimension == 768
        assert config.metric_type == "IP"


class TestCollectionManager:
    """CollectionManager 测试"""

    def test_init_default(self):
        """测试默认初始化（从配置读取）"""
        manager = CollectionManager()
        assert manager.collection_name is not None
        assert manager.dimension == settings.embedding_dimension
        assert manager.metric_type == "L2"

    def test_init_custom(self):
        """测试自定义初始化"""
        manager = CollectionManager(
            collection_name="custom_collection",
            dimension=768,
            metric_type="IP",
        )
        assert manager.collection_name == "custom_collection"
        assert manager.dimension == 768
        assert manager.metric_type == "IP"

    def test_client_property(self):
        """测试客户端属性"""
        mock_client = MagicMock()
        manager = CollectionManager(client=mock_client)
        assert manager.client == mock_client

    @patch("core.milvus.collection.get_milvus_client")
    def test_client_lazy_loading(self, mock_get_client):
        """测试客户端懒加载"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        # 第一次访问应该触发 get_milvus_client
        _ = manager.client

        mock_get_client.assert_called_once()

    def test_exists(self):
        """测试 Collection 存在性检查"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True

        manager = CollectionManager(client=mock_client)
        assert manager.exists() is True

        mock_client.has_collection.return_value = False
        assert manager.exists() is False

    def test_create_collection_new(self):
        """测试创建新 Collection"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.create()

        assert result is True
        mock_client.client.create_collection.assert_called_once()

    def test_create_collection_exists(self):
        """测试 Collection 已存在"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.create()

        assert result is False
        mock_client.client.create_collection.assert_not_called()

    def test_create_collection_drop_if_exists(self):
        """测试删除已存在的 Collection"""
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [True, False]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.create(drop_if_exists=True)

        mock_client.drop_collection.assert_called_once_with("test_collection")
        mock_client.client.create_collection.assert_called_once()

    def test_drop(self):
        """测试删除 Collection"""
        mock_client = MagicMock()

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.drop()

        mock_client.drop_collection.assert_called_once_with("test_collection")

    def test_stats(self):
        """测试获取统计信息"""
        mock_client = MagicMock()
        mock_client.get_collection_stats.return_value = {"row_count": 100}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        stats = manager.stats()

        assert stats["row_count"] == 100
        mock_client.get_collection_stats.assert_called_once_with("test_collection")

    def test_insert_dict(self):
        """测试插入单个字典"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 1}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.insert({
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "content": "测试内容",
            "embedding": [0.1] * 1024,
        })

        assert result["insert_count"] == 1
        mock_client.insert.assert_called_once()

    def test_insert_list(self):
        """测试插入字典列表"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 2}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.insert([
            {
                "doc_id": "doc_001",
                "chunk_id": "chunk_001",
                "content": "内容1",
                "embedding": [0.1] * 1024,
            },
            {
                "doc_id": "doc_001",
                "chunk_id": "chunk_002",
                "content": "内容2",
                "embedding": [0.2] * 1024,
            },
        ])

        assert result["insert_count"] == 2

    def test_insert_normalizes_vector_to_embedding(self):
        """测试 vector 字段自动转换为 embedding"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 1}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.insert({
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "content": "测试内容",
            "vector": [0.1] * 1024,  # 使用 vector 而不是 embedding
        })

        # 检查传入的数据是否包含 embedding 字段
        call_args = mock_client.insert.call_args[0]
        inserted_data = call_args[1]
        assert "embedding" in inserted_data[0]
        assert "vector" not in inserted_data[0]

    def test_insert_adds_default_metadata(self):
        """测试插入时添加默认 metadata"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 1}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.insert({
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "content": "测试内容",
            "embedding": [0.1] * 1024,
            # 没有 metadata 字段
        })

        call_args = mock_client.insert.call_args[0]
        inserted_data = call_args[1]
        assert "metadata" in inserted_data[0]
        assert inserted_data[0]["metadata"] == {}

    def test_insert_single(self):
        """测试单条插入（显式参数）"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 1}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.insert_single(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="测试内容",
            embedding=[0.1] * 1024,
            metadata={"key": "value"},
        )

        assert result["insert_count"] == 1

    def test_insert_batch(self):
        """测试批量插入（显式参数）"""
        mock_client = MagicMock()
        mock_client.insert.return_value = {"insert_count": 3}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.insert_batch(
            doc_ids=["doc_001", "doc_001", "doc_001"],
            chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
            contents=["内容1", "内容2", "内容3"],
            embeddings=[[0.1] * 1024] * 3,
        )

        assert result["insert_count"] == 3

    def test_search(self):
        """测试向量搜索"""
        mock_client = MagicMock()
        mock_client.search_single.return_value = [
            SearchResult(id=1, score=0.95, data={
                "doc_id": "doc_001",
                "chunk_id": "chunk_001",
                "content": "测试内容",
                "metadata": {},
            }),
        ]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        results = manager.search(
            query_embedding=[0.1] * 1024,
            top_k=10,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "测试内容"

    def test_search_with_vector_alias(self):
        """测试使用 vector 参数别名搜索"""
        mock_client = MagicMock()
        mock_client.search_single.return_value = []

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.search(
            vector=[0.1] * 1024,  # 使用 vector 别名
            top_k=10,
        )

        mock_client.search_single.assert_called_once()

    def test_search_with_filter(self):
        """测试带过滤条件的搜索"""
        mock_client = MagicMock()
        mock_client.search_single.return_value = []

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        manager.search(
            query_embedding=[0.1] * 1024,
            top_k=5,
            filter_expr='doc_id == "doc_001"',
        )

        call_kwargs = mock_client.search_single.call_args[1]
        assert call_kwargs["filter_expr"] == 'doc_id == "doc_001"'

    def test_search_with_score_threshold(self):
        """测试分数阈值过滤"""
        mock_client = MagicMock()
        mock_client.search_single.return_value = [
            SearchResult(id=1, score=0.95, data={"content": "高分"}),
            SearchResult(id=2, score=0.5, data={"content": "低分"}),
        ]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        results = manager.search(
            query_embedding=[0.1] * 1024,
            top_k=10,
            score_threshold=0.8,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.95

    def test_search_requires_embedding(self):
        """测试搜索必须提供向量"""
        mock_client = MagicMock()

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )

        with pytest.raises(ValueError, match="必须提供"):
            manager.search(top_k=10)

    def test_query(self):
        """测试标量查询"""
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"id": 1, "doc_id": "doc_001", "content": "内容1"},
        ]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        results = manager.query(
            filter_expr='doc_id == "doc_001"',
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc_001"

    def test_count(self):
        """测试统计数量"""
        mock_client = MagicMock()
        mock_client.get_collection_stats.return_value = {"row_count": 100}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        count = manager.count()

        assert count == 100

    def test_count_with_filter(self):
        """测试带过滤条件的统计"""
        mock_client = MagicMock()
        mock_client.query.return_value = [{"count(*)": 50}]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        count = manager.count(filter_expr='doc_id == "doc_001"')

        assert count == 50

    def test_delete_by_doc_id(self):
        """测试按文档 ID 删除"""
        mock_client = MagicMock()
        mock_client.delete.return_value = {"delete_count": 5}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.delete_by_doc_id("doc_001")

        assert result["delete_count"] == 5
        mock_client.delete.assert_called_once_with(
            "test_collection",
            filter_expr='doc_id == "doc_001"',
        )

    def test_delete_by_ids(self):
        """测试按主键 ID 删除"""
        mock_client = MagicMock()
        mock_client.delete.return_value = {"delete_count": 3}

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        result = manager.delete_by_ids([1, 2, 3])

        assert result["delete_count"] == 3
        mock_client.delete.assert_called_once_with("test_collection", ids=[1, 2, 3])

    def test_get_by_doc_id(self):
        """测试根据文档 ID 获取"""
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"id": 1, "doc_id": "doc_001", "chunk_id": "c1"},
            {"id": 2, "doc_id": "doc_001", "chunk_id": "c2"},
        ]

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        results = manager.get_by_doc_id("doc_001")

        assert len(results) == 2
        mock_client.query.assert_called_once()

    def test_flush_is_noop(self):
        """测试 flush 是空操作（兼容性方法）"""
        mock_client = MagicMock()

        manager = CollectionManager(
            collection_name="test_collection",
            client=mock_client,
        )
        # flush 不应该抛出异常
        manager.flush()
