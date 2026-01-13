"""VectorRetriever 单元测试"""
import pytest
from unittest.mock import MagicMock

from core.retrieval.vector import VectorRetriever, VectorSearchResult
from core.milvus.client import SearchResult


class TestVectorSearchResult:
    """VectorSearchResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = VectorSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="测试内容",
            score=0.95,
            metadata={"key": "value"},
        )
        assert result.doc_id == "doc_001"
        assert result.score == 0.95


class TestVectorRetriever:
    """VectorRetriever 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        mock_embedding = MagicMock()
        mock_collection = MagicMock()

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        assert retriever.embedding_client == mock_embedding
        assert retriever.collection_manager == mock_collection

    def test_search_basic(self):
        """测试基本搜索"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_milvus_result = SearchResult(
            id=1,
            score=0.95,
            data={
                "doc_id": "doc_001",
                "chunk_id": "chunk_001",
                "content": "测试内容",
                "metadata": {"key": "value"},
            },
        )

        mock_collection = MagicMock()
        mock_collection.search.return_value = [mock_milvus_result]

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        results = retriever.search("查询问题", top_k=10)

        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].doc_id == "doc_001"
        assert results[0].score == 0.95

    def test_search_with_filter(self):
        """测试带过滤条件的搜索"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_collection = MagicMock()
        mock_collection.search.return_value = []

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        retriever.search("查询", top_k=5, filter_expr='doc_id == "doc_001"')

        # 验证过滤条件被传递
        call_args = mock_collection.search.call_args
        assert call_args.kwargs["filter_expr"] == 'doc_id == "doc_001"'

    def test_search_no_results(self):
        """测试无结果"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_collection = MagicMock()
        mock_collection.search.return_value = []

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        results = retriever.search("查询")

        assert len(results) == 0

    def test_search_by_embedding(self):
        """测试直接使用向量搜索"""
        mock_embedding = MagicMock()

        mock_milvus_result = SearchResult(
            id=1,
            score=0.9,
            data={
                "doc_id": "doc_001",
                "chunk_id": "chunk_001",
                "content": "内容",
                "metadata": {},
            },
        )

        mock_collection = MagicMock()
        mock_collection.search.return_value = [mock_milvus_result]

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        query_embedding = [0.2] * 4096
        results = retriever.search_by_embedding(query_embedding, top_k=5)

        assert len(results) == 1
        # 不应该调用 embedding_client.embed
        mock_embedding.embed.assert_not_called()

    def test_search_by_embedding_with_filter(self):
        """测试向量搜索带过滤条件"""
        mock_embedding = MagicMock()
        mock_collection = MagicMock()
        mock_collection.search.return_value = []

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        retriever.search_by_embedding(
            query_embedding=[0.1] * 4096,
            top_k=10,
            filter_expr='metadata["user_id"] == "user1"',
        )

        call_args = mock_collection.search.call_args
        assert 'metadata["user_id"]' in call_args.kwargs["filter_expr"]

    def test_search_by_doc_id(self):
        """测试在指定文档内搜索"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_collection = MagicMock()
        mock_collection.search.return_value = []

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        retriever.search_by_doc_id("查询", doc_id="doc_001", top_k=5)

        call_args = mock_collection.search.call_args
        assert 'doc_id == "doc_001"' in call_args.kwargs["filter_expr"]

    def test_search_multiple_results(self):
        """测试多个结果"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_results = [
            SearchResult(
                id=i,
                score=0.9 - i * 0.1,
                data={
                    "doc_id": f"doc_00{i}",
                    "chunk_id": f"chunk_00{i}",
                    "content": f"内容{i}",
                    "metadata": {},
                },
            )
            for i in range(3)
        ]

        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_results

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        results = retriever.search("查询", top_k=10)

        assert len(results) == 3
        # 验证分数降序
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_handles_missing_fields(self):
        """测试处理缺失字段"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 4096

        mock_result = SearchResult(
            id=1,
            score=0.9,
            data={},  # 空数据
        )

        mock_collection = MagicMock()
        mock_collection.search.return_value = [mock_result]

        retriever = VectorRetriever(
            embedding_client=mock_embedding,
            collection_manager=mock_collection,
        )

        results = retriever.search("查询")

        assert len(results) == 1
        assert results[0].doc_id == ""
        assert results[0].content == ""
