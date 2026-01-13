"""HybridRetriever 单元测试"""
import pytest
from unittest.mock import MagicMock

from core.retrieval.hybrid import HybridRetriever, HybridSearchResult
from core.retrieval.text import TextSearchResult
from core.retrieval.vector import VectorSearchResult
from core.rerank.client import RerankResult, RerankItem


class TestHybridSearchResult:
    """HybridSearchResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = HybridSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="测试内容",
            text_score=0.8,
            vector_score=0.9,
            rerank_score=0.95,
            final_score=0.95,
            metadata={},
        )
        assert result.doc_id == "doc_001"
        assert result.final_score == 0.95


class TestHybridRetriever:
    """HybridRetriever 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        retriever = HybridRetriever()
        assert retriever.text_weight == 0.3
        assert retriever.vector_weight == 0.7

    def test_init_custom(self):
        """测试自定义初始化"""
        mock_text = MagicMock()
        mock_vector = MagicMock()
        mock_rerank = MagicMock()

        retriever = HybridRetriever(
            text_retriever=mock_text,
            vector_retriever=mock_vector,
            rerank_client=mock_rerank,
            text_weight=0.4,
            vector_weight=0.6,
        )

        assert retriever.text_weight == 0.4
        assert retriever.vector_weight == 0.6

    def test_search_with_rerank(self):
        """测试带 Rerank 的混合搜索"""
        # Mock vector retriever
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="向量召回内容",
                score=0.9,
                metadata={},
            )
        ]

        # Mock rerank client
        mock_rerank = MagicMock()
        mock_rerank.rerank.return_value = RerankResult(
            results=[RerankItem(index=0, score=0.95, text="向量召回内容")],
            usage={},
        )

        retriever = HybridRetriever(
            vector_retriever=mock_vector,
            rerank_client=mock_rerank,
        )

        results = retriever.search("查询", top_k=5, use_rerank=True)

        assert len(results) == 1
        assert results[0].rerank_score == 0.95
        assert results[0].final_score == 0.95

    def test_search_without_rerank(self):
        """测试不使用 Rerank"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="内容1",
                score=0.9,
                metadata={},
            ),
            VectorSearchResult(
                doc_id="doc_002",
                chunk_id="chunk_002",
                content="内容2",
                score=0.8,
                metadata={},
            ),
        ]

        retriever = HybridRetriever(
            vector_retriever=mock_vector,
            text_weight=0.3,
            vector_weight=0.7,
        )

        results = retriever.search("查询", top_k=5, use_rerank=False)

        assert len(results) == 2
        assert results[0].rerank_score is None
        # final_score 应该是加权分数
        assert results[0].final_score == 0.7 * 0.9  # vector_weight * vector_score

    def test_search_dual_recall(self):
        """测试双路召回"""
        # Mock text retriever
        mock_text = MagicMock()
        mock_text.search.return_value = [
            TextSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="文本召回内容",
                score=0.8,
                metadata={},
            )
        ]

        # Mock vector retriever
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="向量召回内容",
                score=0.9,
                metadata={},
            ),
            VectorSearchResult(
                doc_id="doc_002",
                chunk_id="chunk_002",
                content="仅向量召回",
                score=0.7,
                metadata={},
            ),
        ]

        retriever = HybridRetriever(
            text_retriever=mock_text,
            vector_retriever=mock_vector,
            text_weight=0.3,
            vector_weight=0.7,
        )

        results = retriever.search("查询", use_rerank=False)

        # 应该有 2 个结果（doc_001 合并，doc_002 仅向量）
        assert len(results) == 2
        # doc_001 应该同时有 text_score 和 vector_score
        doc_001_result = next(r for r in results if r.doc_id == "doc_001")
        assert doc_001_result.text_score is not None
        assert doc_001_result.vector_score is not None

    def test_search_with_filter(self):
        """测试带过滤条件的搜索"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = []

        retriever = HybridRetriever(vector_retriever=mock_vector)
        retriever.search("查询", filter_expr='doc_id == "doc_001"', use_rerank=False)

        mock_vector.search.assert_called_once()
        call_args = mock_vector.search.call_args
        # search(query, recall_k, filter_expr) 是位置参数调用
        assert call_args[0][2] == 'doc_id == "doc_001"'

    def test_search_empty_results(self):
        """测试空结果"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = []

        retriever = HybridRetriever(vector_retriever=mock_vector)
        results = retriever.search("查询", use_rerank=False)

        assert len(results) == 0

    def test_vector_only_search(self):
        """测试仅向量召回"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="内容",
                score=0.9,
                metadata={},
            )
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector)
        results = retriever.vector_only_search("查询", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], HybridSearchResult)
        assert results[0].text_score is None
        assert results[0].vector_score == 0.9

    def test_vector_only_search_no_retriever(self):
        """测试无向量检索器时的向量搜索"""
        retriever = HybridRetriever()
        results = retriever.vector_only_search("查询")

        assert len(results) == 0

    def test_text_only_search(self):
        """测试仅文本召回"""
        mock_text = MagicMock()
        mock_text.search.return_value = [
            TextSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="内容",
                score=0.85,
                metadata={},
            )
        ]

        retriever = HybridRetriever(text_retriever=mock_text)
        results = retriever.text_only_search("查询", top_k=5)

        assert len(results) == 1
        assert results[0].vector_score is None
        assert results[0].text_score == 0.85

    def test_text_only_search_no_retriever(self):
        """测试无文本检索器时的文本搜索"""
        retriever = HybridRetriever()
        results = retriever.text_only_search("查询")

        assert len(results) == 0

    def test_search_result_sorted_by_final_score(self):
        """测试结果按 final_score 排序"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="内容1",
                score=0.7,
                metadata={},
            ),
            VectorSearchResult(
                doc_id="doc_002",
                chunk_id="chunk_002",
                content="内容2",
                score=0.9,
                metadata={},
            ),
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector)
        results = retriever.search("查询", use_rerank=False)

        # 应该按分数降序排列
        assert results[0].doc_id == "doc_002"
        assert results[1].doc_id == "doc_001"

    def test_search_respects_top_k(self):
        """测试 top_k 限制"""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            VectorSearchResult(
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                content=f"内容{i}",
                score=0.9 - i * 0.1,
                metadata={},
            )
            for i in range(10)
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector)
        results = retriever.search("查询", top_k=3, use_rerank=False)

        assert len(results) == 3
