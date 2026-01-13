"""混合召回（双路召回 + Rerank）"""
from dataclasses import dataclass

from core.retrieval.text import TextRetriever, TextSearchResult
from core.retrieval.vector import VectorRetriever, VectorSearchResult
from core.rerank import RerankClient


@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    doc_id: str
    chunk_id: str
    content: str
    text_score: float | None
    vector_score: float | None
    rerank_score: float | None
    final_score: float
    metadata: dict


class HybridRetriever:
    """混合召回器

    双路召回（文本 + 向量）+ Rerank 重排序
    """

    def __init__(
        self,
        text_retriever: TextRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        rerank_client: RerankClient | None = None,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.text_retriever = text_retriever
        self.vector_retriever = vector_retriever
        self.rerank_client = rerank_client or RerankClient()
        self.text_weight = text_weight
        self.vector_weight = vector_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        recall_k: int = 50,
        use_rerank: bool = True,
        filter_expr: str | None = None,
    ) -> list[HybridSearchResult]:
        """混合搜索"""
        candidates = {}

        if self.vector_retriever:
            vector_results = self.vector_retriever.search(query, recall_k, filter_expr)
            for result in vector_results:
                key = (result.doc_id, result.chunk_id)
                if key not in candidates:
                    candidates[key] = {
                        "doc_id": result.doc_id,
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "text_score": None,
                        "vector_score": None,
                    }
                candidates[key]["vector_score"] = result.score

        if self.text_retriever:
            text_results = self.text_retriever.search(query, recall_k)
            for result in text_results:
                key = (result.doc_id, result.chunk_id)
                if key not in candidates:
                    candidates[key] = {
                        "doc_id": result.doc_id,
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "text_score": None,
                        "vector_score": None,
                    }
                candidates[key]["text_score"] = result.score

        candidate_list = list(candidates.values())

        if use_rerank and self.rerank_client and candidate_list:
            documents = [c["content"] for c in candidate_list]
            rerank_result = self.rerank_client.rerank(query, documents, top_k=top_k)

            results = []
            for item in rerank_result.results:
                candidate = candidate_list[item.index]
                results.append(HybridSearchResult(
                    doc_id=candidate["doc_id"],
                    chunk_id=candidate["chunk_id"],
                    content=candidate["content"],
                    text_score=candidate["text_score"],
                    vector_score=candidate["vector_score"],
                    rerank_score=item.score,
                    final_score=item.score,
                    metadata=candidate["metadata"],
                ))
            return results

        for candidate in candidate_list:
            text_score = candidate["text_score"] or 0
            vector_score = candidate["vector_score"] or 0
            candidate["final_score"] = (
                self.text_weight * text_score + self.vector_weight * vector_score
            )

        candidate_list.sort(key=lambda x: x["final_score"], reverse=True)

        results = []
        for candidate in candidate_list[:top_k]:
            results.append(HybridSearchResult(
                doc_id=candidate["doc_id"],
                chunk_id=candidate["chunk_id"],
                content=candidate["content"],
                text_score=candidate["text_score"],
                vector_score=candidate["vector_score"],
                rerank_score=None,
                final_score=candidate["final_score"],
                metadata=candidate["metadata"],
            ))

        return results

    def vector_only_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[HybridSearchResult]:
        """仅向量召回"""
        if not self.vector_retriever:
            return []

        results = self.vector_retriever.search(query, top_k, filter_expr)

        return [
            HybridSearchResult(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                content=r.content,
                text_score=None,
                vector_score=r.score,
                rerank_score=None,
                final_score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

    def text_only_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """仅文本召回"""
        if not self.text_retriever:
            return []

        results = self.text_retriever.search(query, top_k)

        return [
            HybridSearchResult(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                content=r.content,
                text_score=r.score,
                vector_score=None,
                rerank_score=None,
                final_score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
