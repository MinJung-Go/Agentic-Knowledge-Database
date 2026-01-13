"""向量召回（语义检索）"""
from dataclasses import dataclass

from core.embedding import EmbeddingClient
from core.milvus import CollectionManager


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict


class VectorRetriever:
    """向量召回器

    基于向量相似度的语义检索
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        collection_manager: CollectionManager | None = None,
    ):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.collection_manager = collection_manager or CollectionManager()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[VectorSearchResult]:
        """向量搜索"""
        query_embedding = self.embedding_client.embed(query)

        milvus_results = self.collection_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["doc_id", "chunk_id", "content", "metadata"],
        )

        results = []
        for item in milvus_results:
            # 兼容字典格式和 SearchResult 对象
            if isinstance(item, dict):
                results.append(VectorSearchResult(
                    doc_id=item.get("doc_id", ""),
                    chunk_id=item.get("chunk_id", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    metadata=item.get("metadata", {}),
                ))
            else:
                # SearchResult 对象
                results.append(VectorSearchResult(
                    doc_id=item.data.get("doc_id", "") if item.data else "",
                    chunk_id=item.data.get("chunk_id", "") if item.data else "",
                    content=item.data.get("content", "") if item.data else "",
                    score=item.score,
                    metadata=item.data.get("metadata", {}) if item.data else {},
                ))

        return results

    def search_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[VectorSearchResult]:
        """通过向量直接搜索"""
        milvus_results = self.collection_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["doc_id", "chunk_id", "content", "metadata"],
        )

        results = []
        for item in milvus_results:
            # 兼容字典格式和 SearchResult 对象
            if isinstance(item, dict):
                results.append(VectorSearchResult(
                    doc_id=item.get("doc_id", ""),
                    chunk_id=item.get("chunk_id", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    metadata=item.get("metadata", {}),
                ))
            else:
                # SearchResult 对象
                results.append(VectorSearchResult(
                    doc_id=item.data.get("doc_id", "") if item.data else "",
                    chunk_id=item.data.get("chunk_id", "") if item.data else "",
                    content=item.data.get("content", "") if item.data else "",
                    score=item.score,
                    metadata=item.data.get("metadata", {}) if item.data else {},
                ))

        return results

    def search_by_doc_id(
        self,
        query: str,
        doc_id: str,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """在指定文档内搜索"""
        filter_expr = f'doc_id == "{doc_id}"'
        return self.search(query, top_k, filter_expr)
