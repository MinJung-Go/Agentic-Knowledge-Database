"""Milvus Collection 管理

提供知识库 Collection 的创建、数据操作和搜索功能。
"""
import logging
from dataclasses import dataclass
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema

from configs.settings import settings
from core.milvus.client import MilvusClient, SearchResult, get_milvus_client

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Collection 配置

    Attributes:
        name: 集合名称
        dimension: 向量维度（从配置读取，0.6B=1024, 8B=4096）
        metric_type: 相似度计算方式（L2 或 IP，Milvus 2.2.x 不支持 COSINE）
        description: 集合描述
    """

    name: str
    dimension: int | None = None  # None 时使用 settings.embedding_dimension
    metric_type: str = "L2"
    description: str = ""

    def __post_init__(self):
        if self.dimension is None:
            self.dimension = settings.embedding_dimension


class CollectionManager:
    """Collection 管理器

    管理知识库 Collection 的创建、索引和数据操作。

    用法:
        # 创建管理器
        manager = CollectionManager("my_knowledge_base")

        # 创建 Collection
        manager.create()

        # 插入数据
        manager.insert(doc_id="doc1", chunk_id="chunk1", content="...", embedding=[...])

        # 搜索
        results = manager.search(query_embedding=[...], top_k=10)
    """

    # 默认 schema 字段定义
    DEFAULT_FIELDS = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # 会被动态替换
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]

    def __init__(
        self,
        collection_name: str | None = None,
        dimension: int | None = None,
        metric_type: str = "L2",  # Milvus 2.2.x 只支持 L2 和 IP
        client: MilvusClient | None = None,
    ):
        """初始化 Collection 管理器

        Args:
            collection_name: 集合名称，默认使用配置中的名称
            dimension: 向量维度，默认从 settings.embedding_dimension 读取
            metric_type: 相似度计算方式 (COSINE, L2, IP)
            client: Milvus 客户端，默认使用全局客户端
        """
        self.collection_name = collection_name or settings.milvus_collection
        self.dimension = dimension if dimension is not None else settings.embedding_dimension
        self.metric_type = metric_type
        self._client = client

    @property
    def client(self) -> MilvusClient:
        """获取 Milvus 客户端"""
        if self._client is None:
            self._client = get_milvus_client()
        return self._client

    # ==================== Collection 操作 ====================

    def exists(self) -> bool:
        """检查 Collection 是否存在"""
        return self.client.has_collection(self.collection_name)

    def create(
        self,
        description: str = "Knowledge base collection",
        drop_if_exists: bool = False,
    ) -> bool:
        """创建 Collection

        Args:
            description: 集合描述
            drop_if_exists: 如果存在是否删除重建

        Returns:
            是否成功创建（已存在返回 False）
        """
        if drop_if_exists and self.exists():
            self.drop()

        if self.exists():
            logger.info(f"Collection {self.collection_name} 已存在")
            return False

        # 构建 schema（启用动态字段，允许插入额外字段）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
        ]
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=True,  # 允许插入额外字段
        )

        # 创建索引参数
        index_params = self.client.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        # 创建 Collection
        self.client.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(f"Collection {self.collection_name} 创建成功 (dim={self.dimension})")
        return True

    def drop(self) -> None:
        """删除 Collection"""
        self.client.drop_collection(self.collection_name)

    def stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return self.client.get_collection_stats(self.collection_name)

    # ==================== 数据操作 ====================

    def insert(
        self,
        data: list[dict[str, Any]] | dict[str, Any],
    ) -> dict[str, Any]:
        """插入数据

        支持两种格式:
        1. 字典列表: [{"doc_id": ..., "content": ..., "vector": ...}, ...]
        2. 单个字典: {"doc_id": ..., "content": ..., "vector": ...}

        Args:
            data: 要插入的数据

        Returns:
            插入结果
        """
        if isinstance(data, dict):
            data = [data]

        # 标准化字段名和添加默认值
        normalized_data = []
        for item in data:
            normalized = dict(item)
            # vector -> embedding
            if "vector" in normalized and "embedding" not in normalized:
                normalized["embedding"] = normalized.pop("vector")
            # 确保 metadata 字段存在
            if "metadata" not in normalized:
                normalized["metadata"] = {}
            normalized_data.append(normalized)

        return self.client.insert(self.collection_name, normalized_data)

    def insert_single(
        self,
        doc_id: str,
        chunk_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """插入单条数据（显式参数版本）

        Args:
            doc_id: 文档 ID
            chunk_id: 分块 ID
            content: 文本内容
            embedding: 向量
            metadata: 元数据

        Returns:
            插入结果
        """
        return self.insert([{
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
        }])

    def insert_batch(
        self,
        doc_ids: list[str],
        chunk_ids: list[str],
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """批量插入数据（显式参数版本）

        Args:
            doc_ids: 文档 ID 列表
            chunk_ids: 分块 ID 列表
            contents: 文本内容列表
            embeddings: 向量列表
            metadatas: 元数据列表

        Returns:
            插入结果
        """
        if metadatas is None:
            metadatas = [{}] * len(doc_ids)

        data = [
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            }
            for doc_id, chunk_id, content, embedding, metadata in zip(
                doc_ids, chunk_ids, contents, embeddings, metadatas
            )
        ]

        return self.client.insert(self.collection_name, data)

    def flush(self) -> None:
        """刷新数据到磁盘（兼容性方法，新 API 自动 flush）"""
        # 新版 MilvusClient 在 insert 后自动 flush
        pass

    def delete_by_doc_id(self, doc_id: str) -> dict[str, Any]:
        """按文档 ID 删除

        Args:
            doc_id: 文档 ID

        Returns:
            删除结果
        """
        return self.client.delete(
            self.collection_name,
            filter_expr=f'doc_id == "{doc_id}"',
        )

    def delete_by_ids(self, ids: list[int]) -> dict[str, Any]:
        """按主键 ID 删除

        Args:
            ids: 主键 ID 列表

        Returns:
            删除结果
        """
        return self.client.delete(self.collection_name, ids=ids)

    def delete(self, filter_expr: str) -> dict[str, Any]:
        """按过滤条件删除

        Args:
            filter_expr: Milvus 过滤表达式，如 'doc_id == "xxx" and metadata["user_id"] == "yyy"'

        Returns:
            删除结果
        """
        return self.client.delete(self.collection_name, filter_expr=filter_expr)

    # ==================== 搜索操作 ====================

    def search(
        self,
        query_embedding: list[float] | None = None,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        score_threshold: float | None = None,
        *,
        vector: list[float] | None = None,  # 别名，向后兼容
    ) -> list[dict[str, Any]]:
        """向量搜索

        Args:
            query_embedding: 查询向量
            vector: 查询向量（别名，向后兼容）
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            output_fields: 返回字段列表
            score_threshold: 分数阈值（过滤低于阈值的结果）

        Returns:
            搜索结果列表（字典格式，包含 score 和字段数据）
        """
        # 支持 vector 参数别名
        query_vector = query_embedding or vector
        if query_vector is None:
            raise ValueError("必须提供 query_embedding 或 vector 参数")

        if output_fields is None:
            output_fields = ["doc_id", "chunk_id", "content", "metadata"]

        results = self.client.search_single(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )

        # 应用分数阈值过滤
        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]

        # 转换为字典格式，便于测试和使用
        return [
            {"id": r.id, "score": r.score, **r.data}
            for r in results
        ]

    def search_batch(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[list[SearchResult]]:
        """批量向量搜索

        Args:
            query_embeddings: 查询向量列表
            top_k: 每个查询返回结果数量
            filter_expr: 过滤表达式
            output_fields: 返回字段列表

        Returns:
            搜索结果列表的列表
        """
        if output_fields is None:
            output_fields = ["doc_id", "chunk_id", "content", "metadata"]

        return self.client.search(
            collection_name=self.collection_name,
            query_vectors=query_embeddings,
            limit=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )

    def query(
        self,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """标量查询

        Args:
            filter_expr: 过滤表达式
            output_fields: 返回字段列表
            limit: 返回数量限制

        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "chunk_id", "content", "metadata"]

        return self.client.query(
            collection_name=self.collection_name,
            filter_expr=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    def count(self, filter_expr: str | None = None) -> int:
        """统计数量

        Args:
            filter_expr: 过滤表达式

        Returns:
            记录数量
        """
        if filter_expr:
            results = self.client.query(
                collection_name=self.collection_name,
                filter_expr=filter_expr,
                output_fields=["count(*)"],
            )
            return results[0].get("count(*)", 0) if results else 0

        stats = self.stats()
        return stats.get("row_count", 0)

    def get_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """根据文档 ID 获取所有分块

        Args:
            doc_id: 文档 ID

        Returns:
            分块数据列表
        """
        return self.query(filter_expr=f'doc_id == "{doc_id}"')
