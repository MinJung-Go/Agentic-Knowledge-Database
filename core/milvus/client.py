"""Milvus 向量数据库客户端

使用 pymilvus.MilvusClient API，提供同步和异步两种接口。
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from pymilvus import AsyncMilvusClient as PyAsyncMilvusClient
from pymilvus import MilvusClient as PyMilvusClient

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""

    id: int | str
    score: float
    data: dict = field(default_factory=dict)


class MilvusClient:
    """Milvus 向量数据库客户端

    使用 pymilvus.MilvusClient API，提供简洁的接口。

    用法:
        # 使用默认配置
        client = MilvusClient()

        # 自定义配置
        client = MilvusClient(uri="http://192.168.25.177:19530")

        # 列出所有 collections
        collections = client.list_collections()

        # 搜索
        results = client.search("my_collection", query_vector, limit=10)
    """

    def __init__(
        self,
        uri: str | None = None,
        token: str = "",
        timeout: float | None = None,
    ):
        """初始化 Milvus 客户端

        Args:
            uri: Milvus 服务地址，格式: http://host:port
            token: 访问令牌（如果 Milvus 启用了认证）
            timeout: 连接超时时间（秒）
        """
        # 构建 URI
        if uri is None:
            host = settings.milvus_host.replace("http://", "").replace("https://", "")
            port = settings.milvus_port
            uri = f"http://{host}:{port}"

        self.uri = uri
        self.token = token
        self.timeout = timeout
        self._client: PyMilvusClient | None = None

    @property
    def client(self) -> PyMilvusClient:
        """获取底层 pymilvus 客户端（懒加载）"""
        if self._client is None:
            self._client = PyMilvusClient(uri=self.uri, token=self.token)
            logger.info(f"已连接到 Milvus: {self.uri}")
        return self._client

    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus 连接已关闭")

    # ==================== Collection 操作 ====================

    def list_collections(self) -> list[str]:
        """列出所有 Collection"""
        return self.client.list_collections()

    def has_collection(self, collection_name: str) -> bool:
        """检查 Collection 是否存在"""
        return self.client.has_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        **kwargs,
    ) -> None:
        """创建 Collection（简化版，使用默认 schema）

        Args:
            collection_name: 集合名称
            dimension: 向量维度
            metric_type: 相似度计算方式 (L2, IP)。注意: Milvus 2.2.x 不支持 COSINE
            **kwargs: 其他参数传递给 pymilvus
        """
        if self.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} 已存在")
            return

        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            **kwargs,
        )
        logger.info(f"Collection {collection_name} 创建成功")

    def drop_collection(self, collection_name: str) -> None:
        """删除 Collection"""
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Collection {collection_name} 已删除")

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """获取 Collection 统计信息"""
        stats = self.client.get_collection_stats(collection_name)
        return {
            "name": collection_name,
            "row_count": stats.get("row_count", 0),
        }

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        """获取 Collection 详细信息"""
        return self.client.describe_collection(collection_name)

    # ==================== 数据操作 ====================

    def insert(
        self,
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """插入数据

        Args:
            collection_name: 集合名称
            data: 数据列表，每个元素是一个字典

        Returns:
            插入结果
        """
        result = self.client.insert(collection_name=collection_name, data=data)
        self.client.flush(collection_name)
        logger.debug(f"插入 {len(data)} 条数据到 {collection_name}")
        return result

    def upsert(
        self,
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """更新或插入数据

        Args:
            collection_name: 集合名称
            data: 数据列表

        Returns:
            操作结果
        """
        result = self.client.upsert(collection_name=collection_name, data=data)
        self.client.flush(collection_name)
        return result

    def delete(
        self,
        collection_name: str,
        ids: list[int | str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """删除数据

        Args:
            collection_name: 集合名称
            ids: 要删除的 ID 列表
            filter_expr: 过滤表达式

        Returns:
            删除结果
        """
        if ids is not None:
            result = self.client.delete(collection_name=collection_name, ids=ids)
        elif filter_expr is not None:
            result = self.client.delete(collection_name=collection_name, filter=filter_expr)
        else:
            raise ValueError("必须指定 ids 或 filter_expr")

        return result

    # ==================== 搜索操作 ====================

    def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """向量搜索

        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            limit: 返回结果数量
            filter_expr: 过滤表达式
            output_fields: 返回字段列表
            **kwargs: 其他搜索参数

        Returns:
            搜索结果列表
        """
        results = self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            **kwargs,
        )

        # 转换为 SearchResult 对象
        search_results = []
        for hits in results:
            batch_results = []
            for hit in hits:
                entity = hit.get("entity", {})
                batch_results.append(
                    SearchResult(
                        id=hit.get("id"),
                        score=hit.get("distance", 0.0),
                        data=entity,
                    )
                )
            search_results.append(batch_results)

        return search_results

    def search_single(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """单向量搜索（便捷方法）

        Args:
            collection_name: 集合名称
            query_vector: 单个查询向量
            limit: 返回结果数量
            filter_expr: 过滤表达式
            output_fields: 返回字段列表

        Returns:
            搜索结果列表
        """
        results = self.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            limit=limit,
            filter_expr=filter_expr,
            output_fields=output_fields,
            **kwargs,
        )
        return results[0] if results else []

    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """标量查询

        Args:
            collection_name: 集合名称
            filter_expr: 过滤表达式
            output_fields: 返回字段列表
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            查询结果列表
        """
        return self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            limit=limit,
            offset=offset,
        )

    # ==================== 上下文管理 ====================

    def __enter__(self) -> "MilvusClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# 全局客户端实例（懒加载）
_default_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """获取全局 Milvus 客户端实例"""
    global _default_client
    if _default_client is None:
        _default_client = MilvusClient()
    return _default_client


# ============================================================================
# 异步客户端
# ============================================================================


def _build_uri(uri: str | None) -> str:
    """构建 Milvus URI"""
    if uri is None:
        host = settings.milvus_host.replace("http://", "").replace("https://", "")
        port = settings.milvus_port
        return f"http://{host}:{port}"
    return uri


class AsyncMilvusClient:
    """Milvus 异步客户端

    使用 pymilvus.AsyncMilvusClient API，提供异步接口。

    注意: pymilvus 的 AsyncMilvusClient API 比同步版本功能少，
    仅支持核心的 CRUD 和搜索操作。如需完整功能请使用同步客户端。

    用法:
        async with AsyncMilvusClient() as client:
            # 搜索
            results = await client.search("my_collection", [query_vector], limit=10)

            # 插入
            await client.insert("my_collection", [{"id": 1, "vector": [...]}])
    """

    def __init__(
        self,
        uri: str | None = None,
        token: str = "",
        timeout: float | None = None,
    ):
        """初始化异步 Milvus 客户端

        Args:
            uri: Milvus 服务地址，格式: http://host:port
            token: 访问令牌（如果 Milvus 启用了认证）
            timeout: 连接超时时间（秒）
        """
        self.uri = _build_uri(uri)
        self.token = token
        self.timeout = timeout
        self._client: PyAsyncMilvusClient | None = None

    @property
    def client(self) -> PyAsyncMilvusClient:
        """获取底层 pymilvus 异步客户端（懒加载）"""
        if self._client is None:
            self._client = PyAsyncMilvusClient(uri=self.uri, token=self.token)
            logger.info(f"已连接到 Milvus (async): {self.uri}")
        return self._client

    async def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Milvus 异步连接已关闭")

    # ==================== Collection 操作 ====================

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        **kwargs,
    ) -> None:
        """创建 Collection"""
        await self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            **kwargs,
        )
        logger.info(f"Collection {collection_name} 创建成功")

    async def drop_collection(self, collection_name: str) -> None:
        """删除 Collection"""
        await self.client.drop_collection(collection_name)
        logger.info(f"Collection {collection_name} 已删除")

    async def load_collection(self, collection_name: str) -> None:
        """加载 Collection 到内存"""
        await self.client.load_collection(collection_name)

    async def release_collection(self, collection_name: str) -> None:
        """释放 Collection"""
        await self.client.release_collection(collection_name)

    # ==================== 数据操作 ====================

    async def insert(
        self,
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """插入数据"""
        result = await self.client.insert(collection_name=collection_name, data=data)
        logger.debug(f"插入 {len(data)} 条数据到 {collection_name}")
        return result

    async def upsert(
        self,
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """更新或插入数据"""
        result = await self.client.upsert(collection_name=collection_name, data=data)
        return result

    async def delete(
        self,
        collection_name: str,
        ids: list[int | str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """删除数据"""
        if ids is not None:
            result = await self.client.delete(collection_name=collection_name, ids=ids)
        elif filter_expr is not None:
            result = await self.client.delete(collection_name=collection_name, filter=filter_expr)
        else:
            raise ValueError("必须指定 ids 或 filter_expr")
        return result

    async def get(
        self,
        collection_name: str,
        ids: list[int | str],
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """根据 ID 获取数据"""
        return await self.client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=output_fields or ["*"],
        )

    # ==================== 搜索操作 ====================

    async def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """向量搜索"""
        results = await self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            **kwargs,
        )

        # 转换为 SearchResult 对象
        search_results = []
        for hits in results:
            batch_results = []
            for hit in hits:
                entity = hit.get("entity", {})
                batch_results.append(
                    SearchResult(
                        id=hit.get("id"),
                        score=hit.get("distance", 0.0),
                        data=entity,
                    )
                )
            search_results.append(batch_results)

        return search_results

    async def search_single(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """单向量搜索（便捷方法）"""
        results = await self.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            limit=limit,
            filter_expr=filter_expr,
            output_fields=output_fields,
            **kwargs,
        )
        return results[0] if results else []

    async def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """标量查询"""
        return await self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            limit=limit,
            offset=offset,
        )

    # ==================== 上下文管理 ====================

    async def __aenter__(self) -> "AsyncMilvusClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# 全局异步客户端实例（懒加载）
_default_async_client: AsyncMilvusClient | None = None


def get_async_milvus_client() -> AsyncMilvusClient:
    """获取全局异步 Milvus 客户端实例"""
    global _default_async_client
    if _default_async_client is None:
        _default_async_client = AsyncMilvusClient()
    return _default_async_client
