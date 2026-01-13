"""Milvus 向量数据库模块

提供 Milvus 客户端和 Collection 管理功能。

用法:
    from core.milvus import MilvusClient, CollectionManager, get_milvus_client

    # 使用全局客户端
    client = get_milvus_client()
    collections = client.list_collections()

    # 使用 Collection 管理器
    manager = CollectionManager("my_collection")
    manager.create()
    results = manager.search(query_embedding=[...])
"""

from core.milvus.client import (
    AsyncMilvusClient,
    MilvusClient,
    SearchResult,
    get_async_milvus_client,
    get_milvus_client,
)
from core.milvus.collection import CollectionConfig, CollectionManager

__all__ = [
    # 同步客户端
    "MilvusClient",
    "get_milvus_client",
    # 异步客户端
    "AsyncMilvusClient",
    "get_async_milvus_client",
    # 数据类型
    "SearchResult",
    # Collection 管理
    "CollectionManager",
    "CollectionConfig",
]
