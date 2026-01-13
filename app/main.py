"""FastAPI 应用入口"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from configs.settings import settings
from core.milvus import MilvusClient, CollectionManager

# 全局组件
milvus_client = MilvusClient()
collection_manager = CollectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化（新版 pymilvus MilvusClient API 使用懒加载，无需显式连接）
    # 访问 client 属性会自动建立连接
    _ = milvus_client.client  # 触发连接

    # 确保 Collection 存在（create 方法已包含索引创建）
    if not collection_manager.exists():
        collection_manager.create()

    yield

    # 关闭时清理连接
    milvus_client.close()


app = FastAPI(
    title=settings.app_name,
    description="企业级 AI 知识库系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 注册路由
from app.routers import document_router, chat_router
app.include_router(document_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """根路径"""
    return {"app": settings.app_name, "status": "running"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
