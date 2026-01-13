"""对话 Schema"""
from pydantic import Field
from app.schemas.base import BaseSchema


class ChatRequest(BaseSchema):
    """对话请求

    支持四种过滤场景：
    1. 只传 user_id          → 用户所有知识库
    2. user_id + knowledge_id → 指定知识库
    3. user_id + doc_id       → 直接定位文档（跨知识库）
    4. 三者都传               → 指定知识库下的指定文档
    """
    user_id: str = Field(..., alias="userId")
    knowledge_id: str | None = Field(default=None, alias="knowledgeId")
    doc_id: str | None = Field(default=None, alias="docId")
    question: str
    top_k: int = Field(default=5, alias="topK")
    stream: bool = False


class SourceInfo(BaseSchema):
    """来源信息"""
    doc_id: str = Field(..., alias="docId")
    chunk_id: str = Field(..., alias="chunkId")
    content: str
    score: float


class ChatResponse(BaseSchema):
    """对话响应"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str | None = Field(default=None, alias="knowledgeId")
    doc_id: str | None = Field(default=None, alias="docId")
    question: str
    answer: str
    sources: list[SourceInfo]
