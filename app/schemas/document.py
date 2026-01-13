"""文档管理 Schema"""
from datetime import datetime
from pydantic import Field
from app.schemas.base import BaseSchema


# ============ 通用 ============

class DocumentInfo(BaseSchema):
    """文档信息"""
    doc_id: str = Field(..., alias="docId")
    filename: str
    chunk_count: int = Field(..., alias="chunkCount")
    status: str
    created_at: datetime = Field(..., alias="createdAt")


# ============ 创建文档 ============

class CreateDocumentRequest(BaseSchema):
    """创建文档请求"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    metadata: dict | None = None


class CreateDocumentResponse(BaseSchema):
    """创建文档响应"""
    user_id: str = Field(..., alias="userId")
    doc_id: str = Field(..., alias="docId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    filename: str
    chunk_count: int = Field(..., alias="chunkCount")
    status: str
    created_at: datetime = Field(..., alias="createdAt")


# ============ 删除文档 ============

class DeleteDocumentRequest(BaseSchema):
    """删除文档请求"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    doc_id: str = Field(..., alias="docId")


class DeleteDocumentResponse(BaseSchema):
    """删除文档响应"""
    user_id: str = Field(..., alias="userId")
    doc_id: str = Field(..., alias="docId")
    deleted: bool
    message: str


# ============ 更新文档 ============

class UpdateDocumentRequest(BaseSchema):
    """更新文档请求"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    doc_id: str = Field(..., alias="docId")
    metadata: dict | None = None


class UpdateDocumentResponse(BaseSchema):
    """更新文档响应"""
    user_id: str = Field(..., alias="userId")
    doc_id: str = Field(..., alias="docId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    filename: str
    chunk_count: int = Field(..., alias="chunkCount")
    status: str
    updated_at: datetime = Field(..., alias="updatedAt")


# ============ 查询文档 ============

class QueryDocumentRequest(BaseSchema):
    """查询文档请求"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    page: int = 1
    page_size: int = Field(default=20, alias="pageSize")


class QueryDocumentResponse(BaseSchema):
    """查询文档响应"""
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    total: int
    page: int
    page_size: int = Field(..., alias="pageSize")
    documents: list[DocumentInfo]
