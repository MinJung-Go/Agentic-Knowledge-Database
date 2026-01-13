"""文档管理 API"""
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.schemas.document import (
    CreateDocumentResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    UpdateDocumentResponse,
    QueryDocumentRequest,
    QueryDocumentResponse,
    DocumentInfo,
)
from core.parsers import MinerUParser, TextChunker
from core.embedding import EmbeddingClient
from core.milvus import CollectionManager

router = APIRouter(prefix="/knowledge/documents", tags=["documents"])

# 初始化核心组件
collection_manager = CollectionManager()
embedding_client = EmbeddingClient()
parser = MinerUParser()
chunker = TextChunker(chunk_size=500, chunk_overlap=50)


@router.post("/create", response_model=CreateDocumentResponse)
async def create_document(
    user_id: str = Form(..., alias="userId"),
    knowledge_id: str = Form(..., alias="knowledgeId"),
    file: UploadFile = File(...),
):
    """上传文档到知识库

    流程:
    1. MinerU 解析文档 -> Markdown
    2. TextChunker 分块
    3. EmbeddingClient 向量化
    4. CollectionManager 存储到 Milvus
    """
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    filename = file.filename or "unknown.pdf"
    now = datetime.now()

    try:
        # 1. 读取文件内容
        content = await file.read()

        # 检查空文件
        if not content:
            raise HTTPException(status_code=400, detail="文档解析失败，无有效内容")

        # 2. MinerU 解析文档（异步版本，避免事件循环冲突）
        parse_result = await parser.parse_bytes_async(content, filename)

        # 3. 文本分块
        chunks = chunker.chunk_text(
            parse_result.content,
            metadata={"doc_id": doc_id, "filename": filename},
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="文档解析失败，无有效内容")

        # 4. 批量向量化
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_client.embed_documents(texts)

        # 5. 存储到 Milvus
        doc_ids = [doc_id] * len(chunks)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "user_id": user_id,
                "knowledge_id": knowledge_id,
                "filename": filename,
                "chunk_index": chunk.index,
            }
            for chunk in chunks
        ]

        collection_manager.insert_batch(
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            contents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return CreateDocumentResponse(
            user_id=user_id,
            doc_id=doc_id,
            knowledge_id=knowledge_id,
            filename=filename,
            chunk_count=len(chunks),
            status="completed",
            created_at=now,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.post("/delete", response_model=DeleteDocumentResponse)
async def delete_document(request: DeleteDocumentRequest):
    """删除指定文档

    从 Milvus 中删除该文档的所有向量（需验证用户和知识库归属）
    """
    try:
        # 构建过滤条件（验证用户和知识库归属）
        filter_expr = (
            f'doc_id == "{request.doc_id}" and '
            f'metadata["user_id"] == "{request.user_id}" and '
            f'metadata["knowledge_id"] == "{request.knowledge_id}"'
        )

        # 先查询是否存在且有权限
        existing = collection_manager.query(filter_expr=filter_expr, limit=1)

        if not existing:
            return DeleteDocumentResponse(
                user_id=request.user_id,
                doc_id=request.doc_id,
                deleted=False,
                message="文档不存在或无权限",
            )

        # 执行删除（使用相同的过滤条件确保安全）
        result = collection_manager.delete(filter_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0

        return DeleteDocumentResponse(
            user_id=request.user_id,
            doc_id=request.doc_id,
            deleted=True,
            message=f"文档删除成功，共删除 {deleted_count} 个分块",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/update", response_model=UpdateDocumentResponse)
async def update_document(
    user_id: str = Form(..., alias="userId"),
    knowledge_id: str = Form(..., alias="knowledgeId"),
    doc_id: str = Form(..., alias="docId"),
    file: UploadFile = File(...),
):
    """更新指定文档

    流程:
    1. 验证文档归属权限
    2. 删除旧向量
    3. 重新解析、分块、向量化、存储
    """
    filename = file.filename or "unknown.pdf"
    now = datetime.now()

    try:
        # 1. 构建过滤条件（验证用户和知识库归属）
        filter_expr = (
            f'doc_id == "{doc_id}" and '
            f'metadata["user_id"] == "{user_id}" and '
            f'metadata["knowledge_id"] == "{knowledge_id}"'
        )

        # 验证文档是否存在且有权限
        existing = collection_manager.query(filter_expr=filter_expr, limit=1)

        if not existing:
            raise HTTPException(status_code=404, detail="文档不存在或无权限")

        # 2. 删除旧数据（使用带权限验证的过滤条件）
        collection_manager.delete(filter_expr)

        # 3. 读取新文件
        content = await file.read()

        # 检查空文件
        if not content:
            raise HTTPException(status_code=400, detail="文档解析失败，无有效内容")

        # 4. 解析文档
        parse_result = await parser.parse_bytes_async(content, filename)

        # 5. 分块
        chunks = chunker.chunk_text(
            parse_result.content,
            metadata={"doc_id": doc_id, "filename": filename},
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="文档解析失败，无有效内容")

        # 6. 向量化
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_client.embed_documents(texts)

        # 7. 存储
        doc_ids = [doc_id] * len(chunks)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "user_id": user_id,
                "knowledge_id": knowledge_id,
                "filename": filename,
                "chunk_index": chunk.index,
            }
            for chunk in chunks
        ]

        collection_manager.insert_batch(
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            contents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return UpdateDocumentResponse(
            user_id=user_id,
            doc_id=doc_id,
            knowledge_id=knowledge_id,
            filename=filename,
            chunk_count=len(chunks),
            status="completed",
            updated_at=now,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.post("/query", response_model=QueryDocumentResponse)
async def query_documents(request: QueryDocumentRequest):
    """查询知识库文档列表

    从 Milvus 中聚合查询文档信息
    """
    try:
        # 构建过滤条件
        filter_expr = f'metadata["user_id"] == "{request.user_id}" and metadata["knowledge_id"] == "{request.knowledge_id}"'

        # 查询所有匹配的 chunks
        results = collection_manager.search(
            query_embedding=[0.0] * 4096,  # 占位向量，仅用于过滤
            top_k=10000,
            filter_expr=filter_expr,
            output_fields=["doc_id", "metadata"],
        )

        # 聚合文档信息
        doc_map: dict[str, dict] = {}
        for result in results:
            doc_id = result.get("doc_id", "")
            if doc_id not in doc_map:
                metadata = result.get("metadata", {})
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_count": 0,
                    "status": "completed",
                    "created_at": datetime.now(),
                }
            doc_map[doc_id]["chunk_count"] += 1

        # 分页
        all_docs = list(doc_map.values())
        total = len(all_docs)
        start = (request.page - 1) * request.page_size
        end = start + request.page_size
        paged_docs = all_docs[start:end]

        documents = [
            DocumentInfo(
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                chunk_count=doc["chunk_count"],
                status=doc["status"],
                created_at=doc["created_at"],
            )
            for doc in paged_docs
        ]

        return QueryDocumentResponse(
            user_id=request.user_id,
            knowledge_id=request.knowledge_id,
            total=total,
            page=request.page,
            page_size=request.page_size,
            documents=documents,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
