"""对话 API"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, SourceInfo
from core.embedding import EmbeddingClient
from core.milvus import CollectionManager
from core.rerank import RerankClient
from core.llm import AsyncLLMClient, prompt_manager

router = APIRouter(prefix="/knowledge", tags=["chat"])


def build_filter_expr(user_id: str, knowledge_id: str | None, doc_id: str | None) -> str:
    """根据传入参数构建过滤表达式

    支持四种过滤场景：
    1. 只传 user_id          → 用户所有知识库
    2. user_id + knowledge_id → 指定知识库
    3. user_id + doc_id       → 直接定位文档（跨知识库）
    4. 三者都传               → 指定知识库下的指定文档
    """
    conditions = [f'metadata["user_id"] == "{user_id}"']

    if knowledge_id:
        conditions.append(f'metadata["knowledge_id"] == "{knowledge_id}"')

    if doc_id:
        conditions.append(f'doc_id == "{doc_id}"')

    return " and ".join(conditions)

# 初始化核心组件
embedding_client = EmbeddingClient()
collection_manager = CollectionManager()
rerank_client = RerankClient()
llm_client = AsyncLLMClient()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """知识库问答

    流程:
    1. 向量召回: EmbeddingClient + CollectionManager
    2. 重排序: RerankClient
    3. 生成回答: AsyncLLMClient
    """
    try:
        # 1. 构建过滤条件（支持 user_id / knowledge_id / doc_id 灵活组合）
        filter_expr = build_filter_expr(request.user_id, request.knowledge_id, request.doc_id)

        # 2. 向量召回
        query_embedding = embedding_client.embed(request.question)
        search_results = collection_manager.search(
            query_embedding=query_embedding,
            top_k=request.top_k * 3,  # 召回更多用于重排序
            filter_expr=filter_expr,
            output_fields=["doc_id", "chunk_id", "content", "metadata"],
        )

        if not search_results:
            return ChatResponse(
                user_id=request.user_id,
                knowledge_id=request.knowledge_id,
                doc_id=request.doc_id,
                question=request.question,
                answer="抱歉，未找到相关文档，无法回答您的问题。",
                sources=[],
            )

        # 3. Rerank 重排序
        documents = [r.get("content", "") for r in search_results]
        rerank_result = rerank_client.rerank(
            query=request.question,
            documents=documents,
            top_k=request.top_k,
        )

        # 4. 构建上下文
        context_parts = []
        sources = []
        for item in rerank_result.results:
            result = search_results[item.index]
            content = result.get("content", "")
            context_parts.append(f"[文档 {len(sources) + 1}]\n{content}")
            sources.append(SourceInfo(
                doc_id=result.get("doc_id", ""),
                chunk_id=result.get("chunk_id", ""),
                content=content[:200],
                score=item.score,
            ))

        context = "\n\n".join(context_parts)

        # 5. LLM 生成回答
        messages = prompt_manager.build_rag_messages(
            question=request.question,
            context=context,
        )
        result = await llm_client.chat(messages)

        return ChatResponse(
            user_id=request.user_id,
            knowledge_id=request.knowledge_id,
            doc_id=request.doc_id,
            question=request.question,
            answer=result.content,
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式知识库问答

    流程同 /chat，但使用流式输出
    """
    async def generate():
        try:
            # 1. 构建过滤条件（支持 user_id / knowledge_id / doc_id 灵活组合）
            filter_expr = build_filter_expr(request.user_id, request.knowledge_id, request.doc_id)

            # 2. 向量召回
            query_embedding = embedding_client.embed(request.question)
            search_results = collection_manager.search(
                query_embedding=query_embedding,
                top_k=request.top_k * 3,
                filter_expr=filter_expr,
                output_fields=["doc_id", "chunk_id", "content", "metadata"],
            )

            if not search_results:
                yield "data: 抱歉，未找到相关文档，无法回答您的问题。\n\n"
                yield "data: [DONE]\n\n"
                return

            # 3. Rerank 重排序
            documents = [r.get("content", "") for r in search_results]
            rerank_result = rerank_client.rerank(
                query=request.question,
                documents=documents,
                top_k=request.top_k,
            )

            # 4. 构建上下文
            context_parts = []
            for item in rerank_result.results:
                result = search_results[item.index]
                content = result.get("content", "")
                context_parts.append(f"[文档 {len(context_parts) + 1}]\n{content}")

            context = "\n\n".join(context_parts)

            # 5. 流式生成
            messages = prompt_manager.build_rag_messages(
                question=request.question,
                context=context,
            )

            async for chunk in llm_client.stream_chat(messages):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
