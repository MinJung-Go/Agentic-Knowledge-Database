"""RAG Agent - 检索增强生成"""
from dataclasses import dataclass
from typing import AsyncGenerator

from core.agent.base import BaseAgent
from core.retrieval import HybridRetriever
from core.llm import AsyncLLMClient, prompt_manager


@dataclass
class RAGResponse:
    """RAG 响应"""
    answer: str
    sources: list[dict]
    usage: dict


class RAGAgent(BaseAgent):
    """RAG Agent

    整合检索召回和 LLM 生成，实现知识库问答
    """

    name: str = "rag"

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm_client: AsyncLLMClient | None = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm_client = llm_client or AsyncLLMClient()
        self.top_k = top_k
        super().__init__()

    async def run(self, input_data: dict) -> RAGResponse:
        """执行 RAG 流程

        Args:
            input_data: {"question": "用户问题", "filter": "可选过滤条件"}
        """
        question = input_data.get("question", "")
        filter_expr = input_data.get("filter")
        top_k = input_data.get("top_k", self.top_k)

        return await self.query(question, filter_expr, top_k)

    async def query(
        self,
        question: str,
        filter_expr: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """查询接口"""
        top_k = top_k or self.top_k

        search_results = self.retriever.search(
            query=question,
            top_k=top_k,
            filter_expr=filter_expr,
        )

        context = self._build_context(search_results)

        messages = prompt_manager.build_rag_messages(
            question=question,
            context=context,
        )

        result = await self.llm_client.chat(messages)

        sources = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "content": r.content[:200],
                "score": r.final_score,
            }
            for r in search_results
        ]

        return RAGResponse(
            answer=result.content,
            sources=sources,
            usage=result.usage,
        )

    async def stream_query(
        self,
        question: str,
        filter_expr: str | None = None,
        top_k: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式查询接口"""
        top_k = top_k or self.top_k

        search_results = self.retriever.search(
            query=question,
            top_k=top_k,
            filter_expr=filter_expr,
        )

        context = self._build_context(search_results)

        messages = prompt_manager.build_rag_messages(
            question=question,
            context=context,
        )

        async for chunk in self.llm_client.stream_chat(messages):
            yield chunk

    def _build_context(self, search_results: list) -> str:
        """构建上下文"""
        if not search_results:
            return "未找到相关文档。"

        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[文档 {i}]\n{result.content}")

        return "\n\n".join(context_parts)
