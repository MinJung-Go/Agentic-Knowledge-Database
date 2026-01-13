"""RAGAgent 单元测试"""
import pytest
from unittest.mock import MagicMock, AsyncMock

from core.agent.rag import RAGAgent, RAGResponse
from core.retrieval.hybrid import HybridSearchResult
from core.llm.client import ChatResult


class TestRAGResponse:
    """RAGResponse 数据类测试"""

    def test_creation(self):
        """测试创建"""
        response = RAGResponse(
            answer="AI 生成的回答",
            sources=[{"doc_id": "doc_001", "content": "来源内容"}],
            usage={"total_tokens": 100},
        )
        assert response.answer == "AI 生成的回答"
        assert len(response.sources) == 1


class TestRAGAgent:
    """RAGAgent 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        agent = RAGAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
            top_k=5,
        )

        assert agent.retriever == mock_retriever
        assert agent.llm_client == mock_llm
        assert agent.top_k == 5

    def test_name(self):
        """测试 Agent 名称"""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)

        assert agent.name == "rag"

    @pytest.mark.asyncio
    async def test_run(self):
        """测试 run 方法"""
        mock_search_result = HybridSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="相关文档内容",
            text_score=None,
            vector_score=0.9,
            rerank_score=0.95,
            final_score=0.95,
            metadata={},
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_search_result]

        mock_chat_result = ChatResult(
            content="AI 生成的回答",
            usage={"total_tokens": 100},
            finish_reason="stop",
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=mock_chat_result)

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        result = await agent.run({"question": "测试问题"})

        assert isinstance(result, RAGResponse)
        assert result.answer == "AI 生成的回答"
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_run_with_filter(self):
        """测试带过滤条件的 run"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=ChatResult(
            content="回答",
            usage={},
            finish_reason="stop",
        ))

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        await agent.run({
            "question": "问题",
            "filter": 'doc_id == "doc_001"',
        })

        call_args = mock_retriever.search.call_args
        assert call_args.kwargs["filter_expr"] == 'doc_id == "doc_001"'

    @pytest.mark.asyncio
    async def test_query(self):
        """测试 query 方法"""
        mock_search_results = [
            HybridSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="文档1内容" * 50,  # 超过 200 字符
                text_score=None,
                vector_score=0.9,
                rerank_score=None,
                final_score=0.9,
                metadata={},
            ),
            HybridSearchResult(
                doc_id="doc_002",
                chunk_id="chunk_002",
                content="文档2内容",
                text_score=None,
                vector_score=0.8,
                rerank_score=None,
                final_score=0.8,
                metadata={},
            ),
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = mock_search_results

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=ChatResult(
            content="综合回答",
            usage={"total_tokens": 150},
            finish_reason="stop",
        ))

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        result = await agent.query("测试问题", top_k=2)

        assert result.answer == "综合回答"
        assert len(result.sources) == 2
        # 验证 content 被截断到 200 字符
        assert len(result.sources[0]["content"]) <= 200

    @pytest.mark.asyncio
    async def test_query_no_results(self):
        """测试无检索结果"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=ChatResult(
            content="无法找到相关信息",
            usage={},
            finish_reason="stop",
        ))

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        result = await agent.query("问题")

        assert len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_stream_query(self):
        """测试流式查询"""
        mock_search_result = HybridSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="内容",
            text_score=None,
            vector_score=0.9,
            rerank_score=None,
            final_score=0.9,
            metadata={},
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_search_result]

        async def mock_stream():
            yield "第"
            yield "一"
            yield "部"
            yield "分"

        mock_llm = MagicMock()
        mock_llm.stream_chat = MagicMock(return_value=mock_stream())

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)

        chunks = []
        async for chunk in agent.stream_query("问题"):
            chunks.append(chunk)

        assert chunks == ["第", "一", "部", "分"]

    def test_build_context(self):
        """测试构建上下文"""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)

        search_results = [
            HybridSearchResult(
                doc_id="doc_001",
                chunk_id="chunk_001",
                content="第一段内容",
                text_score=None,
                vector_score=0.9,
                rerank_score=None,
                final_score=0.9,
                metadata={},
            ),
            HybridSearchResult(
                doc_id="doc_002",
                chunk_id="chunk_002",
                content="第二段内容",
                text_score=None,
                vector_score=0.8,
                rerank_score=None,
                final_score=0.8,
                metadata={},
            ),
        ]

        context = agent._build_context(search_results)

        assert "[文档 1]" in context
        assert "[文档 2]" in context
        assert "第一段内容" in context
        assert "第二段内容" in context

    def test_build_context_empty(self):
        """测试空结果构建上下文"""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        context = agent._build_context([])

        assert "未找到相关文档" in context

    @pytest.mark.asyncio
    async def test_query_uses_prompt_manager(self):
        """测试使用 PromptManager 构建消息"""
        mock_search_result = HybridSearchResult(
            doc_id="doc_001",
            chunk_id="chunk_001",
            content="内容",
            text_score=None,
            vector_score=0.9,
            rerank_score=None,
            final_score=0.9,
            metadata={},
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_search_result]

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=ChatResult(
            content="回答",
            usage={},
            finish_reason="stop",
        ))

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
        await agent.query("测试问题")

        # 验证 chat 被调用，且消息格式正确
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]  # 第一个位置参数

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_run_custom_top_k(self):
        """测试自定义 top_k"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=ChatResult(
            content="回答",
            usage={},
            finish_reason="stop",
        ))

        agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm, top_k=5)
        await agent.run({"question": "问题", "top_k": 10})

        call_args = mock_retriever.search.call_args
        assert call_args.kwargs["top_k"] == 10
