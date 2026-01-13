"""端到端流程集成测试

测试完整的知识库流程：
1. 文档解析与分块
2. 向量化入库
3. 检索查询
4. RAG 问答

运行方式：
    pytest tests/integration/test_e2e_pipeline.py -v -m integration
"""
import pytest

from tests.integration.conftest import (
    requires_milvus,
    requires_embedding,
    requires_llm,
    requires_rerank,
    requires_all_services,
)


@pytest.mark.integration
class TestDocumentIngestion:
    """文档入库流程测试"""

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_text_chunking_and_embedding(
        self,
        async_embedding_client,
        test_collection,
        sample_texts,
    ):
        """测试文本分块与向量化"""
        from core.parsers.chunker import TextChunker

        # 1. 文本分块
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)
        all_chunks = []
        for text in sample_texts[:3]:
            chunks = chunker.chunk_text(text)  # 正确的方法名
            all_chunks.extend(chunks)

        assert len(all_chunks) >= 3

        # 2. 批量向量化
        contents = [chunk.content for chunk in all_chunks]
        vectors = await async_embedding_client.embed_documents(contents)

        assert len(vectors) == len(contents)
        assert len(vectors[0]) == 1024  # qwen3-0.6b-embedding 向量维度

        # 3. 写入 Milvus
        data = [
            {
                "doc_id": f"doc_{i:03d}",
                "chunk_id": f"chunk_{i:03d}",
                "content": chunk.content,
                "vector": vectors[i],
            }
            for i, chunk in enumerate(all_chunks)
        ]

        result = test_collection.insert(data)
        assert "ids" in result or "insert_count" in result
        # 新 API 返回 {"ids": [...], "insert_count": N}

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_document_with_metadata(
        self,
        async_embedding_client,
        test_collection,
    ):
        """测试带元数据的文档入库"""
        documents = [
            {
                "doc_id": "meta_doc_001",
                "chunk_id": "meta_chunk_001",
                "content": "这是一份关于产品规格的技术文档。",
                "metadata": {
                    "filename": "product_spec.pdf",
                    "page": 1,
                    "category": "技术文档",
                    "author": "张三",
                },
            },
            {
                "doc_id": "meta_doc_001",
                "chunk_id": "meta_chunk_002",
                "content": "产品支持多种数据格式的导入和导出。",
                "metadata": {
                    "filename": "product_spec.pdf",
                    "page": 2,
                    "category": "技术文档",
                    "author": "张三",
                },
            },
        ]

        # 向量化
        contents = [doc["content"] for doc in documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(documents):
            doc["vector"] = vectors[i]

        # 入库
        result = test_collection.insert(documents)
        assert "ids" in result or "insert_count" in result


@pytest.mark.integration
class TestVectorSearch:
    """向量检索测试"""

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_basic_vector_search(
        self,
        async_embedding_client,
        test_collection,
        sample_documents,
    ):
        """测试基础向量检索"""
        # 准备数据
        contents = [doc["content"] for doc in sample_documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(sample_documents):
            doc["vector"] = vectors[i]

        test_collection.insert(sample_documents)

        # 等待索引生效
        test_collection.flush()

        # 执行搜索
        query = "什么是深度学习和神经网络？"
        query_vector = await async_embedding_client.embed(query)

        results = test_collection.search(
            vector=query_vector,
            top_k=3,
            output_fields=["content", "doc_id", "chunk_id"],
        )

        assert len(results) > 0
        assert "content" in results[0]
        # 验证相关性 - 应该返回深度学习相关内容
        top_content = results[0]["content"]
        assert any(kw in top_content for kw in ["深度学习", "神经网络", "机器学习"])

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        async_embedding_client,
        test_collection,
    ):
        """测试带过滤条件的检索"""
        # 准备不同来源的数据
        documents = [
            {"doc_id": "doc_a", "chunk_id": "c1", "content": "AI 技术文档内容", "source": "tech"},
            {"doc_id": "doc_b", "chunk_id": "c2", "content": "AI 产品说明内容", "source": "product"},
            {"doc_id": "doc_c", "chunk_id": "c3", "content": "AI 研究论文内容", "source": "research"},
        ]

        contents = [doc["content"] for doc in documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(documents):
            doc["vector"] = vectors[i]

        test_collection.insert(documents)
        test_collection.flush()

        # 带过滤条件搜索
        query_vector = await async_embedding_client.embed("AI 相关内容")

        results = test_collection.search(
            vector=query_vector,
            top_k=10,
            filter_expr='source == "tech"',
            output_fields=["content", "source"],
        )

        # 所有结果应该都是 tech 来源
        for result in results:
            assert result.get("source") == "tech"


@pytest.mark.integration
class TestHybridRetrieval:
    """混合检索测试"""

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_hybrid_search(
        self,
        async_embedding_client,
        embedding_client,
        test_collection,
        sample_documents,
    ):
        """测试双路召回"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever

        # 准备数据（使用 async client 进行批量向量化）
        contents = [doc["content"] for doc in sample_documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(sample_documents):
            doc["vector"] = vectors[i]

        test_collection.insert(sample_documents)
        test_collection.flush()

        # 创建向量检索器（使用 sync client）
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )

        # 创建混合检索器
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
        )

        # 执行混合检索（sync 方法）
        results = retriever.search(
            query="深度学习如何使用神经网络",
            top_k=5,
            use_rerank=False,  # 不使用 rerank
        )

        assert len(results) > 0
        # 验证结果包含必要字段
        assert results[0].content is not None
        assert results[0].final_score is not None

    @requires_milvus
    @requires_embedding
    @requires_rerank
    @pytest.mark.asyncio
    async def test_hybrid_search_with_rerank(
        self,
        async_embedding_client,
        embedding_client,
        rerank_client,
        test_collection,
        sample_documents,
    ):
        """测试带 Rerank 的混合检索"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever

        # 准备数据
        contents = [doc["content"] for doc in sample_documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(sample_documents):
            doc["vector"] = vectors[i]

        test_collection.insert(sample_documents)
        test_collection.flush()

        # 创建向量检索器
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )

        # 创建带 Rerank 的混合检索器
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            rerank_client=rerank_client,
        )

        results = retriever.search(
            query="什么是 RAG 检索增强生成",
            top_k=5,
            use_rerank=True,
        )

        assert len(results) > 0
        # 验证 Rerank 分数存在
        assert results[0].rerank_score is not None


@pytest.mark.integration
@pytest.mark.e2e
class TestRAGPipeline:
    """完整 RAG 流程测试"""

    @requires_all_services
    @pytest.mark.asyncio
    async def test_full_rag_query(
        self,
        async_embedding_client,
        embedding_client,
        async_llm_client,
        test_collection,
        sample_documents,
    ):
        """测试完整 RAG 问答流程"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever
        from core.agent.rag import RAGAgent

        # 1. 数据入库
        contents = [doc["content"] for doc in sample_documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(sample_documents):
            doc["vector"] = vectors[i]

        test_collection.insert(sample_documents)
        test_collection.flush()

        # 2. 创建 RAG Agent
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )
        retriever = HybridRetriever(vector_retriever=vector_retriever)

        agent = RAGAgent(
            retriever=retriever,
            llm_client=async_llm_client,
            top_k=3,
        )

        # 3. 执行问答
        response = await agent.query("什么是机器学习？它和深度学习有什么关系？")

        # 验证回答
        assert response.answer is not None
        assert len(response.answer) > 50  # 回答应该有一定长度
        assert len(response.sources) > 0  # 应该有来源引用

    @requires_all_services
    @pytest.mark.asyncio
    async def test_rag_stream_query(
        self,
        async_embedding_client,
        embedding_client,
        async_llm_client,
        test_collection,
        sample_documents,
    ):
        """测试 RAG 流式问答"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever
        from core.agent.rag import RAGAgent

        # 数据入库
        contents = [doc["content"] for doc in sample_documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(sample_documents):
            doc["vector"] = vectors[i]

        test_collection.insert(sample_documents)
        test_collection.flush()

        # 创建 RAG Agent
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )
        retriever = HybridRetriever(vector_retriever=vector_retriever)

        agent = RAGAgent(
            retriever=retriever,
            llm_client=async_llm_client,
        )

        # 流式问答
        chunks = []
        async for chunk in agent.stream_query("简要介绍人工智能"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_answer = "".join(chunks)
        assert len(full_answer) > 20

    @requires_all_services
    @pytest.mark.asyncio
    async def test_rag_with_no_relevant_docs(
        self,
        async_embedding_client,
        embedding_client,
        async_llm_client,
        test_collection,
    ):
        """测试无相关文档时的 RAG 行为"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever
        from core.agent.rag import RAGAgent

        # 只入库无关数据
        documents = [
            {
                "doc_id": "irrelevant_001",
                "chunk_id": "c1",
                "content": "今天天气很好，适合户外运动。",
            },
        ]

        vectors = await async_embedding_client.embed_documents([d["content"] for d in documents])
        for i, doc in enumerate(documents):
            doc["vector"] = vectors[i]

        test_collection.insert(documents)
        test_collection.flush()

        # 创建 RAG Agent
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )
        retriever = HybridRetriever(vector_retriever=vector_retriever)

        agent = RAGAgent(
            retriever=retriever,
            llm_client=async_llm_client,
        )

        # 问一个完全无关的问题
        response = await agent.query("量子计算机的工作原理是什么？")

        # 应该仍然能够回答（可能基于 LLM 自身知识或说明无法找到相关信息）
        assert response.answer is not None


@pytest.mark.integration
class TestMultiDocumentRAG:
    """多文档 RAG 测试"""

    @requires_all_services
    @pytest.mark.asyncio
    async def test_cross_document_query(
        self,
        async_embedding_client,
        embedding_client,
        async_llm_client,
        test_collection,
    ):
        """测试跨文档综合问答"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever
        from core.agent.rag import RAGAgent

        # 准备来自不同文档的内容
        documents = [
            {
                "doc_id": "ml_guide",
                "chunk_id": "ml_1",
                "content": "机器学习的三大类型包括：监督学习、无监督学习和强化学习。监督学习需要标注数据。",
                "metadata": {"filename": "ml_guide.pdf"},
            },
            {
                "doc_id": "dl_intro",
                "chunk_id": "dl_1",
                "content": "深度学习是机器学习的子领域，使用深层神经网络。CNN 用于图像处理，RNN 用于序列数据。",
                "metadata": {"filename": "dl_intro.pdf"},
            },
            {
                "doc_id": "ai_overview",
                "chunk_id": "ai_1",
                "content": "人工智能涵盖机器学习、知识表示、自然语言处理等多个领域。目前机器学习是最活跃的方向。",
                "metadata": {"filename": "ai_overview.pdf"},
            },
        ]

        contents = [doc["content"] for doc in documents]
        vectors = await async_embedding_client.embed_documents(contents)

        for i, doc in enumerate(documents):
            doc["vector"] = vectors[i]

        test_collection.insert(documents)
        test_collection.flush()

        # 创建 RAG Agent
        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=test_collection,
        )
        retriever = HybridRetriever(vector_retriever=vector_retriever)

        agent = RAGAgent(
            retriever=retriever,
            llm_client=async_llm_client,
            top_k=3,
        )

        # 问一个需要综合多个文档才能回答的问题
        response = await agent.query("机器学习和深度学习的关系是什么？各自有哪些类型？")

        assert response.answer is not None
        # 应该引用了多个来源
        assert len(response.sources) >= 2

        # 验证来源来自不同文档
        source_docs = {s.get("doc_id") for s in response.sources}
        assert len(source_docs) >= 2
