"""性能压测

测试各组件的性能指标：
- 吞吐量 (Throughput)
- 延迟 (Latency): 平均、P50、P95、P99
- 并发能力 (Concurrency)

运行方式：
    pytest tests/integration/test_performance.py -v -m "integration and slow"
"""
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import pytest

from tests.integration.conftest import (
    requires_milvus,
    requires_embedding,
    requires_llm,
    requires_rerank,
    requires_all_services,
)


@dataclass
class PerformanceMetrics:
    """性能指标"""

    total_requests: int
    total_time_s: float
    latencies_ms: list[float]

    @property
    def throughput(self) -> float:
        """吞吐量 (requests/s)"""
        return self.total_requests / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        """平均延迟"""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p50_latency_ms(self) -> float:
        """P50 延迟"""
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        """P95 延迟"""
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        """P99 延迟"""
        return self._percentile(99)

    @property
    def min_latency_ms(self) -> float:
        """最小延迟"""
        return min(self.latencies_ms) if self.latencies_ms else 0

    @property
    def max_latency_ms(self) -> float:
        """最大延迟"""
        return max(self.latencies_ms) if self.latencies_ms else 0

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def report(self, name: str) -> str:
        """生成性能报告"""
        return f"""
========== {name} 性能报告 ==========
请求数量: {self.total_requests}
总耗时: {self.total_time_s:.2f}s
吞吐量: {self.throughput:.2f} req/s
延迟统计:
  - 平均: {self.avg_latency_ms:.2f}ms
  - P50: {self.p50_latency_ms:.2f}ms
  - P95: {self.p95_latency_ms:.2f}ms
  - P99: {self.p99_latency_ms:.2f}ms
  - 最小: {self.min_latency_ms:.2f}ms
  - 最大: {self.max_latency_ms:.2f}ms
==========================================
"""


@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingPerformance:
    """Embedding 服务性能测试"""

    @requires_embedding
    @pytest.mark.asyncio
    async def test_single_embedding_latency(self, async_embedding_client):
        """单条 Embedding 延迟测试"""
        text = "这是一段用于测试向量化性能的文本内容。" * 5
        iterations = 20

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            await async_embedding_client.embed(text)
            latencies.append((time.perf_counter() - start) * 1000)

        metrics = PerformanceMetrics(
            total_requests=iterations,
            total_time_s=sum(latencies) / 1000,
            latencies_ms=latencies,
        )

        print(metrics.report("单条 Embedding"))

        # 性能断言
        assert metrics.avg_latency_ms < 500, f"平均延迟过高: {metrics.avg_latency_ms}ms"
        assert metrics.p99_latency_ms < 1000, f"P99 延迟过高: {metrics.p99_latency_ms}ms"

    @requires_embedding
    @pytest.mark.asyncio
    async def test_batch_embedding_throughput(self, async_embedding_client):
        """批量 Embedding 吞吐量测试"""
        # 准备测试数据
        texts = [f"测试文本 {i}，包含一些用于向量化的内容。" * 3 for i in range(100)]
        batch_size = 32

        start = time.perf_counter()
        await async_embedding_client.embed_documents(texts, batch_size=batch_size)
        elapsed = time.perf_counter() - start

        throughput = len(texts) / elapsed

        print(f"""
========== 批量 Embedding 性能报告 ==========
文本数量: {len(texts)}
批次大小: {batch_size}
总耗时: {elapsed:.2f}s
吞吐量: {throughput:.2f} 条/s
==============================================
""")

        # 性能断言
        assert throughput > 5, f"吞吐量过低: {throughput:.2f} 条/s"

    @requires_embedding
    @pytest.mark.asyncio
    async def test_concurrent_embedding(self, async_embedding_client):
        """并发 Embedding 测试"""
        text = "并发测试文本内容。" * 5
        concurrency = 10
        requests_per_task = 5

        async def single_task():
            latencies = []
            for _ in range(requests_per_task):
                start = time.perf_counter()
                await async_embedding_client.embed(text)
                latencies.append((time.perf_counter() - start) * 1000)
            return latencies

        start = time.perf_counter()
        results = await asyncio.gather(*[single_task() for _ in range(concurrency)])
        total_time = time.perf_counter() - start

        all_latencies = [lat for task_lats in results for lat in task_lats]
        total_requests = concurrency * requests_per_task

        metrics = PerformanceMetrics(
            total_requests=total_requests,
            total_time_s=total_time,
            latencies_ms=all_latencies,
        )

        print(metrics.report(f"并发 Embedding (并发数={concurrency})"))

        # 性能断言
        assert metrics.throughput > 3, f"并发 QPS 过低: {metrics.throughput:.2f}"


@pytest.mark.integration
@pytest.mark.slow
class TestMilvusPerformance:
    """Milvus 性能测试"""

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_insert_throughput(
        self,
        async_embedding_client,
        test_collection,
    ):
        """插入吞吐量测试"""
        # 准备数据
        num_docs = 500
        texts = [f"文档 {i} 的内容，用于测试插入性能。" for i in range(num_docs)]
        vectors = await async_embedding_client.embed_documents(texts, batch_size=32)

        documents = [
            {
                "doc_id": f"perf_doc_{i:04d}",
                "chunk_id": f"perf_chunk_{i:04d}",
                "content": texts[i],
                "vector": vectors[i],
            }
            for i in range(num_docs)
        ]

        # 分批插入测试
        batch_size = 100
        latencies = []

        for i in range(0, num_docs, batch_size):
            batch = documents[i : i + batch_size]
            start = time.perf_counter()
            test_collection.insert(batch)
            latencies.append((time.perf_counter() - start) * 1000)

        total_time = sum(latencies) / 1000
        throughput = num_docs / total_time

        print(f"""
========== Milvus 插入性能报告 ==========
文档数量: {num_docs}
批次大小: {batch_size}
总耗时: {total_time:.2f}s
吞吐量: {throughput:.2f} 条/s
批次延迟:
  - 平均: {statistics.mean(latencies):.2f}ms
  - 最大: {max(latencies):.2f}ms
==========================================
""")

        assert throughput > 20, f"插入吞吐量过低: {throughput:.2f} 条/s"

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_search_latency(
        self,
        async_embedding_client,
        shared_collection,
    ):
        """搜索延迟测试"""
        # 确保有数据
        if shared_collection.count() < 100:
            texts = [f"预置数据 {i}" for i in range(200)]
            vectors = await async_embedding_client.embed_documents(texts)
            documents = [
                {
                    "doc_id": f"pre_{i:04d}",
                    "chunk_id": f"c_{i:04d}",
                    "content": texts[i],
                    "vector": vectors[i],
                }
                for i in range(200)
            ]
            shared_collection.insert(documents)
            shared_collection.flush()

        # 搜索测试
        query_vector = await async_embedding_client.embed("测试查询")
        iterations = 50

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            shared_collection.search(
                vector=query_vector,
                top_k=10,
                output_fields=["content"],
            )
            latencies.append((time.perf_counter() - start) * 1000)

        metrics = PerformanceMetrics(
            total_requests=iterations,
            total_time_s=sum(latencies) / 1000,
            latencies_ms=latencies,
        )

        print(metrics.report("Milvus 搜索"))

        # 性能断言
        assert metrics.avg_latency_ms < 100, f"平均搜索延迟过高: {metrics.avg_latency_ms}ms"
        assert metrics.p99_latency_ms < 300, f"P99 搜索延迟过高: {metrics.p99_latency_ms}ms"

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_concurrent_search(
        self,
        async_embedding_client,
        shared_collection,
    ):
        """并发搜索测试"""
        query_vector = await async_embedding_client.embed("并发查询测试")
        concurrency = 20
        requests_per_task = 10

        async def search_task():
            latencies = []
            for _ in range(requests_per_task):
                start = time.perf_counter()
                shared_collection.search(
                    vector=query_vector,
                    top_k=10,
                    output_fields=["content"],
                )
                latencies.append((time.perf_counter() - start) * 1000)
            return latencies

        start = time.perf_counter()
        results = await asyncio.gather(*[search_task() for _ in range(concurrency)])
        total_time = time.perf_counter() - start

        all_latencies = [lat for task_lats in results for lat in task_lats]

        metrics = PerformanceMetrics(
            total_requests=concurrency * requests_per_task,
            total_time_s=total_time,
            latencies_ms=all_latencies,
        )

        print(metrics.report(f"Milvus 并发搜索 (并发数={concurrency})"))

        assert metrics.throughput > 20, f"并发搜索 QPS 过低: {metrics.throughput:.2f}"


@pytest.mark.integration
@pytest.mark.slow
class TestRerankPerformance:
    """Rerank 服务性能测试"""

    @requires_rerank
    @pytest.mark.asyncio
    async def test_rerank_latency(self, async_rerank_client):
        """Rerank 延迟测试"""
        query = "什么是机器学习？"
        documents = [
            "机器学习是人工智能的一个分支。",
            "深度学习使用神经网络。",
            "自然语言处理处理文本数据。",
            "计算机视觉处理图像数据。",
            "强化学习通过奖励信号学习。",
        ] * 2  # 10 个文档

        iterations = 20
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await async_rerank_client.rerank(query, documents, top_k=5)
            latencies.append((time.perf_counter() - start) * 1000)

        metrics = PerformanceMetrics(
            total_requests=iterations,
            total_time_s=sum(latencies) / 1000,
            latencies_ms=latencies,
        )

        print(metrics.report("Rerank"))

        assert metrics.avg_latency_ms < 500, f"Rerank 平均延迟过高: {metrics.avg_latency_ms}ms"


@pytest.mark.integration
@pytest.mark.slow
class TestLLMPerformance:
    """LLM 服务性能测试"""

    @requires_llm
    @pytest.mark.asyncio
    async def test_llm_latency(self, async_llm_client):
        """LLM 推理延迟测试（非流式）"""
        messages = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "用一句话解释什么是人工智能。"},
        ]

        iterations = 5  # LLM 较慢，减少迭代次数
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await async_llm_client.chat(messages, max_tokens=100)
            latencies.append((time.perf_counter() - start) * 1000)

        metrics = PerformanceMetrics(
            total_requests=iterations,
            total_time_s=sum(latencies) / 1000,
            latencies_ms=latencies,
        )

        print(metrics.report("LLM 推理"))

        # LLM 延迟通常较高
        assert metrics.avg_latency_ms < 10000, f"LLM 平均延迟过高: {metrics.avg_latency_ms}ms"

    @requires_llm
    @pytest.mark.asyncio
    async def test_llm_first_token_latency(self, async_llm_client):
        """LLM 首 Token 延迟测试（TTFT）"""
        messages = [
            {"role": "user", "content": "写一段 100 字的自我介绍。"},
        ]

        iterations = 3
        ttft_latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            first_token_received = False

            async for chunk in async_llm_client.stream_chat(messages, max_tokens=150):
                if not first_token_received:
                    ttft_latencies.append((time.perf_counter() - start) * 1000)
                    first_token_received = True
                    break

        avg_ttft = statistics.mean(ttft_latencies)
        print(f"""
========== LLM TTFT 性能报告 ==========
测试次数: {iterations}
平均首 Token 延迟: {avg_ttft:.2f}ms
最小: {min(ttft_latencies):.2f}ms
最大: {max(ttft_latencies):.2f}ms
========================================
""")

        assert avg_ttft < 5000, f"首 Token 延迟过高: {avg_ttft}ms"


@pytest.mark.integration
@pytest.mark.slow
class TestE2EPerformance:
    """端到端性能测试"""

    @requires_all_services
    @pytest.mark.asyncio
    async def test_rag_query_latency(
        self,
        async_embedding_client,
        embedding_client,
        async_llm_client,
        shared_collection,
    ):
        """RAG 查询端到端延迟测试"""
        from core.retrieval.hybrid import HybridRetriever
        from core.retrieval.vector import VectorRetriever
        from core.agent.rag import RAGAgent

        # 确保有数据
        if shared_collection.count() < 50:
            texts = [f"知识库文档 {i}，包含各种技术内容。" for i in range(100)]
            vectors = await async_embedding_client.embed_documents(texts)
            documents = [
                {
                    "doc_id": f"rag_{i:04d}",
                    "chunk_id": f"c_{i:04d}",
                    "content": texts[i],
                    "vector": vectors[i],
                }
                for i in range(100)
            ]
            shared_collection.insert(documents)
            shared_collection.flush()

        vector_retriever = VectorRetriever(
            embedding_client=embedding_client,
            collection_manager=shared_collection,
        )
        retriever = HybridRetriever(vector_retriever=vector_retriever)

        agent = RAGAgent(
            retriever=retriever,
            llm_client=async_llm_client,
            top_k=3,
        )

        # 测试 RAG 查询
        queries = [
            "什么是机器学习？",
            "深度学习有哪些应用？",
            "如何选择合适的算法？",
        ]

        latencies = []
        for query in queries:
            start = time.perf_counter()
            await agent.query(query)
            latencies.append((time.perf_counter() - start) * 1000)

        metrics = PerformanceMetrics(
            total_requests=len(queries),
            total_time_s=sum(latencies) / 1000,
            latencies_ms=latencies,
        )

        print(metrics.report("RAG 端到端"))

        # RAG 包含检索 + LLM，延迟较高
        assert metrics.avg_latency_ms < 15000, f"RAG 平均延迟过高: {metrics.avg_latency_ms}ms"


@pytest.mark.integration
@pytest.mark.slow
class TestScalability:
    """可扩展性测试"""

    @requires_milvus
    @requires_embedding
    @pytest.mark.asyncio
    async def test_search_latency_vs_data_size(
        self,
        async_embedding_client,
        milvus_client,
    ):
        """测试搜索延迟与数据量的关系"""
        from core.milvus.collection import CollectionManager
        import uuid

        collection_name = f"scale_test_{uuid.uuid4().hex[:8]}"
        collection = CollectionManager(
            collection_name=collection_name,
            dimension=1024,  # qwen3-0.6b-embedding outputs 1024 dimensions
            client=milvus_client,
        )
        collection.create()

        try:
            data_sizes = [100, 500, 1000]
            results = []

            for target_size in data_sizes:
                current_size = collection.count()
                if current_size < target_size:
                    # 补充数据
                    num_to_add = target_size - current_size
                    texts = [f"扩展性测试数据 {i}" for i in range(num_to_add)]
                    vectors = await async_embedding_client.embed_documents(texts, batch_size=32)
                    documents = [
                        {
                            "doc_id": f"scale_{i:05d}",
                            "chunk_id": f"c_{i:05d}",
                            "content": texts[i],
                            "vector": vectors[i],
                        }
                        for i in range(num_to_add)
                    ]
                    collection.insert(documents)
                    collection.flush()

                # 测试搜索延迟
                query_vector = await async_embedding_client.embed("扩展性测试查询")
                latencies = []

                for _ in range(20):
                    start = time.perf_counter()
                    collection.search(vector=query_vector, top_k=10)
                    latencies.append((time.perf_counter() - start) * 1000)

                avg_latency = statistics.mean(latencies)
                results.append((target_size, avg_latency))

            print("""
========== 搜索延迟 vs 数据量 ==========
数据量\t\t平均延迟(ms)
""")
            for size, latency in results:
                print(f"{size}\t\t{latency:.2f}")
            print("========================================")

            # 验证延迟增长在合理范围内
            # 从 100 到 1000 数据，延迟增长不应超过 5 倍
            if len(results) >= 2:
                ratio = results[-1][1] / results[0][1]
                assert ratio < 5, f"延迟增长比例过高: {ratio:.2f}x"

        finally:
            try:
                collection.drop()
            except Exception:
                pass
