"""Qwen3-Reranker 重排序客户端

vLLM 部署命令:
    vllm serve Qwen/Qwen3-Reranker-8B --task score --host 0.0.0.0 --port 8002

API 参考:
- https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- Cohere Rerank API 兼容
"""
import httpx
from dataclasses import dataclass

from configs.settings import settings


@dataclass
class RerankItem:
    """重排序结果项"""
    index: int
    score: float
    text: str


@dataclass
class RerankResult:
    """重排序结果"""
    results: list[RerankItem]
    usage: dict


class RerankClient:
    """Qwen3-Reranker 重排序客户端

    使用 vLLM 的 Cohere 兼容接口

    API 端点: POST /v1/rerank
    请求格式:
    {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": "What is the capital of France?",
        "documents": ["doc1", "doc2", "doc3"],
        "top_n": 5
    }

    响应格式:
    {
        "id": "rerank-xxx",
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.82}
        ],
        "meta": {"api_version": {...}, "billed_units": {...}}
    }
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        top_k: int | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or settings.rerank_base_url
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.rerank_api_key
        self.top_k = top_k or settings.rerank_top_k
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> RerankResult:
        """对文档进行重排序

        vLLM /v1/rerank 请求格式 (Cohere 兼容)
        """
        top_k = top_k or self.top_k
        if top_k <= 0:
            top_k = len(documents)

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        response = self.client.post("/v1/rerank", json=payload)
        response.raise_for_status()

        data = response.json()

        results = [
            RerankItem(
                index=item["index"],
                score=item.get("relevance_score", item.get("score", 0.0)),
                text=documents[item["index"]],
            )
            for item in data.get("results", [])
        ]

        return RerankResult(
            results=results,
            usage=data.get("meta", {}).get("billed_units", {}),
        )

    def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict],
        text_key: str = "content",
        top_k: int | None = None,
    ) -> list[dict]:
        """对带元数据的文档进行重排序"""
        texts = [doc[text_key] for doc in documents]
        result = self.rerank(query, texts, top_k)

        reranked_docs = []
        for item in result.results:
            doc = documents[item.index].copy()
            doc["rerank_score"] = item.score
            reranked_docs.append(doc)

        return reranked_docs

    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncRerankClient:
    """异步 Rerank 客户端"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        top_k: int | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or settings.rerank_base_url
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.rerank_api_key
        self.top_k = top_k or settings.rerank_top_k
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> RerankResult:
        """对文档进行重排序"""
        top_k = top_k or self.top_k
        if top_k <= 0:
            top_k = len(documents)

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        response = await self.client.post("/v1/rerank", json=payload)
        response.raise_for_status()

        data = response.json()

        results = [
            RerankItem(
                index=item["index"],
                score=item.get("relevance_score", item.get("score", 0.0)),
                text=documents[item["index"]],
            )
            for item in data.get("results", [])
        ]

        return RerankResult(
            results=results,
            usage=data.get("meta", {}).get("billed_units", {}),
        )

    async def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict],
        text_key: str = "content",
        top_k: int | None = None,
    ) -> list[dict]:
        """对带元数据的文档进行重排序"""
        texts = [doc[text_key] for doc in documents]
        result = await self.rerank(query, texts, top_k)

        reranked_docs = []
        for item in result.results:
            doc = documents[item.index].copy()
            doc["rerank_score"] = item.score
            reranked_docs.append(doc)

        return reranked_docs

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
