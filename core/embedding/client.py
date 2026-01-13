"""Qwen3-Embedding 向量化客户端

支持两种部署方式:
1. vLLM OpenAI 兼容接口: /v1/embeddings
2. SageMaker/DJL-LMI 接口: /invocations

vLLM 部署命令:
    vllm serve Qwen/Qwen3-Embedding-8B --task embed --host 0.0.0.0 --port 8001
"""
import httpx
from dataclasses import dataclass
from enum import Enum

from configs.settings import settings


class EndpointType(str, Enum):
    """端点类型"""
    OPENAI = "openai"           # vLLM OpenAI 兼容: /v1/embeddings
    INVOCATIONS = "invocations" # SageMaker/DJL-LMI: /invocations


@dataclass
class EmbeddingResult:
    """向量化结果"""
    embeddings: list[list[float]]
    usage: dict


class EmbeddingClient:
    """Qwen3-Embedding 向量化客户端

    支持两种端点类型:
    - openai: vLLM OpenAI 兼容接口 (/v1/embeddings)
    - invocations: SageMaker/DJL-LMI 接口 (/invocations)

    使用示例:
        # OpenAI 兼容模式
        client = EmbeddingClient(base_url="http://localhost:8001")

        # SageMaker/invocations 模式
        client = EmbeddingClient(
            base_url="http://192.168.25.10:30264",
            endpoint_type="invocations"
        )
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        endpoint_type: str | EndpointType | None = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.embedding_api_key
        self.timeout = timeout

        # 自动检测端点类型
        if endpoint_type:
            self.endpoint_type = EndpointType(endpoint_type)
        elif "/invocations" in self.base_url:
            self.endpoint_type = EndpointType.INVOCATIONS
            # 移除 URL 中的 /invocations，后面会自动添加
            self.base_url = self.base_url.replace("/invocations", "")
        else:
            self.endpoint_type = EndpointType.OPENAI

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def embed(self, text: str) -> list[float]:
        """单条文本向量化"""
        result = self.embed_batch([text])
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """批量文本向量化"""
        if self.endpoint_type == EndpointType.INVOCATIONS:
            return self._embed_invocations(texts)
        else:
            return self._embed_openai(texts)

    def _embed_openai(self, texts: list[str]) -> EmbeddingResult:
        """OpenAI 兼容格式请求 (/v1/embeddings)"""
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        response = self.client.post("/v1/embeddings", json=payload)
        response.raise_for_status()

        data = response.json()

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in sorted_data]
        usage = data.get("usage", {})

        return EmbeddingResult(embeddings=embeddings, usage=usage)

    def _embed_invocations(self, texts: list[str]) -> EmbeddingResult:
        """SageMaker/DJL-LMI 格式请求 (/invocations)

        支持多种响应格式:
        1. OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}]}
        2. SageMaker 格式: {"embeddings": [[...], [...]]}
        3. 简单列表格式: [[...], [...]]
        """
        # 尝试 OpenAI 兼容格式
        payload = {
            "model": self.model,
            "input": texts,
        }

        response = self.client.post("/invocations", json=payload)

        # 如果 OpenAI 格式失败，尝试 SageMaker 原生格式
        if response.status_code >= 400:
            payload = {"inputs": texts}
            response = self.client.post("/invocations", json=payload)

        response.raise_for_status()
        data = response.json()

        # 解析不同格式的响应
        embeddings = self._parse_invocations_response(data)
        usage = data.get("usage", {})

        return EmbeddingResult(embeddings=embeddings, usage=usage)

    def _parse_invocations_response(self, data: dict | list) -> list[list[float]]:
        """解析 /invocations 响应的多种格式"""
        # 格式 1: OpenAI 兼容 {"data": [{"embedding": [...]}]}
        if isinstance(data, dict) and "data" in data:
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]

        # 格式 2: {"embeddings": [[...]]}
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]

        # 格式 3: {"vectors": [[...]]}
        if isinstance(data, dict) and "vectors" in data:
            return data["vectors"]

        # 格式 4: 直接返回列表 [[...], [...]]
        if isinstance(data, list):
            # 检查是否是嵌套列表（多个向量）
            if data and isinstance(data[0], list):
                return data
            # 单个向量
            return [data]

        # 格式 5: {"output": [[...]]}
        if isinstance(data, dict) and "output" in data:
            output = data["output"]
            if isinstance(output, list):
                return output if isinstance(output[0], list) else [output]

        raise ValueError(f"无法解析响应格式: {type(data)}, keys={data.keys() if isinstance(data, dict) else 'N/A'}")

    def embed_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量文档向量化（支持分批处理）"""
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = self.embed_batch(batch)
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if self.endpoint_type == EndpointType.INVOCATIONS:
                response = self.client.get("/ping")
            else:
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


class AsyncEmbeddingClient:
    """异步 Embedding 客户端"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        endpoint_type: str | EndpointType | None = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.embedding_api_key
        self.timeout = timeout

        # 自动检测端点类型
        if endpoint_type:
            self.endpoint_type = EndpointType(endpoint_type)
        elif "/invocations" in self.base_url:
            self.endpoint_type = EndpointType.INVOCATIONS
            self.base_url = self.base_url.replace("/invocations", "")
        else:
            self.endpoint_type = EndpointType.OPENAI

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def embed(self, text: str) -> list[float]:
        """单条文本向量化"""
        result = await self.embed_batch([text])
        return result.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """批量文本向量化"""
        if self.endpoint_type == EndpointType.INVOCATIONS:
            return await self._embed_invocations(texts)
        else:
            return await self._embed_openai(texts)

    async def _embed_openai(self, texts: list[str]) -> EmbeddingResult:
        """OpenAI 兼容格式请求"""
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        response = await self.client.post("/v1/embeddings", json=payload)
        response.raise_for_status()

        data = response.json()

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in sorted_data]
        usage = data.get("usage", {})

        return EmbeddingResult(embeddings=embeddings, usage=usage)

    async def _embed_invocations(self, texts: list[str]) -> EmbeddingResult:
        """SageMaker/DJL-LMI 格式请求"""
        payload = {
            "model": self.model,
            "input": texts,
        }

        response = await self.client.post("/invocations", json=payload)

        if response.status_code >= 400:
            payload = {"inputs": texts}
            response = await self.client.post("/invocations", json=payload)

        response.raise_for_status()
        data = response.json()

        embeddings = self._parse_invocations_response(data)
        usage = data.get("usage", {})

        return EmbeddingResult(embeddings=embeddings, usage=usage)

    def _parse_invocations_response(self, data: dict | list) -> list[list[float]]:
        """解析 /invocations 响应"""
        if isinstance(data, dict) and "data" in data:
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]

        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]

        if isinstance(data, dict) and "vectors" in data:
            return data["vectors"]

        if isinstance(data, list):
            if data and isinstance(data[0], list):
                return data
            return [data]

        if isinstance(data, dict) and "output" in data:
            output = data["output"]
            if isinstance(output, list):
                return output if isinstance(output[0], list) else [output]

        raise ValueError(f"无法解析响应格式: {type(data)}")

    async def embed_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量文档向量化"""
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = await self.embed_batch(batch)
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if self.endpoint_type == EndpointType.INVOCATIONS:
                response = await self.client.get("/ping")
            else:
                response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
