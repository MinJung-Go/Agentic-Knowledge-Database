"""vLLM + Qwen3 LLM 客户端

vLLM 部署命令:
    vllm serve Qwen/Qwen3-VL-8B --host 0.0.0.0 --port 8000

使用 OpenAI SDK 调用 vLLM 的 OpenAI 兼容接口
"""
from dataclasses import dataclass
from typing import Generator, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

from configs.settings import settings


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str


@dataclass
class ChatResult:
    """聊天结果"""
    content: str
    usage: dict
    finish_reason: str


class LLMClient:
    """LLM 客户端

    使用 OpenAI SDK 调用 vLLM
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict] | list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> ChatResult:
        """发送聊天请求"""
        if messages and isinstance(messages[0], ChatMessage):
            messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage

        return ChatResult(
            content=choice.message.content or "",
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """简单文本生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self.chat(messages, temperature, max_tokens, **kwargs)
        return result.content

    def stream_chat(
        self,
        messages: list[dict] | list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式聊天"""
        if messages and isinstance(messages[0], ChatMessage):
            messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AsyncLLMClient:
    """异步 LLM 客户端

    使用 OpenAI AsyncClient 调用 vLLM
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[dict] | list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> ChatResult:
        """发送聊天请求"""
        if messages and isinstance(messages[0], ChatMessage):
            messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage

        return ChatResult(
            content=choice.message.content or "",
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """简单文本生成"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = await self.chat(messages, temperature, max_tokens, **kwargs)
        return result.content

    async def stream_chat(
        self,
        messages: list[dict] | list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """流式聊天"""
        if messages and isinstance(messages[0], ChatMessage):
            messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
