"""LLMClient 单元测试"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.llm.client import LLMClient, AsyncLLMClient, ChatMessage, ChatResult


class TestChatMessage:
    """ChatMessage 数据类测试"""

    def test_creation(self):
        """测试创建"""
        msg = ChatMessage(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"


class TestChatResult:
    """ChatResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = ChatResult(
            content="AI 回复",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )
        assert result.content == "AI 回复"
        assert result.usage["total_tokens"] == 30
        assert result.finish_reason == "stop"


class TestLLMClient:
    """LLMClient 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch("core.llm.client.OpenAI"):
            client = LLMClient()
            assert client.base_url is not None
            assert client.model is not None

    def test_init_custom(self):
        """测试自定义初始化"""
        with patch("core.llm.client.OpenAI"):
            client = LLMClient(
                base_url="http://custom:8000",
                model="custom-model",
                api_key="test-key",
                timeout=60.0,
            )
            assert client.base_url == "http://custom:8000"
            assert client.model == "custom-model"

    @patch("core.llm.client.OpenAI")
    def test_chat_basic(self, mock_openai_class):
        """测试基本对话"""
        # 设置 Mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI 回复"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        result = client.chat([{"role": "user", "content": "你好"}])

        assert isinstance(result, ChatResult)
        assert result.content == "AI 回复"
        assert result.usage["total_tokens"] == 30

    @patch("core.llm.client.OpenAI")
    def test_chat_with_chat_message(self, mock_openai_class):
        """测试使用 ChatMessage 对象"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "回复"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        messages = [ChatMessage(role="user", content="你好")]
        result = client.chat(messages)

        assert result.content == "回复"

    @patch("core.llm.client.OpenAI")
    def test_generate(self, mock_openai_class):
        """测试文本生成"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "生成的文本"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        result = client.generate("写一首诗")

        assert result == "生成的文本"

    @patch("core.llm.client.OpenAI")
    def test_generate_with_system_prompt(self, mock_openai_class):
        """测试带系统提示词的生成"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "结果"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        result = client.generate(
            prompt="写诗",
            system_prompt="你是一个诗人",
        )

        # 验证消息格式
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("core.llm.client.OpenAI")
    def test_stream_chat(self, mock_openai_class):
        """测试流式对话"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # 创建流式响应
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "你"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "好"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None  # 结束

        mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

        client = LLMClient()
        chunks = list(client.stream_chat([{"role": "user", "content": "问好"}]))

        assert chunks == ["你", "好"]

    @patch("core.llm.client.OpenAI")
    def test_chat_no_usage(self, mock_openai_class):
        """测试无 usage 信息的响应"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "回复"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        result = client.chat([{"role": "user", "content": "测试"}])

        assert result.usage["total_tokens"] == 0

    @patch("core.llm.client.OpenAI")
    def test_chat_empty_content(self, mock_openai_class):
        """测试空内容响应"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        result = client.chat([{"role": "user", "content": "测试"}])

        assert result.content == ""


class TestAsyncLLMClient:
    """AsyncLLMClient 测试"""

    def test_init(self):
        """测试初始化"""
        with patch("core.llm.client.AsyncOpenAI"):
            client = AsyncLLMClient(
                base_url="http://localhost:8000",
                model="test-model",
            )
            assert client.base_url == "http://localhost:8000"
            assert client.model == "test-model"

    @pytest.mark.asyncio
    @patch("core.llm.client.AsyncOpenAI")
    async def test_async_chat(self, mock_openai_class):
        """测试异步对话"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "异步回复"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = AsyncLLMClient()
        result = await client.chat([{"role": "user", "content": "你好"}])

        assert isinstance(result, ChatResult)
        assert result.content == "异步回复"

    @pytest.mark.asyncio
    @patch("core.llm.client.AsyncOpenAI")
    async def test_async_generate(self, mock_openai_class):
        """测试异步生成"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "生成结果"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = AsyncLLMClient()
        result = await client.generate("写诗")

        assert result == "生成结果"

    @pytest.mark.asyncio
    @patch("core.llm.client.AsyncOpenAI")
    async def test_async_stream_chat(self, mock_openai_class):
        """测试异步流式对话"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # 创建异步迭代器
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "异"
            yield chunk1

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = "步"
            yield chunk2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        client = AsyncLLMClient()
        chunks = []
        async for chunk in client.stream_chat([{"role": "user", "content": "测试"}]):
            chunks.append(chunk)

        assert chunks == ["异", "步"]

    @pytest.mark.asyncio
    @patch("core.llm.client.AsyncOpenAI")
    async def test_async_chat_with_chat_message(self, mock_openai_class):
        """测试使用 ChatMessage 对象"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "回复"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = AsyncLLMClient()
        messages = [ChatMessage(role="user", content="你好")]
        result = await client.chat(messages)

        assert result.content == "回复"
