"""Qwen3-VL 视觉理解客户端

vLLM 部署命令:
    vllm serve Qwen/Qwen3-VL-8B --host 0.0.0.0 --port 8000

API 参考: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
"""
import base64
import httpx
from pathlib import Path
from dataclasses import dataclass

from configs.settings import settings


@dataclass
class VisionResult:
    """视觉理解结果"""
    content: str
    usage: dict


class VisionParser:
    """Qwen3-VL 视觉解析器

    用于图表理解、OCR 识别等视觉任务

    API 端点: POST /v1/chat/completions
    请求格式 (OpenAI Vision API 兼容):
    {
        "model": "Qwen/Qwen3-VL-8B",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "描述这张图片"}
            ]
        }],
        "max_tokens": 4096
    }
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def parse_image(self, image_path: str, prompt: str | None = None) -> VisionResult:
        """解析图片文件"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = path.suffix.lower().lstrip(".")
        media_type = self._get_media_type(suffix)

        return self.parse_base64(image_data, media_type, prompt)

    def parse_base64(
        self,
        image_base64: str,
        media_type: str = "image/png",
        prompt: str | None = None,
    ) -> VisionResult:
        """解析 base64 编码的图片"""
        if prompt is None:
            prompt = "请详细描述这张图片的内容，如果是表格请提取表格数据，如果是图表请描述图表信息。"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return self._chat(messages)

    def parse_url(self, image_url: str, prompt: str | None = None) -> VisionResult:
        """解析图片 URL"""
        if prompt is None:
            prompt = "请详细描述这张图片的内容，如果是表格请提取表格数据，如果是图表请描述图表信息。"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return self._chat(messages)

    def ocr(self, image_path: str) -> str:
        """OCR 文字识别"""
        result = self.parse_image(
            image_path,
            prompt="请识别图片中的所有文字，保持原有格式输出。",
        )
        return result.content

    def extract_table(self, image_path: str) -> str:
        """提取表格数据"""
        result = self.parse_image(
            image_path,
            prompt="请提取图片中的表格数据，以 Markdown 表格格式输出。",
        )
        return result.content

    def describe_chart(self, image_path: str) -> str:
        """描述图表内容"""
        result = self.parse_image(
            image_path,
            prompt="请详细分析这个图表，包括图表类型、数据趋势、关键数值等信息。",
        )
        return result.content

    def _chat(self, messages: list) -> VisionResult:
        """发送聊天请求"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return VisionResult(content=content, usage=usage)

    def _get_media_type(self, suffix: str) -> str:
        """获取媒体类型"""
        media_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        return media_types.get(suffix, "image/png")

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
