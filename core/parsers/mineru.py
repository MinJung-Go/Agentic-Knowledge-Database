"""MinerU 文档解析客户端

使用 mineru 库进行文档解析，支持 PDF 和图片。
通过 vlm-http-client backend 连接远程 VLM 服务。

API 参考: https://github.com/opendatalab/MinerU
"""
import os
import tempfile
import asyncio
from pathlib import Path
from typing import BinaryIO
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from configs.settings import settings
from core.parsers.base import BaseParser


@dataclass
class ParseResult:
    """解析结果"""
    content: str
    images: list[str]
    tables: list[str]
    metadata: dict


# 全局线程池，避免在 FastAPI 事件循环中直接运行 mineru
_executor = ThreadPoolExecutor(max_workers=4)


def _do_mineru_parse(
    pdf_bytes: bytes,
    filename: str,
    server_url: str,
    lang: str = "zh",
) -> str:
    """在线程池中执行 mineru 解析（同步函数）"""
    from mineru.cli.client import do_parse

    with tempfile.TemporaryDirectory() as output_dir:
        # 调用 mineru 解析
        do_parse(
            output_dir=output_dir,
            pdf_file_names=[filename],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend="vlm-http-client",
            server_url=server_url,
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
        )

        # 读取生成的 Markdown 文件
        md_content = ""
        output_path = Path(output_dir)

        # 查找生成的 .md 文件
        md_files = list(output_path.rglob("*.md"))
        if md_files:
            # 按文件名排序，合并所有 md 内容
            md_files.sort()
            md_contents = []
            for md_file in md_files:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_contents.append(f.read())
            md_content = "\n\n".join(md_contents)

        return md_content


class MinerUParser(BaseParser):
    """MinerU 文档解析器

    使用 mineru 库的 vlm-http-client backend 进行文档解析。
    支持 PDF 和图片格式。

    使用方式:
        parser = MinerUParser()
        result = parser.parse_bytes(content, "document.pdf")
        print(result.content)
    """

    SUPPORTED_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def __init__(
        self,
        base_url: str | None = None,
        lang: str = "zh",
        timeout: float = 300.0,
    ):
        self.base_url = base_url or settings.mineru_base_url
        self.lang = lang
        self.timeout = timeout

    def parse(self, file_path: str) -> str:
        """解析文档，返回 Markdown 文本"""
        result = self.parse_document(file_path)
        return result.content

    def parse_document(self, file_path: str) -> ParseResult:
        """解析文档，返回结构化结果"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(path, "rb") as f:
            return self.parse_bytes(f.read(), path.name)

    def parse_file(self, file: BinaryIO, filename: str) -> ParseResult:
        """解析文件流"""
        content = file.read()
        return self.parse_bytes(content, filename)

    def parse_bytes(self, content: bytes, filename: str) -> ParseResult:
        """解析字节内容（同步版本，用于非异步环境）"""
        ext = Path(filename).suffix.lower()

        if ext not in self.SUPPORTED_TYPES:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_TYPES}")

        # 在线程池中执行解析，避免阻塞事件循环
        md_content = _do_mineru_parse(
            pdf_bytes=content,
            filename=filename,
            server_url=self.base_url,
            lang=self.lang,
        )

        images = self._extract_images(md_content)
        tables = self._extract_tables(md_content)

        return ParseResult(
            content=md_content,
            images=images,
            tables=tables,
            metadata={"filename": filename},
        )

    async def parse_bytes_async(self, content: bytes, filename: str) -> ParseResult:
        """解析字节内容（异步版本，用于 FastAPI 等异步环境）"""
        ext = Path(filename).suffix.lower()

        if ext not in self.SUPPORTED_TYPES:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_TYPES}")

        # 在线程池中执行解析，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        md_content = await loop.run_in_executor(
            _executor,
            _do_mineru_parse,
            content,
            filename,
            self.base_url,
            self.lang,
        )

        images = self._extract_images(md_content)
        tables = self._extract_tables(md_content)

        return ParseResult(
            content=md_content,
            images=images,
            tables=tables,
            metadata={"filename": filename},
        )

    def _extract_images(self, content: str) -> list[str]:
        """从 Markdown 内容中提取图片引用"""
        import re
        pattern = r'!\[.*?\]\((.*?)\)'
        return re.findall(pattern, content)

    def _extract_tables(self, content: str) -> list[str]:
        """从 Markdown 内容中提取表格"""
        import re
        pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)*)'
        return re.findall(pattern, content)

    def health_check(self) -> bool:
        """健康检查"""
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """关闭客户端"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
