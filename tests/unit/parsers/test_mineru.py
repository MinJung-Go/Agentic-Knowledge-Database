"""MinerUParser 单元测试"""
import pytest
import respx
import httpx
from io import BytesIO

from core.parsers.mineru import MinerUParser, ParseResult


class TestParseResult:
    """ParseResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = ParseResult(
            content="# 标题\n\n内容",
            images=["image1.png"],
            tables=["| col1 | col2 |"],
            metadata={"filename": "test.pdf"},
        )
        assert result.content == "# 标题\n\n内容"
        assert len(result.images) == 1
        assert len(result.tables) == 1
        assert result.metadata["filename"] == "test.pdf"


class TestMinerUParser:
    """MinerUParser 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        parser = MinerUParser()
        assert parser.base_url is not None
        assert parser.timeout == 120.0

    def test_init_custom(self):
        """测试自定义初始化"""
        parser = MinerUParser(
            base_url="http://custom:8003",
            timeout=60.0,
        )
        assert parser.base_url == "http://custom:8003"
        assert parser.timeout == 60.0

    @respx.mock
    def test_parse_bytes_success(self):
        """测试解析字节成功"""
        mock_response = {
            "content": "# 文档标题\n\n这是文档内容。"
        }

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        result = parser.parse_bytes(b"PDF content", "test.pdf")

        assert isinstance(result, ParseResult)
        assert "文档标题" in result.content
        assert result.metadata["filename"] == "test.pdf"

    @respx.mock
    def test_parse_bytes_with_images(self):
        """测试解析包含图片的文档"""
        mock_response = {
            "content": "# 标题\n\n![图片描述](image.png)\n\n内容"
        }

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        result = parser.parse_bytes(b"PDF content", "test.pdf")

        assert len(result.images) == 1
        assert result.images[0] == "image.png"

    @respx.mock
    def test_parse_bytes_with_tables(self):
        """测试解析包含表格的文档"""
        mock_response = {
            "content": "# 标题\n\n| 列1 | 列2 |\n|---|---|\n| 值1 | 值2 |\n"
        }

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        result = parser.parse_bytes(b"PDF content", "test.pdf")

        assert len(result.tables) == 1

    @respx.mock
    def test_parse_bytes_empty_content(self):
        """测试解析空内容"""
        mock_response = {"content": ""}

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        result = parser.parse_bytes(b"empty", "empty.pdf")

        assert result.content == ""
        assert len(result.images) == 0
        assert len(result.tables) == 0

    @respx.mock
    def test_parse_file(self):
        """测试解析文件流"""
        mock_response = {"content": "文件内容"}

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        file_obj = BytesIO(b"PDF content")
        result = parser.parse_file(file_obj, "test.pdf")

        assert result.content == "文件内容"

    @respx.mock
    def test_parse_timeout(self):
        """测试请求超时"""
        respx.post("http://localhost:8003/api/parse").mock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        parser = MinerUParser(base_url="http://localhost:8003", timeout=1.0)

        with pytest.raises(httpx.TimeoutException):
            parser.parse_bytes(b"content", "test.pdf")

    @respx.mock
    def test_parse_http_error(self):
        """测试 HTTP 错误"""
        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        parser = MinerUParser(base_url="http://localhost:8003")

        with pytest.raises(httpx.HTTPStatusError):
            parser.parse_bytes(b"content", "test.pdf")

    @respx.mock
    def test_health_check_success(self):
        """测试健康检查成功"""
        respx.get("http://localhost:8003/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        assert parser.health_check() is True

    @respx.mock
    def test_health_check_failure(self):
        """测试健康检查失败"""
        respx.get("http://localhost:8003/health").mock(
            return_value=httpx.Response(500)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        assert parser.health_check() is False

    def test_health_check_connection_error(self):
        """测试健康检查连接错误"""
        parser = MinerUParser(base_url="http://invalid-host:9999")
        # 连接失败应返回 False
        assert parser.health_check() is False

    def test_extract_images(self):
        """测试提取图片"""
        parser = MinerUParser()
        content = "文本 ![图1](img1.png) 更多文本 ![图2](img2.jpg)"
        images = parser._extract_images(content)

        assert len(images) == 2
        assert "img1.png" in images
        assert "img2.jpg" in images

    def test_extract_images_no_images(self):
        """测试无图片"""
        parser = MinerUParser()
        content = "纯文本内容，没有图片"
        images = parser._extract_images(content)

        assert len(images) == 0

    def test_extract_tables(self):
        """测试提取表格"""
        parser = MinerUParser()
        content = """
# 标题

| 姓名 | 年龄 |
|------|------|
| 张三 | 25 |
| 李四 | 30 |

其他内容
"""
        tables = parser._extract_tables(content)

        assert len(tables) == 1

    def test_extract_tables_no_tables(self):
        """测试无表格"""
        parser = MinerUParser()
        content = "纯文本内容，没有表格"
        tables = parser._extract_tables(content)

        assert len(tables) == 0

    def test_context_manager(self):
        """测试上下文管理器"""
        with MinerUParser(base_url="http://localhost:8003") as parser:
            assert parser.client is not None

    @respx.mock
    def test_parse_method(self):
        """测试 parse 方法返回字符串"""
        mock_response = {"content": "解析内容"}

        respx.post("http://localhost:8003/api/parse").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        parser = MinerUParser(base_url="http://localhost:8003")
        # parse_document 需要真实文件，这里测试 parse_bytes 后的 content
        result = parser.parse_bytes(b"content", "test.pdf")
        assert isinstance(result.content, str)
