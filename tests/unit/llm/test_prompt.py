"""PromptManager 单元测试"""
import pytest

from core.llm.prompt import (
    PromptTemplate,
    PromptManager,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    SUMMARY_PROMPT,
    QA_EXTRACTION_PROMPT,
    prompt_manager,
)


class TestPromptTemplate:
    """PromptTemplate 测试"""

    def test_init(self):
        """测试初始化"""
        template = PromptTemplate(
            template="Hello, $name!",
            variables=["name"],
        )
        assert template.template == "Hello, $name!"
        assert template.variables == ["name"]

    def test_format_single_variable(self):
        """测试单变量格式化"""
        template = PromptTemplate("Hello, $name!")
        result = template.format(name="World")
        assert result == "Hello, World!"

    def test_format_multiple_variables(self):
        """测试多变量格式化"""
        template = PromptTemplate("$greeting, $name!")
        result = template.format(greeting="Hi", name="Alice")
        assert result == "Hi, Alice!"

    def test_format_missing_variable(self):
        """测试缺失变量（safe_substitute）"""
        template = PromptTemplate("Hello, $name! Your age is $age.")
        result = template.format(name="Bob")
        # safe_substitute 保留未替换的变量
        assert result == "Hello, Bob! Your age is $age."

    def test_callable(self):
        """测试 __call__ 方法"""
        template = PromptTemplate("Hello, $name!")
        result = template(name="World")
        assert result == "Hello, World!"


class TestBuiltinTemplates:
    """内置模板测试"""

    def test_rag_system_prompt(self):
        """测试 RAG 系统提示词"""
        result = RAG_SYSTEM_PROMPT.format()
        assert "知识库助手" in result
        assert "参考文档" in result

    def test_rag_user_prompt(self):
        """测试 RAG 用户提示词"""
        result = RAG_USER_PROMPT.format(
            context="这是参考文档内容",
            question="这是用户问题",
        )
        assert "这是参考文档内容" in result
        assert "这是用户问题" in result

    def test_summary_prompt(self):
        """测试摘要提示词"""
        result = SUMMARY_PROMPT.format(
            content="文档内容",
            max_length="200",
        )
        assert "文档内容" in result
        assert "200" in result

    def test_qa_extraction_prompt(self):
        """测试问答提取提示词"""
        result = QA_EXTRACTION_PROMPT.format(
            content="文档内容",
            num_qa="5",
        )
        assert "文档内容" in result
        assert "5" in result
        assert "JSON" in result


class TestPromptManager:
    """PromptManager 测试"""

    def test_init(self):
        """测试初始化"""
        manager = PromptManager()
        assert "rag_system" in manager.templates
        assert "rag_user" in manager.templates
        assert "summary" in manager.templates
        assert "qa_extraction" in manager.templates

    def test_get_existing_template(self):
        """测试获取已存在的模板"""
        manager = PromptManager()
        template = manager.get("rag_system")
        assert isinstance(template, PromptTemplate)

    def test_get_nonexistent_template(self):
        """测试获取不存在的模板"""
        manager = PromptManager()
        with pytest.raises(KeyError):
            manager.get("nonexistent")

    def test_register_template(self):
        """测试注册模板"""
        manager = PromptManager()
        new_template = PromptTemplate("Custom: $message")
        manager.register("custom", new_template)

        assert "custom" in manager.templates
        result = manager.get("custom").format(message="test")
        assert result == "Custom: test"

    def test_format(self):
        """测试格式化"""
        manager = PromptManager()
        result = manager.format(
            "rag_user",
            context="测试上下文",
            question="测试问题",
        )
        assert "测试上下文" in result
        assert "测试问题" in result

    def test_build_rag_messages_default_system(self):
        """测试构建 RAG 消息（默认系统提示词）"""
        manager = PromptManager()
        messages = manager.build_rag_messages(
            question="什么是人工智能？",
            context="人工智能是计算机科学的一个分支。",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "知识库助手" in messages[0]["content"]
        assert "人工智能是计算机科学的一个分支" in messages[1]["content"]
        assert "什么是人工智能？" in messages[1]["content"]

    def test_build_rag_messages_custom_system(self):
        """测试构建 RAG 消息（自定义系统提示词）"""
        manager = PromptManager()
        custom_system = "你是一个专业的技术顾问。"
        messages = manager.build_rag_messages(
            question="问题",
            context="上下文",
            system_prompt=custom_system,
        )

        assert messages[0]["content"] == custom_system


class TestGlobalPromptManager:
    """全局 prompt_manager 测试"""

    def test_global_instance(self):
        """测试全局实例"""
        assert isinstance(prompt_manager, PromptManager)

    def test_global_instance_has_templates(self):
        """测试全局实例包含模板"""
        assert len(prompt_manager.templates) >= 4

    def test_global_instance_format(self):
        """测试全局实例格式化"""
        result = prompt_manager.format("rag_system")
        assert len(result) > 0
