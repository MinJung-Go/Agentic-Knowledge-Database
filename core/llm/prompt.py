"""Prompt 模板管理"""
from string import Template


class PromptTemplate:
    """Prompt 模板"""

    def __init__(self, template: str, variables: list[str] | None = None):
        self.template = template
        self.variables = variables or []
        self._template = Template(template)

    def format(self, **kwargs) -> str:
        """格式化模板"""
        return self._template.safe_substitute(**kwargs)

    def __call__(self, **kwargs) -> str:
        return self.format(**kwargs)


RAG_SYSTEM_PROMPT = PromptTemplate(
    template="""你是一个专业的知识库助手。请根据提供的参考文档回答用户问题。

要求：
1. 只根据参考文档中的内容回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确告知用户
3. 回答要准确、简洁、专业
4. 如有必要，可以引用文档中的具体内容""",
    variables=[],
)


RAG_USER_PROMPT = PromptTemplate(
    template="""参考文档：
$context

用户问题：$question

请根据参考文档回答用户问题。""",
    variables=["context", "question"],
)


SUMMARY_PROMPT = PromptTemplate(
    template="""请对以下文档内容进行摘要：

$content

要求：
1. 提取关键信息
2. 保持逻辑完整
3. 摘要长度控制在 $max_length 字以内""",
    variables=["content", "max_length"],
)


QA_EXTRACTION_PROMPT = PromptTemplate(
    template="""请根据以下文档内容，生成问答对：

$content

要求：
1. 生成 $num_qa 个问答对
2. 问题要具体、有价值
3. 答案要准确、完整
4. 输出格式为 JSON 数组：[{"question": "...", "answer": "..."}]""",
    variables=["content", "num_qa"],
)


class PromptManager:
    """Prompt 管理器"""

    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {
            "rag_system": RAG_SYSTEM_PROMPT,
            "rag_user": RAG_USER_PROMPT,
            "summary": SUMMARY_PROMPT,
            "qa_extraction": QA_EXTRACTION_PROMPT,
        }

    def get(self, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in self.templates:
            raise KeyError(f"模板 {name} 不存在")
        return self.templates[name]

    def register(self, name: str, template: PromptTemplate):
        """注册模板"""
        self.templates[name] = template

    def format(self, name: str, **kwargs) -> str:
        """格式化模板"""
        return self.get(name).format(**kwargs)

    def build_rag_messages(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """构建 RAG 消息"""
        if system_prompt is None:
            system_prompt = self.format("rag_system")

        user_prompt = self.format("rag_user", context=context, question=question)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


prompt_manager = PromptManager()
