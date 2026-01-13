"""Gradio Web Demo - 知识库问答系统

功能：
1. 文档上传与管理（含更新）
2. 知识库问答（支持层级过滤）
3. 流式/非流式回答切换
4. API 健康检查

启动方式：
    # 先启动后端服务
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

    # 再启动 Gradio Demo
    python -m app.demo
    # 或
    gradio app/demo.py

环境变量：
    KNOWLEDGE_API_URL: 后端 API 地址（默认 http://localhost:8000）
"""

import os
import gradio as gr
import httpx
from typing import Generator

# API 配置（支持环境变量）
API_BASE_URL = os.getenv("KNOWLEDGE_API_URL", "http://localhost:8000")


# ==================== 辅助函数 ====================

def check_api_health() -> str:
    """检查 API 服务状态"""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            return "✅ API 服务正常运行"
        return f"⚠️ API 响应异常: {response.status_code}"
    except httpx.ConnectError:
        return "❌ 无法连接到 API 服务"
    except Exception as e:
        return f"❌ 检查失败: {str(e)}"


def get_scope_text(user_id: str, knowledge_id: str, doc_id: str) -> str:
    """获取检索范围描述"""
    if knowledge_id and doc_id:
        return f"知识库 [{knowledge_id}] 中的文档 [{doc_id}]"
    elif knowledge_id:
        return f"知识库 [{knowledge_id}]"
    elif doc_id:
        return f"文档 [{doc_id}]"
    return "用户所有知识库"


# ==================== API 调用函数 ====================

def upload_document(user_id: str, knowledge_id: str, file) -> str:
    """上传文档到知识库"""
    if not file:
        return "❌ 请选择文件"
    if not user_id or not knowledge_id:
        return "❌ 请填写用户ID和知识库ID"

    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name.split("/")[-1], f)}
            data = {"userId": user_id, "knowledgeId": knowledge_id}

            response = httpx.post(
                f"{API_BASE_URL}/knowledge/documents/create",
                data=data,
                files=files,
                timeout=120.0,
            )

        if response.status_code == 200:
            result = response.json()
            return f"""✅ 文档上传成功！

📄 文档ID: `{result.get('docId', 'N/A')}`
📚 知识库: {result.get('knowledgeId', 'N/A')}
📊 分块数: {result.get('chunkCount', 0)}
📁 文件名: {result.get('filename', 'N/A')}
⏰ 状态: {result.get('status', 'N/A')}
"""
        else:
            return f"❌ 上传失败: {response.text}"

    except httpx.ConnectError:
        return "❌ 无法连接到服务器，请确保 API 服务已启动"
    except Exception as e:
        return f"❌ 上传出错: {str(e)}"


def update_document(user_id: str, knowledge_id: str, doc_id: str, file) -> str:
    """更新已有文档"""
    if not file:
        return "❌ 请选择新文件"
    if not user_id or not knowledge_id or not doc_id:
        return "❌ 请填写完整的用户ID、知识库ID和文档ID"

    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name.split("/")[-1], f)}
            data = {
                "userId": user_id,
                "knowledgeId": knowledge_id,
                "docId": doc_id,
            }

            response = httpx.post(
                f"{API_BASE_URL}/knowledge/documents/update",
                data=data,
                files=files,
                timeout=120.0,
            )

        if response.status_code == 200:
            result = response.json()
            return f"""✅ 文档更新成功！

📄 文档ID: `{result.get('docId', 'N/A')}`
📚 知识库: {result.get('knowledgeId', 'N/A')}
📊 新分块数: {result.get('chunkCount', 0)}
📁 文件名: {result.get('filename', 'N/A')}
⏰ 状态: {result.get('status', 'N/A')}
"""
        elif response.status_code == 404:
            return "❌ 文档不存在或无权限更新"
        else:
            return f"❌ 更新失败: {response.text}"

    except httpx.ConnectError:
        return "❌ 无法连接到服务器"
    except Exception as e:
        return f"❌ 更新出错: {str(e)}"


def query_documents(user_id: str, knowledge_id: str) -> str:
    """查询知识库中的文档列表"""
    if not user_id or not knowledge_id:
        return "❌ 请填写用户ID和知识库ID"

    try:
        response = httpx.post(
            f"{API_BASE_URL}/knowledge/documents/query",
            json={"userId": user_id, "knowledgeId": knowledge_id},
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            documents = result.get("documents", [])

            if not documents:
                return "📭 该知识库暂无文档"

            output = f"📚 知识库 [{knowledge_id}] 共有 {len(documents)} 个文档：\n\n"
            for i, doc in enumerate(documents, 1):
                output += f"""**{i}. {doc.get('filename', 'N/A')}**
   - 文档ID: `{doc.get('docId', 'N/A')}`
   - 分块数: {doc.get('chunkCount', 0)}
   - 状态: {doc.get('status', 'N/A')}

"""
            return output
        else:
            return f"❌ 查询失败: {response.text}"

    except httpx.ConnectError:
        return "❌ 无法连接到服务器"
    except Exception as e:
        return f"❌ 查询出错: {str(e)}"


def delete_document(user_id: str, knowledge_id: str, doc_id: str) -> str:
    """删除指定文档"""
    if not user_id or not knowledge_id or not doc_id:
        return "❌ 请填写完整的用户ID、知识库ID和文档ID"

    try:
        response = httpx.post(
            f"{API_BASE_URL}/knowledge/documents/delete",
            json={
                "userId": user_id,
                "knowledgeId": knowledge_id,
                "docId": doc_id,
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("deleted"):
                return f"✅ {result.get('message', '删除成功')}"
            else:
                return f"⚠️ {result.get('message', '删除失败')}"
        else:
            return f"❌ 删除失败: {response.text}"

    except httpx.ConnectError:
        return "❌ 无法连接到服务器"
    except Exception as e:
        return f"❌ 删除出错: {str(e)}"


def chat_stream(
    user_id: str,
    knowledge_id: str,
    doc_id: str,
    question: str,
    top_k: int,
    history: list,
) -> Generator:
    """与知识库对话（流式输出）"""
    if not user_id:
        yield history + [[question, "❌ 请填写用户ID"]]
        return

    if not question.strip():
        yield history + [[question, "❌ 请输入问题"]]
        return

    # 构建请求参数
    payload = {
        "userId": user_id,
        "question": question,
        "topK": top_k,
        "stream": True,
    }

    if knowledge_id and knowledge_id.strip():
        payload["knowledgeId"] = knowledge_id
    if doc_id and doc_id.strip():
        payload["docId"] = doc_id

    scope = get_scope_text(user_id, knowledge_id, doc_id)

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/knowledge/chat/stream",
            json=payload,
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                yield history + [[question, f"❌ 请求失败: {response.text}"]]
                return

            answer = ""
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    if data.startswith("[ERROR]"):
                        answer += f"\n\n❌ {data}"
                        break
                    answer += data
                    yield history + [[question, f"🔍 *检索范围: {scope}*\n\n{answer}"]]

            yield history + [[question, f"🔍 *检索范围: {scope}*\n\n{answer}"]]

    except httpx.ConnectError:
        yield history + [[question, "❌ 无法连接到服务器，请确保 API 服务已启动"]]
    except Exception as e:
        yield history + [[question, f"❌ 请求出错: {str(e)}"]]


def chat_non_stream(
    user_id: str,
    knowledge_id: str,
    doc_id: str,
    question: str,
    top_k: int,
    history: list,
) -> tuple:
    """与知识库对话（非流式）"""
    if not user_id:
        return history + [[question, "❌ 请填写用户ID"]], ""

    if not question.strip():
        return history, ""

    payload = {
        "userId": user_id,
        "question": question,
        "topK": top_k,
        "stream": False,
    }

    if knowledge_id and knowledge_id.strip():
        payload["knowledgeId"] = knowledge_id
    if doc_id and doc_id.strip():
        payload["docId"] = doc_id

    scope = get_scope_text(user_id, knowledge_id, doc_id)

    try:
        response = httpx.post(
            f"{API_BASE_URL}/knowledge/chat",
            json=payload,
            timeout=120.0,
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "无回答")
            sources = result.get("sources", [])

            source_text = ""
            if sources:
                source_text = "\n\n---\n📚 **参考来源：**\n"
                for i, src in enumerate(sources, 1):
                    source_text += f"\n{i}. [{src.get('docId', 'N/A')}] (相关度: {src.get('score', 0):.2f})\n"
                    content_preview = src.get('content', '')[:100]
                    if content_preview:
                        source_text += f"   > {content_preview}...\n"

            full_answer = f"🔍 *检索范围: {scope}*\n\n{answer}{source_text}"
            return history + [[question, full_answer]], ""
        else:
            return history + [[question, f"❌ 请求失败: {response.text}"]], ""

    except httpx.ConnectError:
        return history + [[question, "❌ 无法连接到服务器"]], ""
    except Exception as e:
        return history + [[question, f"❌ 请求出错: {str(e)}"]], ""


def chat_handler(
    user_id: str,
    knowledge_id: str,
    doc_id: str,
    question: str,
    top_k: int,
    use_stream: bool,
    history: list,
):
    """统一的聊天处理器，根据 use_stream 选择模式"""
    if use_stream:
        yield from chat_stream(user_id, knowledge_id, doc_id, question, top_k, history)
    else:
        result, _ = chat_non_stream(user_id, knowledge_id, doc_id, question, top_k, history)
        yield result


def clear_history() -> tuple:
    """清空对话历史"""
    return [], ""


# ==================== Gradio UI ====================

def create_demo():
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="知识库问答系统",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .status-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # 🧠 企业知识库问答系统

            基于 RAG (检索增强生成) 技术，支持文档上传、智能检索和问答对话。

            ---
            """
        )

        # 顶部状态栏
        with gr.Row():
            api_status = gr.Markdown(value=check_api_health())
            refresh_btn = gr.Button("🔄 刷新状态", size="sm", scale=0)

        refresh_btn.click(fn=check_api_health, outputs=[api_status])

        with gr.Tabs():
            # ==================== Tab 1: 知识问答 ====================
            with gr.TabItem("💬 知识问答", id="chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 检索设置")

                        chat_user_id = gr.Textbox(
                            label="用户ID",
                            placeholder="user_001",
                            value="user_001",
                        )
                        chat_knowledge_id = gr.Textbox(
                            label="知识库ID（可选）",
                            placeholder="留空则搜索所有知识库",
                        )
                        chat_doc_id = gr.Textbox(
                            label="文档ID（可选）",
                            placeholder="留空则搜索所有文档",
                        )
                        chat_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="召回数量 (Top-K)",
                        )
                        chat_stream_toggle = gr.Checkbox(
                            label="流式输出",
                            value=True,
                            info="开启后实时显示生成内容",
                        )

                        gr.Markdown(
                            """
                            ---
                            **📌 过滤规则：**
                            - 只填用户ID → 全局搜索
                            - 填知识库ID → 知识库内搜索
                            - 填文档ID → 精确到文档
                            """
                        )

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=500,
                            show_copy_button=True,
                        )

                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入您的问题...",
                                scale=4,
                                show_label=False,
                            )
                            chat_btn = gr.Button("发送", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空对话", scale=1)

                # 绑定事件
                chat_btn.click(
                    fn=chat_handler,
                    inputs=[
                        chat_user_id,
                        chat_knowledge_id,
                        chat_doc_id,
                        chat_input,
                        chat_top_k,
                        chat_stream_toggle,
                        chatbot,
                    ],
                    outputs=[chatbot],
                ).then(
                    fn=lambda: "",
                    outputs=[chat_input],
                )

                chat_input.submit(
                    fn=chat_handler,
                    inputs=[
                        chat_user_id,
                        chat_knowledge_id,
                        chat_doc_id,
                        chat_input,
                        chat_top_k,
                        chat_stream_toggle,
                        chatbot,
                    ],
                    outputs=[chatbot],
                ).then(
                    fn=lambda: "",
                    outputs=[chat_input],
                )

                clear_btn.click(
                    fn=clear_history,
                    outputs=[chatbot, chat_input],
                )

            # ==================== Tab 2: 文档上传 ====================
            with gr.TabItem("📤 文档上传", id="upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 上传新文档")

                        upload_user_id = gr.Textbox(
                            label="用户ID",
                            placeholder="user_001",
                            value="user_001",
                        )
                        upload_knowledge_id = gr.Textbox(
                            label="知识库ID",
                            placeholder="kb_001",
                            value="kb_001",
                        )
                        upload_file = gr.File(
                            label="选择文档",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        )
                        upload_btn = gr.Button("📤 上传文档", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 上传结果")
                        upload_result = gr.Markdown("等待上传...")

                upload_btn.click(
                    fn=upload_document,
                    inputs=[upload_user_id, upload_knowledge_id, upload_file],
                    outputs=[upload_result],
                )

            # ==================== Tab 3: 文档管理 ====================
            with gr.TabItem("📁 文档管理", id="manage"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 查询文档列表")

                        query_user_id = gr.Textbox(
                            label="用户ID",
                            placeholder="user_001",
                            value="user_001",
                        )
                        query_knowledge_id = gr.Textbox(
                            label="知识库ID",
                            placeholder="kb_001",
                            value="kb_001",
                        )
                        query_btn = gr.Button("🔍 查询文档", variant="primary")

                        gr.Markdown("---")
                        gr.Markdown("### 更新文档")

                        update_user_id = gr.Textbox(
                            label="用户ID",
                            placeholder="user_001",
                            value="user_001",
                        )
                        update_knowledge_id = gr.Textbox(
                            label="知识库ID",
                            placeholder="kb_001",
                        )
                        update_doc_id = gr.Textbox(
                            label="要更新的文档ID",
                            placeholder="doc_xxx",
                        )
                        update_file = gr.File(
                            label="选择新文件",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        )
                        update_btn = gr.Button("🔄 更新文档", variant="secondary")

                        gr.Markdown("---")
                        gr.Markdown("### 删除文档")

                        delete_user_id = gr.Textbox(
                            label="用户ID",
                            placeholder="user_001",
                            value="user_001",
                        )
                        delete_knowledge_id = gr.Textbox(
                            label="知识库ID",
                            placeholder="kb_001",
                        )
                        delete_doc_id = gr.Textbox(
                            label="文档ID",
                            placeholder="doc_xxx",
                        )
                        delete_btn = gr.Button("🗑️ 删除文档", variant="stop")

                    with gr.Column():
                        gr.Markdown("### 操作结果")
                        manage_result = gr.Markdown("等待操作...")

                query_btn.click(
                    fn=query_documents,
                    inputs=[query_user_id, query_knowledge_id],
                    outputs=[manage_result],
                )

                update_btn.click(
                    fn=update_document,
                    inputs=[update_user_id, update_knowledge_id, update_doc_id, update_file],
                    outputs=[manage_result],
                )

                delete_btn.click(
                    fn=delete_document,
                    inputs=[delete_user_id, delete_knowledge_id, delete_doc_id],
                    outputs=[manage_result],
                )

            # ==================== Tab 4: 使用说明 ====================
            with gr.TabItem("📖 使用说明", id="help"):
                gr.Markdown(
                    f"""
                    ## 🚀 快速开始

                    ### 1. 启动后端服务

                    ```bash
                    # 激活 conda 环境
                    source /root/miniforge3/bin/activate agent

                    # 启动 FastAPI 服务
                    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
                    ```

                    ### 2. 启动 Gradio Demo

                    ```bash
                    # 方式一：直接运行
                    python -m app.demo

                    # 方式二：使用 gradio 命令
                    gradio app/demo.py

                    # 方式三：指定自定义 API 地址
                    KNOWLEDGE_API_URL=http://your-api-server:8000 python -m app.demo
                    ```

                    **当前 API 地址：** `{API_BASE_URL}`

                    ### 3. 上传文档

                    1. 切换到「📤 文档上传」标签页
                    2. 填写用户ID和知识库ID
                    3. 选择 PDF/Word/TXT/Markdown 文件
                    4. 点击「上传文档」

                    ### 4. 开始问答

                    1. 切换到「💬 知识问答」标签页
                    2. 填写用户ID（必填）
                    3. 可选填写知识库ID或文档ID来限定检索范围
                    4. 输入问题并发送

                    ---

                    ## 🔍 检索范围说明

                    | 填写参数 | 检索范围 |
                    |----------|----------|
                    | 只填用户ID | 搜索该用户的所有知识库 |
                    | 用户ID + 知识库ID | 只在指定知识库中搜索 |
                    | 用户ID + 文档ID | 只在指定文档中搜索 |
                    | 全部填写 | 最精确的范围限定 |

                    ---

                    ## 📚 支持的文档格式

                    - PDF (.pdf)
                    - Word (.docx, .doc)
                    - 纯文本 (.txt)
                    - Markdown (.md)

                    ---

                    ## 🔄 流式输出

                    开启「流式输出」选项后，模型生成的回答会实时显示，无需等待完整生成。
                    适合长回答场景，提升用户体验。

                    ---

                    ## ⚠️ 常见问题

                    **Q: 无法连接到服务器？**

                    A: 请确保：
                    - 后端 API 服务已启动
                    - API 地址配置正确（默认 `http://localhost:8000`）
                    - 可通过环境变量 `KNOWLEDGE_API_URL` 修改

                    **Q: 上传文档失败？**

                    A: 请检查：
                    - 文件格式是否支持
                    - 文件大小是否过大
                    - MinerU 解析服务是否正常

                    **Q: 回答质量不理想？**

                    A: 可以尝试：
                    - 增加 Top-K 召回数量
                    - 缩小检索范围（指定知识库或文档）
                    - 优化问题的表述方式

                    **Q: 如何更新已有文档？**

                    A: 切换到「📁 文档管理」标签页，填写文档ID和新文件，点击「更新文档」。
                    系统会自动删除旧内容并重新解析存储。
                    """
                )

        gr.Markdown(
            """
            ---
            <center>

            **🧠 企业知识库问答系统** | 基于 RAG 技术 | Powered by Qwen3 + Milvus

            </center>
            """
        )

    return demo


# ==================== 主入口 ====================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
