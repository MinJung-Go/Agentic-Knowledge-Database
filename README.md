# Agentic-Knowledge-Database

> **AI 应用开发实战** — 课程一：手把手教你打造企业知识库

从零构建企业级 AI 知识库系统，覆盖完整生命周期：需求分析 → 架构设计 → 开发实现 → 部署上线。

---

## 模块一：项目规划与需求分析

### 1.1 企业知识库的业务场景与痛点

**典型场景：**
- 内部文档检索（技术文档、规章制度、产品手册）
- 智能客服问答
- 员工知识助手
- 合规审查辅助

**核心痛点：**
- 信息分散在多个系统，检索效率低
- 传统关键词搜索无法理解语义
- 知识更新后难以同步
- 缺乏智能问答能力

### 1.2 需求调研与边界定义

**功能边界：**
- 支持的文档类型：PDF、Word、Markdown、TXT
- 支持的交互方式：API 接口
- 支持的语言：中文为主

**非功能需求：**
- 响应时间 < 3s
- 支持并发用户数 > 100
- 数据完全私有化部署

### 1.3 数据源梳理

| 数据类型 | 格式 | 预估数量 | 更新频率 |
|---------|------|---------|---------|
| 技术文档 | PDF/MD | 1000+ | 周 |
| 产品手册 | Word/PDF | 500+ | 月 |
| FAQ | TXT/MD | 2000+ | 日 |

---

## 模块二：技术选型与架构设计

### 2.1 RAG 技术原理

RAG（Retrieval-Augmented Generation）通过检索增强的方式，让 LLM 基于私有知识库生成准确回答。

```
用户提问 → 向量召回 → Rerank 重排序 → 构建 Prompt → LLM 生成回答
    │           │              │              │              │
    │     Embedding        Cross-Encoder   System+User    Qwen3-VL
    │     相似度检索        精排相关性       Prompt模板     流式生成
```

### 2.2 自部署技术选型

| 组件 | 选型 | 部署命令 |
|------|------|------|----------|
| 文档解析 | MinerU | `docker run mineru-api` |
| LLM + Vision | vLLM + Qwen3-VL-8B | `vllm serve Qwen/Qwen3-VL-8B` |
| Embedding | Qwen3-Embedding-0.6B | `vllm serve Qwen/Qwen3-Embedding-0.6B --task embed` |
| Rerank | Qwen3-Reranker-8B | `vllm serve Qwen/Qwen3-Reranker-8B --task score` |
| 向量数据库 | Milvus 2.x | `docker-compose up milvus` |

---

#### 2.2.1 LLM 推理服务对比

| 框架 | 定位 | 性能 | 适用场景 |
|------|------|------|----------|
| **vLLM** | 生产级高性能推理 | 吞吐量比 Ollama 高 3.23x，峰值负载下 RPS 可达 llama.cpp 的 35x | 多用户生产环境、低延迟要求 |
| **Ollama** | 开发友好、开箱即用 | 单命令安装，支持离线运行 | 快速原型开发、个人使用 |
| **LocalAI** | OpenAI API 替代方案 | 支持文本/图像/语音/向量，内置向量存储 | 需要完整 OpenAI 兼容栈 |

**为什么选择 vLLM：**
- **PagedAttention 技术**：创新的内存管理机制，显著提升推理效率
- **高并发能力**：基准测试显示吞吐量比标准 HuggingFace Transformers 高 24x
- **硬件灵活性**：支持 NVIDIA/AMD GPU、Intel CPU、TPU 等多种硬件
- **OpenAI 兼容 API**：无缝迁移云服务到本地部署

> 💡 **推荐策略**：开发阶段使用 Ollama 快速迭代，生产部署切换到 vLLM 或 SGLang

---

#### 2.2.2 向量数据库对比

| 数据库 | 核心特性 | 性能表现 | 适用规模 |
|--------|----------|----------|----------|
| **Milvus** | 分布式架构、丰富索引（IVF/HNSW/DiskANN）、分层存储 | 召回率 <0.95 时吞吐量最高 | 十亿级向量、企业级 |
| **Qdrant** | Rust 实现、强过滤能力、低资源占用 | 延迟稳定，过滤与搜索一体化执行 | 中小规模、边缘部署 |
| **Weaviate** | 混合搜索（BM25+ANN）、知识图谱、GraphQL | 5000 万向量内高效，更大规模需更多资源 | 需要结构化+语义混合查询 |

**为什么选择 Milvus：**
- **生产验证**：在十亿级向量场景有长期稳定运行记录
- **索引丰富**：支持的索引策略最多（IVF、HNSW、DiskANN 等）
- **Kubernetes 原生**：天然支持分布式部署和弹性扩展
- **开源可控**：Apache 2.0 许可，完全私有化部署

> ⚠️ **注意**：Milvus 运维复杂度较高，适合有数据工程能力的团队。小规模场景可考虑 Qdrant。

---

#### 2.2.3 Embedding 模型选型

| 模型 | 参数量 | 向量维度 | MTEB 得分 | 特点 |
|------|--------|----------|-----------|------|
| **Qwen3-Embedding-8B** | 8B | 4096 | 70.58（多语言 #1） | 最强效果，适合质量优先场景 |
| **Qwen3-Embedding-0.6B** | 0.6B | 1024 | - | 轻量高效，适合资源受限场景 |
| **BGE-M3** | 0.5B | 1024 | 66.x | 多语言稳定，社区成熟 |

**Qwen3 Embedding 优势：**
- **MTEB 多语言榜首**：2025.06 得分 70.58，超越所有开源和闭源模型
- **代码理解**：MTEB Code 得分 80.68，超越 Gemini-Embedding
- **多语言支持**：覆盖 100+ 语言，包括编程语言
- **Apache 2.0 许可**：完全开源，可商用

> 💡 **本项目选择 0.6B 版本**：平衡效果与资源消耗，1024 维向量存储成本更低

---

#### 2.2.4 Rerank 模型选型

| 模型 | 类型 | 特点 |
|------|------|------|
| **Qwen3-Reranker-8B** | 交叉编码器 | 指令感知，支持任务定制化 prompt |
| **BGE-Reranker** | 交叉编码器 | 中文效果好，资源占用低 |
| **Cohere Rerank** | API 服务 | 效果顶尖，但需联网 |

**为什么选择 Qwen3-Reranker：**
- 与 Qwen3 Embedding 同系列，语义理解一致性更好
- 支持自定义 rerank 指令，可针对不同任务优化
- 本地部署，数据不出域

---

#### 2.2.5 服务部署汇总

```bash
# 1. Milvus（向量数据库）
docker-compose -f docker-compose-milvus.yml up -d

# 2. vLLM 服务（需要 GPU）
# LLM + Vision
vllm serve Qwen/Qwen3-VL-8B --port 8000

# Embedding
vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001

# Rerank
vllm serve Qwen/Qwen3-Reranker-8B --task score --port 8002

# 3. MinerU（文档解析）
docker run -d -p 8003:8003 mineru-api
```

**资源需求参考：**

| 服务 | 最低 GPU 显存 | 推荐配置 |
|------|---------------|----------|
| Qwen3-VL-8B | 16GB | 24GB (A10/4090) |
| Qwen3-Embedding-0.6B | 2GB | 4GB |
| Qwen3-Reranker-8B | 16GB | 24GB |
| Milvus | - | 16GB RAM, SSD |

---

#### 2.2.6 服务地址配置

> ⚙️ **所有底层服务地址均可在 `configs/settings.py` 中配置，支持通过环境变量或 `.env` 文件覆盖。**

**可配置项一览：**

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `mineru_base_url` | `MINERU_BASE_URL` | `http://localhost:8003` | MinerU 文档解析服务 |
| `llm_base_url` | `LLM_BASE_URL` | `http://localhost:8000` | LLM 推理服务（vLLM） |
| `llm_model` | `LLM_MODEL` | `Qwen/Qwen3-VL-8B` | LLM 模型名称 |
| `embedding_base_url` | `EMBEDDING_BASE_URL` | `http://localhost:8001` | Embedding 向量化服务 |
| `embedding_model` | `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding 模型名称 |
| `rerank_base_url` | `RERANK_BASE_URL` | `http://localhost:8002` | Rerank 重排序服务 |
| `rerank_model` | `RERANK_MODEL` | `Qwen/Qwen3-Reranker-8B` | Rerank 模型名称 |
| `milvus_host` | `MILVUS_HOST` | `localhost` | Milvus 服务地址 |
| `milvus_port` | `MILVUS_PORT` | `19530` | Milvus 服务端口 |

**配置方式：**

```bash
# 方式 1: 环境变量
export MILVUS_HOST=192.168.1.100
export LLM_BASE_URL=http://gpu-server:8000

# 方式 2: .env 文件（推荐）
cat > .env << EOF
MILVUS_HOST=192.168.1.100
MILVUS_PORT=19530
LLM_BASE_URL=http://gpu-server:8000
EMBEDDING_BASE_URL=http://gpu-server:8001
RERANK_BASE_URL=http://gpu-server:8002
MINERU_BASE_URL=http://gpu-server:8003
EOF
```

> 💡 参考 `.env.example` 获取完整配置模板

---

> 📚 **参考资料：**
> - [vLLM vs Ollama 完整对比](https://blog.alphabravo.io/ollama-vs-vllm-the-definitive-guide-to-local-llm-frameworks-in-2025/)
> - [向量数据库对比 2025](https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3)
> - [Qwen3 Embedding 官方博客](https://qwenlm.github.io/blog/qwen3-embedding/)

### 2.3 系统架构

#### 2.3.1 文档处理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         文档上传 API                                 │
│                   POST /knowledge/documents/create                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  原始文档    │     │    MinerU       │     │  Qwen3-VL-8B    │
│ PDF/Word/图片│────▶│  POST /api/parse │────▶│  图表/OCR理解    │
└─────────────┘     │  → Markdown     │     │  (可选增强)      │
                    └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
          ┌─────────────────┐     ┌──────────────────────┐
          │  TextChunker    │     │ Qwen3-Embedding-0.6B │
          │  chunk_size=500 │────▶│ POST /v1/embeddings  │
          │  overlap=50     │     │  → 1024 维向量       │
          └─────────────────┘     └──────────┬───────────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │     Milvus      │
                                   │  Collection:    │
                                   │  - doc_id       │
                                   │  - chunk_id     │
                                   │  - content      │
                                   │  - embedding    │
                                   │  - metadata     │
                                   └─────────────────┘
```

#### 2.3.2 知识问答流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                          问答 API                                    │
│                     POST /knowledge/chat                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌─────────────────┐         ┌─────────────────────┐
          │ Qwen3-Embedding │         │      Milvus         │
          │  问题向量化      │────────▶│  L2 相似度检索       │
          └─────────────────┘         │  Top-K × 3 召回     │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │ Qwen3-Reranker-8B   │
                                      │ POST /v1/rerank     │
                                      │ 交叉编码器精排       │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │   PromptManager     │
                                      │ 构建 System + User  │
                                      │ [文档1] [文档2] ... │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │   Qwen3-VL-8B       │
                                      │ POST /v1/chat/...   │
                                      │ 流式生成回答         │
                                      └─────────────────────┘
```

### 2.4 API 接口设计

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 创建文档 | POST | `/knowledge/documents/create` | 上传并解析文档 |
| 删除文档 | POST | `/knowledge/documents/delete` | 删除文档及向量 |
| 更新文档 | POST | `/knowledge/documents/update` | 重新解析文档 |
| 查询文档 | POST | `/knowledge/documents/query` | 分页查询列表 |
| 知识问答 | POST | `/knowledge/chat` | RAG 问答 |
| 流式问答 | POST | `/knowledge/chat/stream` | SSE 流式输出 |

#### 2.4.1 灵活的层级过滤检索

知识问答接口支持灵活的层级过滤，根据传入参数自动构建检索范围：

| 传入参数 | 检索范围 | 适用场景 |
|----------|----------|----------|
| 只传 `userId` | 用户所有知识库 | 全局搜索 |
| `userId` + `knowledgeId` | 指定知识库 | 知识库内搜索 |
| `userId` + `docId` | 指定文档（跨知识库） | 文档精确定位 |
| 三者都传 | 指定知识库下的指定文档 | 最精确匹配 |

**请求示例：**

```json
// 场景 1: 在用户所有知识库中搜索
{
  "userId": "user_001",
  "question": "什么是机器学习？"
}

// 场景 2: 在指定知识库中搜索
{
  "userId": "user_001",
  "knowledgeId": "kb_001",
  "question": "产品保修政策是什么？"
}

// 场景 3: 直接定位到某个文档
{
  "userId": "user_001",
  "docId": "doc_abc123",
  "question": "这份文档的主要内容是什么？"
}
```

> 🔒 **安全说明**：所有检索操作强制包含 `userId` 过滤，确保用户数据隔离。删除/更新操作会验证文档归属权限。

### 2.5 项目核心目录结构

```
├── app/                          # FastAPI 应用层
│   ├── main.py                   # 应用入口 + 生命周期
│   ├── schemas/                  # Pydantic Schema
│   │   ├── base.py               # 基类（驼峰别名）
│   │   ├── document.py           # 文档 CRUD Schema
│   │   └── chat.py               # 对话 Schema
│   └── routers/                  # API 路由
│       ├── document.py           # 文档管理 API
│       └── chat.py               # 对话 API
│
├── core/                         # 核心业务层
│   ├── parsers/                  # 文档解析
│   │   ├── mineru.py             # MinerU 客户端
│   │   ├── vision.py             # Qwen3-VL 视觉理解
│   │   └── chunker.py            # 文本分块策略
│   ├── embedding/                # 向量化
│   │   └── client.py             # Qwen3-Embedding 客户端
│   ├── milvus/                   # 向量存储
│   │   ├── client.py             # Milvus 连接管理
│   │   └── collection.py         # Collection 操作
│   ├── rerank/                   # 重排序
│   │   └── client.py             # Qwen3-Reranker 客户端
│   ├── retrieval/                # 检索召回
│   │   ├── text.py               # BM25 文本召回
│   │   ├── vector.py             # 向量召回
│   │   └── hybrid.py             # 混合召回
│   ├── llm/                      # LLM 生成
│   │   ├── client.py             # OpenAI SDK 客户端
│   │   └── prompt.py             # Prompt 模板管理
│   └── agent/                    # Agent 编排
│       └── rag.py                # RAG Agent
│
├── configs/                      # 配置管理
│   └── settings.py               # Pydantic Settings
│
├── requirements.txt              # Python 依赖
└── Dockerfile                    # 容器构建
```

---

## 模块三：开发流程

### 3.1 接口设计与 Schema 定义

#### 为什么优先定义接口？

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         接口先行开发模式                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: 定义 Schema          Step 2: Mock 数据           Step 3: 并行开发  │
│   ┌─────────────────┐         ┌─────────────────┐         ┌───────────────┐ │
│   │ Request Schema  │         │ Mock Server     │         │ 前端开发       │ │
│   │ Response Schema │   ──►   │ 返回模拟数据     │   ──►   │ 调用 Mock API │ │
│   │ 字段定义 + 类型   │         │ 符合 Schema     │         │ 界面开发       │ │
│   └─────────────────┘         └─────────────────┘         ├───────────────┤ │
│                                                           │ 后端开发       │ │
│                                                           │ 实现真实逻辑   │ │
│                                                           │ 替换 Mock      │ │
│                                                           └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心价值：**

| 收益 | 说明 |
|------|------|
| **前后端并行** | Schema 确定后，前端可基于 Mock 数据开发，无需等待后端完成 |
| **功能明确** | 清晰的输入输出定义，避免开发中的理解偏差 |
| **团队协作** | 统一的接口文档，减少沟通成本 |
| **快速联调** | Mock 与真实接口一致，联调时只需切换 baseURL |
| **测试前置** | 可基于 Schema 提前编写测试用例 |

---

#### 3.1.1 核心概念

```
User (userId) ──┬── Knowledge (knowledgeId) ──┬── Document (docId) ──┬── Chunk (chunkId)
                │                              │                      │
                │   用户可创建多个知识库         │   知识库包含多个文档    │   文档拆分为多个块
                └──────────────────────────────┴──────────────────────┴──────────────────
```

#### 3.1.2 Schema 基类设计

使用 Pydantic `Field(alias=...)` 实现驼峰命名：

```python
# app/schemas/base.py
from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
```

#### 3.1.3 文档管理 Schema

```python
# app/schemas/document.py

# 创建文档 - Form 表单 + 文件上传
# 输入: userId, knowledgeId (Form) + file (UploadFile)
# 输出:
class CreateDocumentResponse(BaseSchema):
    user_id: str = Field(..., alias="userId")
    doc_id: str = Field(..., alias="docId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    filename: str
    chunk_count: int = Field(..., alias="chunkCount")
    status: str  # processing / completed / failed
    created_at: datetime = Field(..., alias="createdAt")

# 删除文档
class DeleteDocumentRequest(BaseSchema):
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    doc_id: str = Field(..., alias="docId")

class DeleteDocumentResponse(BaseSchema):
    user_id: str = Field(..., alias="userId")
    doc_id: str = Field(..., alias="docId")
    deleted: bool
    message: str

# 查询文档列表
class QueryDocumentRequest(BaseSchema):
    user_id: str = Field(..., alias="userId")
    knowledge_id: str = Field(..., alias="knowledgeId")
    page: int = 1
    page_size: int = Field(default=20, alias="pageSize")
```

#### 3.1.4 对话 Schema

```python
# app/schemas/chat.py

class ChatRequest(BaseSchema):
    """对话请求 - 支持灵活的层级过滤

    过滤场景：
    1. 只传 user_id          → 用户所有知识库
    2. user_id + knowledge_id → 指定知识库
    3. user_id + doc_id       → 直接定位文档（跨知识库）
    4. 三者都传               → 指定知识库下的指定文档
    """
    user_id: str = Field(..., alias="userId")
    knowledge_id: str | None = Field(default=None, alias="knowledgeId")  # 可选
    doc_id: str | None = Field(default=None, alias="docId")  # 可选
    question: str
    top_k: int = Field(default=5, alias="topK")
    stream: bool = False

class SourceInfo(BaseSchema):
    doc_id: str = Field(..., alias="docId")
    chunk_id: str = Field(..., alias="chunkId")
    content: str
    score: float

class ChatResponse(BaseSchema):
    user_id: str = Field(..., alias="userId")
    knowledge_id: str | None = Field(default=None, alias="knowledgeId")
    doc_id: str | None = Field(default=None, alias="docId")
    question: str
    answer: str
    sources: list[SourceInfo]  # 引用来源
```

#### 3.1.5 Mock 数据示例

Schema 定义完成后，立即创建 Mock 数据，前端可直接开发：

```python
# tests/mock_data.py - Mock 数据定义

MOCK_DOCUMENTS = {
    "create": {
        "userId": "user_001",
        "docId": "doc_a1b2c3d4",
        "knowledgeId": "kb_001",
        "filename": "产品说明书.pdf",
        "chunkCount": 15,
        "status": "completed",
        "createdAt": "2025-01-08T10:30:00Z"
    },
    "query": {
        "userId": "user_001",
        "knowledgeId": "kb_001",
        "total": 3,
        "page": 1,
        "pageSize": 20,
        "documents": [
            {"docId": "doc_001", "filename": "产品说明书.pdf", "chunkCount": 15, "status": "completed"},
            {"docId": "doc_002", "filename": "用户手册.docx", "chunkCount": 8, "status": "completed"},
            {"docId": "doc_003", "filename": "技术规格.pdf", "chunkCount": 22, "status": "processing"}
        ]
    }
}

MOCK_CHAT = {
    # 场景 1: 指定知识库搜索
    "response_with_kb": {
        "userId": "user_001",
        "knowledgeId": "kb_001",
        "docId": None,
        "question": "产品的保修期是多久？",
        "answer": "根据产品说明书，标准保修期为 2 年，自购买之日起计算。",
        "sources": [
            {"docId": "doc_001", "chunkId": "doc_001_chunk_3", "content": "保修期限：2年...", "score": 0.92}
        ]
    },
    # 场景 2: 全局搜索（只传 userId）
    "response_global": {
        "userId": "user_001",
        "knowledgeId": None,
        "docId": None,
        "question": "什么是机器学习？",
        "answer": "机器学习是人工智能的一个分支...",
        "sources": [...]
    }
}
```

```python
# 快速启动 Mock Server (使用 FastAPI)
# app/routers/mock.py

from fastapi import APIRouter
from tests.mock_data import MOCK_DOCUMENTS, MOCK_CHAT

router = APIRouter(prefix="/mock", tags=["mock"])

@router.post("/knowledge/documents/create")
async def mock_create_document():
    return MOCK_DOCUMENTS["create"]

@router.post("/knowledge/documents/query")
async def mock_query_documents():
    return MOCK_DOCUMENTS["query"]

@router.post("/knowledge/chat")
async def mock_chat():
    return MOCK_CHAT["response"]
```

**前端开发流程：**
```
1. 启动 Mock Server: uvicorn app.main:app --reload
2. 前端配置 baseURL: http://localhost:8000/mock
3. 开发完成后切换: http://localhost:8000  (真实接口)
```

---

### 3.2 核心模块开发

#### 3.2.1 开发顺序（按数据流）

```
Step 1: parsers    →  Step 2: embedding  →  Step 3: milvus
   │                      │                     │
   │ 文档解析              │ 文本向量化           │ 向量存储
   ▼                      ▼                     ▼
Step 4: rerank     →  Step 5: retrieval  →  Step 6: llm     →  Step 7: agent
   │                      │                     │                   │
   │ 重排序                │ 检索召回             │ 回答生成           │ 流程编排
```

#### 3.2.2 各模块详细说明

| 模块 | 文件 | 核心类 | 说明 |
|------|------|--------|------|
| **parsers** | `mineru.py` | `MinerUParser` | MinerU API 客户端，PDF→Markdown |
| | `vision.py` | `VisionParser` | Qwen3-VL 图表/OCR 理解 |
| | `chunker.py` | `TextChunker` | 递归分块，支持 Markdown |
| **embedding** | `client.py` | `EmbeddingClient` | OpenAI 兼容 /v1/embeddings |
| **milvus** | `client.py` | `MilvusClient` | pymilvus MilvusClient(uri=...) 封装，支持同步/异步 |
| | `collection.py` | `CollectionManager` | Collection CRUD + 向量检索 |
| **rerank** | `client.py` | `RerankClient` | Cohere 兼容 /v1/rerank |
| **retrieval** | `text.py` | `TextRetriever` | BM25 关键词召回 |
| | `vector.py` | `VectorRetriever` | 向量语义召回 |
| | `hybrid.py` | `HybridRetriever` | 双路召回 + Rerank |
| **llm** | `client.py` | `LLMClient` | OpenAI SDK 调用 vLLM |
| | `prompt.py` | `PromptManager` | RAG Prompt 模板 |
| **agent** | `rag.py` | `RAGAgent` | 完整 RAG 流程编排 |

#### 3.2.3 核心代码示例

**文档上传流程：**
```python
# app/routers/document.py
@router.post("/create")
async def create_document(user_id, knowledge_id, file):
    # 1. MinerU 解析
    parse_result = parser.parse_bytes(content, filename)

    # 2. 文本分块
    chunks = chunker.chunk_text(parse_result.content)

    # 3. 批量向量化
    embeddings = embedding_client.embed_documents(texts)

    # 4. 存储到 Milvus
    collection_manager.insert_batch(doc_ids, chunk_ids, contents, embeddings, metadatas)
```

**知识问答流程：**
```python
# app/routers/chat.py
@router.post("/chat")
async def chat(request: ChatRequest):
    # 1. 向量召回
    query_embedding = embedding_client.embed(request.question)
    search_results = collection_manager.search(query_embedding, top_k=request.top_k * 3)

    # 2. Rerank 重排序
    rerank_result = rerank_client.rerank(request.question, documents, top_k=request.top_k)

    # 3. 构建 Prompt
    messages = prompt_manager.build_rag_messages(question, context)

    # 4. LLM 生成
    result = await llm_client.chat(messages)
```

#### 3.2.4 配置管理

```python
# configs/settings.py
class Settings(BaseSettings):
    # MinerU
    mineru_base_url: str = "http://localhost:8003"

    # vLLM + Qwen3-VL
    llm_base_url: str = "http://localhost:8000"
    llm_model: str = "Qwen/Qwen3-VL-8B"

    # Embedding（1024 维向量）
    embedding_base_url: str = "http://localhost:8001"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Rerank
    rerank_base_url: str = "http://localhost:8002"
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"

    # Milvus（使用 uri 格式连接）
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"

    class Config:
        env_file = ".env"
```

#### 3.2.5 Milvus 使用说明

**客户端连接方式（pymilvus 2.x MilvusClient API）：**
```python
from core.milvus import MilvusClient, get_milvus_client

# 方式 1: 使用全局单例
client = get_milvus_client()

# 方式 2: 自定义配置
client = MilvusClient(uri="http://192.168.25.177:19530")

# 列出所有 Collection
collections = client.list_collections()
```

**Collection 管理：**
```python
from core.milvus import CollectionManager

# 创建 Collection（默认维度 1024，度量类型 L2）
manager = CollectionManager(
    collection_name="my_knowledge_base",
    dimension=1024,
    metric_type="L2",  # Milvus 2.2.x 仅支持 L2 和 IP
)
manager.create()

# 插入数据（支持字典或列表格式）
manager.insert([{
    "doc_id": "doc_001",
    "chunk_id": "chunk_001",
    "content": "文档内容...",
    "embedding": [0.1] * 1024,  # 或使用 "vector" 字段名
    "metadata": {"source": "manual"},
}])

# 向量搜索
results = manager.search(
    query_embedding=[0.1] * 1024,
    top_k=10,
    filter_expr='doc_id == "doc_001"',
)
```

**异步客户端（用于高并发场景）：**
```python
from core.milvus import AsyncMilvusClient, get_async_milvus_client

async with AsyncMilvusClient() as client:
    results = await client.search(
        collection_name="my_collection",
        query_vectors=[[0.1] * 1024],
        limit=10,
    )
```

---

### 3.3 单元测试

#### 3.3.1 测试框架与依赖

```bash
# requirements.txt 中添加测试依赖
pytest>=8.0.0           # 测试框架
pytest-asyncio>=0.23.0  # 异步测试支持
pytest-cov>=4.1.0       # 覆盖率报告
pytest-mock>=3.12.0     # Mock 支持
respx>=0.20.0           # httpx 请求 Mock
```

#### 3.3.2 测试目录结构

```
tests/
├── conftest.py                    # 全局 fixtures
├── unit/                          # 单元测试
│   ├── parsers/
│   │   ├── test_chunker.py       # TextChunker 测试
│   │   └── test_mineru.py        # MinerU 解析器测试
│   ├── embedding/
│   │   └── test_client.py        # Embedding 客户端测试
│   ├── milvus/
│   │   ├── test_client.py        # Milvus 连接测试
│   │   └── test_collection.py    # Collection 操作测试
│   ├── rerank/
│   │   └── test_client.py        # Rerank 客户端测试
│   ├── retrieval/
│   │   ├── test_text.py          # BM25 文本检索测试
│   │   ├── test_vector.py        # 向量检索测试
│   │   └── test_hybrid.py        # 混合检索测试
│   ├── llm/
│   │   ├── test_client.py        # LLM 客户端测试
│   │   └── test_prompt.py        # Prompt 模板测试
│   └── agent/
│       └── test_rag.py           # RAG Agent 测试
├── api/                           # API 路由测试
│   ├── test_document_router.py
│   └── test_chat_router.py
└── integration/                   # 集成测试
    └── ...
```

#### 3.3.3 Mock 策略

| 外部服务 | Mock 方式 | 说明 |
|----------|-----------|------|
| MinerU API | `respx` | Mock HTTP POST /api/parse |
| vLLM Embedding | `respx` | Mock /v1/embeddings |
| vLLM Rerank | `respx` | Mock /v1/rerank |
| vLLM LLM | `unittest.mock` | Mock OpenAI SDK |
| Milvus | `pytest-mock` | Mock pymilvus 模块 |

#### 3.3.4 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio pytest-cov pytest-mock respx

# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行并生成覆盖率报告
pytest tests/ --cov=core --cov=app --cov-report=html

# 运行特定模块测试
pytest tests/unit/parsers/ -v

# 运行 API 测试
pytest tests/api/ -v
```

#### 3.3.5 测试用例示例

**纯逻辑模块测试（无需 Mock）：**
```python
# tests/unit/parsers/test_chunker.py
def test_chunk_text_basic():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_text("这是一段很长的测试文本...")

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
```

**HTTP 客户端测试（使用 respx Mock）：**
```python
# tests/unit/embedding/test_client.py
@respx.mock
def test_embed_single_text():
    mock_response = {
        "data": [{"embedding": [0.1] * 4096, "index": 0}],
        "usage": {"prompt_tokens": 5},
    }
    respx.post("http://localhost:8001/v1/embeddings").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    client = EmbeddingClient(base_url="http://localhost:8001")
    embedding = client.embed("测试文本")

    assert len(embedding) == 4096
```

**异步测试（使用 AsyncMock）：**
```python
# tests/unit/agent/test_rag.py
@pytest.mark.asyncio
async def test_rag_query():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [mock_search_result]

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=mock_chat_result)

    agent = RAGAgent(retriever=mock_retriever, llm_client=mock_llm)
    result = await agent.query("测试问题")

    assert result.answer == "AI 生成的回答"
```

#### 3.3.6 测试覆盖范围

| 模块 | 测试文件 | 用例数量 | 覆盖功能 |
|------|----------|----------|----------|
| parsers | test_chunker.py | 12 | 分块、重叠、元数据 |
| parsers | test_mineru.py | 14 | 解析、提取、健康检查 |
| embedding | test_client.py | 14 | 单条/批量向量化、分批处理 |
| milvus | test_client.py | 12 | 连接管理、Collection 操作 |
| milvus | test_collection.py | 16 | CRUD、搜索、删除 |
| rerank | test_client.py | 15 | 重排序、带元数据重排序 |
| retrieval | test_text.py | 14 | BM25、关键词搜索 |
| retrieval | test_vector.py | 10 | 向量搜索、过滤 |
| retrieval | test_hybrid.py | 12 | 双路召回、Rerank |
| llm | test_client.py | 13 | 对话、流式输出 |
| llm | test_prompt.py | 11 | 模板管理、RAG 消息 |
| agent | test_rag.py | 12 | RAG 流程、流式查询 |
| API | test_*.py | 17 | 路由端点测试 |

**目标覆盖率：核心模块 ≥ 80%**

---

### 3.4 集成测试

集成测试使用真实服务验证组件间协作，位于 `tests/integration/` 目录。

#### 测试文件

| 文件 | 说明 | 测试用例数 |
|------|------|-----------|
| conftest.py | 集成测试配置、服务检测、Fixtures | - |
| test_e2e_pipeline.py | 端到端流程测试 | 10+ |
| test_performance.py | 性能压测 | 12+ |

#### 测试分类

**端到端流程测试（E2E）**
- 文档分块与向量化入库
- 基础向量检索与过滤检索
- 混合检索（双路召回 + Rerank）
- 完整 RAG 问答流程
- 流式问答测试
- 多文档综合问答

**性能压测**
- Embedding 吞吐量与延迟
- Milvus 插入/搜索性能
- Rerank 延迟
- LLM 推理延迟与首 Token 延迟（TTFT）
- 并发能力测试
- 可扩展性测试（延迟 vs 数据量）

#### 运行方式

```bash
# 运行所有集成测试
pytest tests/integration/ -v -m integration

# 仅运行端到端测试
pytest tests/integration/test_e2e_pipeline.py -v

# 仅运行性能测试（较慢）
pytest tests/integration/test_performance.py -v -m "integration and slow"

# 跳过慢速测试
pytest tests/integration/ -v -m "integration and not slow"
```

#### 服务依赖

集成测试需要以下服务运行：

| 服务 | 端口 | 用途 |
|------|------|------|
| Milvus | 19530 | 向量数据库 |
| Embedding API | 8001 | 文本向量化 |
| Rerank API | 8002 | 重排序（可选） |
| LLM (Ollama) | 11434 | 大语言模型 |
| MinerU | 8003 | 文档解析（可选） |

> 未启动的服务会自动跳过相关测试，不影响其他测试执行。

---

### 3.5 部署上线

#### 构建镜像

```bash
docker build -t knowledge-db .
```

#### 运行容器

```bash
# 基础运行
docker run -d -p 8000:8000 --name knowledge-db knowledge-db

# 指定外部服务地址
docker run -d -p 8000:8000 --name knowledge-db \
  -e MILVUS_HOST=192.168.1.100 \
  -e LLM_BASE_URL=http://gpu-server:8000 \
  -e EMBEDDING_BASE_URL=http://gpu-server:8001 \
  -e RERANK_BASE_URL=http://gpu-server:8002 \
  -e MINERU_BASE_URL=http://gpu-server:8003 \
  knowledge-db
```

#### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MILVUS_HOST | localhost | Milvus 地址 |
| MILVUS_PORT | 19530 | Milvus 端口 |
| LLM_BASE_URL | http://localhost:8000 | LLM 服务地址 |
| EMBEDDING_BASE_URL | http://localhost:8001 | Embedding 服务地址 |
| RERANK_BASE_URL | http://localhost:8002 | Rerank 服务地址 |
| MINERU_BASE_URL | http://localhost:8003 | MinerU 服务地址 |

> 完整配置参考 `.env.example`

#### 健康检查

```bash
curl http://localhost:8000/health
```

#### 常用命令

```bash
# 查看日志
docker logs -f knowledge-db

# 进入容器
docker exec -it knowledge-db bash

# 停止服务
docker stop knowledge-db

# 删除容器
docker rm knowledge-db
```

---

### 3.6 Web Demo

提供基于 Gradio 的可视化演示界面，支持文档上传、知识问答和文档管理。

#### 启动方式

```bash
# 方式 1: 直接运行
python -m app.demo

# 方式 2: 使用 gradio 命令
gradio app/demo.py

# 访问地址
# http://localhost:7860
```

#### 功能模块

| 模块 | 功能说明 |
|------|----------|
| 💬 知识问答 | 与知识库对话，支持层级过滤（用户/知识库/文档） |
| 📤 文档上传 | 上传 PDF/Word/TXT/Markdown 文档到知识库 |
| 📁 文档管理 | 查询和删除知识库中的文档 |
| 📖 使用说明 | 快速入门指南和常见问题 |

#### 界面预览

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 企业知识库问答系统                                        │
├─────────────────────────────────────────────────────────────┤
│  [💬 知识问答] [📤 文档上传] [📁 文档管理] [📖 使用说明]        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔧 检索设置          │  对话区域                            │
│  ┌─────────────┐      │  ┌─────────────────────────────┐    │
│  │ 用户ID      │      │  │ 用户: 什么是机器学习？        │    │
│  │ 知识库ID    │      │  │                             │    │
│  │ 文档ID      │      │  │ AI: 机器学习是人工智能的      │    │
│  │ Top-K: 5    │      │  │     一个分支...              │    │
│  └─────────────┘      │  └─────────────────────────────┘    │
│                       │  [输入问题...]          [发送]       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 配置修改

如需修改 API 地址，编辑 `app/demo.py` 中的配置：

```python
API_BASE_URL = "http://localhost:8000"  # 改为实际 API 地址
```
