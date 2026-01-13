# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

企业级 AI 知识库系统，基于 RAG（检索增强生成）架构，支持文档解析、向量检索、智能问答。所有服务自部署，确保数据安全。

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| LLM | vLLM + Qwen3-VL | 大语言模型推理 |
| Embedding | Qwen3-Embedding | 文本向量化|
| Rerank | Qwen3-Reranker | 重排序优化检索结果 |
| 向量数据库 | Milvus | 向量存储与 ANN 检索 |
| 文档解析 | MinerU | PDF/图片解析，支持 OCR |
| Web 框架 | FastAPI | 异步 API 服务 |
| Demo | Gradio | Web 演示界面 |

## 目录结构

```
├── app/                    # FastAPI 应用层
│   ├── main.py            # 应用入口
│   ├── demo.py            # Gradio 演示
│   ├── routers/           # API 路由
│   │   ├── chat.py        # 对话接口
│   │   └── document.py    # 文档管理接口
│   └── schemas/           # Pydantic 数据模型
├── core/                   # 核心业务逻辑
│   ├── agent/             # RAG Agent
│   ├── embedding/         # 向量化客户端
│   ├── llm/               # LLM 客户端
│   ├── milvus/            # Milvus 向量库操作
│   ├── parsers/           # 文档解析 (MinerU, Chunker)
│   ├── rerank/            # 重排序客户端
│   ├── retrieval/         # 检索策略 (向量/文本/混合)
│   └── utils/             # 工具函数
├── configs/               # 配置管理
│   └── settings.py        # Pydantic Settings
├── scripts/               # 运维脚本
├── tests/                 # 测试用例
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── .env.example           # 环境变量模板
└── requirements.txt       # Python 依赖
```

## 开发命令

```bash
# 环境设置
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 运行单元测试（不需要外部服务）
pytest tests/unit/ -v

# 启动 API 服务
uvicorn app.main:app --reload --port 8080

# 启动 Gradio Demo
python -m app.demo
```

## 环境变量

复制 `.env.example` 为 `.env`，配置以下服务地址：

- `LLM_BASE_URL` - vLLM 服务地址
- `EMBEDDING_BASE_URL` - Embedding 服务地址
- `RERANK_BASE_URL` - Rerank 服务地址
- `MINERU_BASE_URL` - MinerU 文档解析服务地址
- `MILVUS_HOST` - Milvus 向量数据库地址

## 开发规范

- Python 3.11+
- 异步优先，使用 `async/await`
- 类型注解，使用 Pydantic 做数据校验
- API 遵循 RESTful 设计
- 配置通过环境变量注入，禁止硬编码敏感信息

## 核心模块说明

### RAG Pipeline

```
文档 → MinerU解析 → Chunker分块 → Embedding向量化 → Milvus存储
                                                        ↓
用户查询 → Embedding → Milvus检索 → Rerank重排 → LLM生成回答
```

### 检索策略

- `VectorRetriever` - 纯向量语义检索
- `TextRetriever` - BM25 文本检索
- `HybridRetriever` - 混合检索 (向量 + 文本)
