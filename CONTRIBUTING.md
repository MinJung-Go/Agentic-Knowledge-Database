# 贡献指南

感谢你愿意改进 Agentic Knowledge Database。本项目同时包含检索、解析、模型服务适配和 API 层，提交时请明确改动边界以及依赖的外部服务。

## 开始之前

- Bug 和较大的功能建议请先创建 Issue，附上最小复现步骤、预期结果、实际结果和环境信息。
- 从默认分支创建短生命周期功能分支，保持一次 Pull Request 只解决一个问题。
- 不要提交 `.env`、真实文档、API Key、访问令牌、用户数据或包含敏感内容的日志。

## 本地环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

单元测试使用 mock，不要求所有外部服务在线。集成测试会按用例访问 Milvus、Embedding、Rerank、LLM/Ollama 或 MinerU；请只配置你有权限使用的测试服务与隔离数据。

可先检查当前服务可用性：

```bash
python scripts/check_services.py
```

## 测试

运行单元测试：

```bash
pytest tests/unit/ -v
```

运行完整测试目录：

```bash
pytest tests/ -v
```

运行需要外部服务的集成测试：

```bash
pytest tests/integration/ -v -m integration
```

运行慢速性能用例：

```bash
pytest tests/integration/test_performance.py -v -m "integration and slow"
```

性能结果依赖模型、硬件、数据规模和服务拓扑。提交结果时请同时记录环境与测试参数，不要把单次本地结果表述为通用性能承诺。

## 代码与文档要求

- 遵循现有模块结构、类型标注和异步接口风格。
- 修改配置项时同步更新 `.env.example`、`configs/settings.py` 和 README。
- 修改 API、数据结构、索引字段或权限过滤时，补充对应测试和迁移说明。
- 用户隔离相关代码必须保留 `userId` 边界，并覆盖跨用户访问的负向测试。
- 新增依赖时说明用途以及是否会引入外部网络、模型或系统组件。

## Pull Request 检查清单

- 说明改动原因、实现方式和兼容性影响。
- 关联对应 Issue。
- 列出实际运行的检查、结果以及跳过项的原因。
- 对接口、配置或操作流程的变化同步更新文档。
- 不混入无关格式化、生成文件或真实测试数据。

提交 Pull Request 即表示你同意你的贡献按本仓库的 [MIT License](LICENSE) 发布。
