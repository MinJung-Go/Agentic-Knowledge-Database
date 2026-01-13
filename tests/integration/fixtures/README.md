# 集成测试 Fixtures

此目录存放集成测试所需的测试数据文件。

## 文件说明

| 文件 | 用途 |
|------|------|
| sample.txt | 纯文本测试文件 |
| sample.pdf | PDF 测试文件（需手动添加） |
| sample.docx | Word 测试文件（需手动添加） |

## 准备测试文件

### PDF 文件

由于 PDF 文件较大且为二进制格式，请手动准备测试 PDF：

```bash
# 方式 1：使用现有 PDF
cp /path/to/your/document.pdf tests/integration/fixtures/sample.pdf

# 方式 2：使用工具生成
# 可以使用 pandoc 将 markdown 转换为 PDF
pandoc sample.txt -o sample.pdf
```

### 测试数据要求

- 文件大小建议在 1MB 以内，避免测试耗时过长
- 文档内容应包含可识别的文本，便于验证解析结果
- 中英文混合内容有助于测试多语言支持
