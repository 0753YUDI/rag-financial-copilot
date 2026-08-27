# rag-financial-copilot

面向券商研究员与金融分析师的"可溯源、可复核"智能财报解读工具，
解决大模型在金融分析中的幻觉问题，实现数值高精度抽取与结论原文回溯。

## 背景与定位
传统LLM在财报和研报分析中容易产生幻觉、给出无法验证的结论。本项目要求
每一项数据与判断都提供原文引用，可回溯至财报具体章节，并保留人工
复审环节，可服务于券商研究员的实际投研流程。

## 系统架构
[PDF财报] → 解析(pdf_parser) → 切块(chunker) → 向量化(embedder)
    → 向量数据库 → 检索(retriever) → 多Agent协同(agents) → 报告输出

- **RAG流水线**：基于 LangChain 构建，financial report → chunk → embedding → vector store
- **多Agent协同**：财务提取 Agent / 风险识别 Agent / 总结生成 Agent 分工协作
- **规则驱动预警**：如营收下滑 >20% 自动触发风险标记，辅助快速定位异常

## 效果验证
| 指标 | 数值 |
|---|---|
| 信源忠实度（Faithfulness） | 93.8% |
| 财务数值精准度 | 96.2% |
| 跨章节多跳推理合格率 | 83.3% |

（基于测试研报集 + 人工标注验证）
## 技术栈

### 数据存储
- **数据库**：PostgreSQL（Docker容器部署）+ pgvector 扩展，用于向量存储与检索

### 模型服务
- **Embedding（向量化）**：智谱AI Embedding API
  - `https://open.bigmodel.cn/api/paas/v4`
- **LLM（财务提取 / 风险识别 / 总结生成 Agent）**：DeepSeek API

### 核心框架
- LangChain（RAG流水线编排）
- PyMuPDF / pdfplumber（PDF财报解析，具体看你 `pdf_parser.py` 用的库）

### 环境依赖
详见 `requirements.txt`
