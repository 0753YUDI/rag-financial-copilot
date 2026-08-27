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
- PyMuPDF / pdfplumber

### 环境依赖
详见 `requirements.txt`

### Demo演示
<img width="1008" height="505" alt="截屏2026-08-27 14 08 55" src="https://github.com/user-attachments/assets/08c80fa5-7cb8-46de-8497-672a35e5530c" />
<img width="1431" height="738" alt="截屏2026-08-27 14 11 16" src="https://github.com/user-attachments/assets/8abc75af-f41a-488b-afae-f4d73f46b7d0" />


### chunk匹配策略：关键词检索+向量匹配度检索双策略 再进行Rerank
<img width="1048" height="662" alt="截屏2026-08-27 14 40 05" src="https://github.com/user-attachments/assets/28865054-ae51-4104-9951-df060c048b86" />

### 深度分析与研报生成展示
<img width="1083" height="691" alt="截屏2026-08-27 14 30 49" src="https://github.com/user-attachments/assets/e11bde70-48b0-4e00-bcaa-134c27ce9e28" />

<img width="1083" height="711" alt="截屏2026-08-27 14 31 08" src="https://github.com/user-attachments/assets/3cfd1edf-a90a-4ea0-a5c1-14cd310c415a" />

<img width="1083" height="647" alt="截屏2026-08-27 14 31 17" src="https://github.com/user-attachments/assets/122080bb-d2aa-42d2-b44a-a561031ee9c7" />

<img width="1083" height="647" alt="截屏2026-08-27 14 31 29" src="https://github.com/user-attachments/assets/c3079941-8ada-4e65-a005-33619e043f10" />

<img width="1048" height="707" alt="截屏2026-08-27 14 39 56" src="https://github.com/user-attachments/assets/27876b65-6cf6-4699-9e59-27e7dfbf633a" />

