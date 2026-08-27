import sys, os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


# ============================================================
# Agent 1：财务指标提取
# ============================================================
def agent_extract_metrics(chunks: list) -> dict:
    """从检索到的chunk里提取关键财务指标"""

    context = "\n\n".join([f"【第{c['page_num']}页】{c['text']}" for c in chunks])

    prompt = f"""你是专业的财务数据提取助手。请从以下财报原文中提取关键财务指标。

【财报原文】
{context}

请严格按照以下JSON格式输出，如果某项数据原文中没有则填null：
{{
    "营业收入": "具体数值和单位",
    "营业收入增长率": "百分比",
    "净利润": "具体数值和单位",
    "净利润增长率": "百分比",
    "毛利率": "百分比",
    "资产负债率": "百分比",
    "经营现金流": "具体数值和单位",
    "数据来源页码": [页码列表]
}}

只输出JSON，不要输出其他内容。"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    import json
    try:
        raw = response.choices[0].message.content
        raw = raw.replace("```json", "").replace("```", "").strip()
        metrics = json.loads(raw)
    except:
        metrics = {"error": "指标提取失败", "raw": response.choices[0].message.content}

    print("✅ Agent1 财务指标提取完成")
    return metrics


# ============================================================
# Agent 2：风险识别
# ============================================================
def agent_identify_risks(metrics: dict, chunks: list) -> dict:
    """基于财务指标识别风险信号"""

    # 基于规则的风险预警
    rule_risks = []

    def parse_percent(val):
        if val and val != "null" and val is not None:
            try:
                return float(str(val).replace("%", "").replace("％", "").strip())
            except:
                return None
        return None

    # 规则1：营收下滑超20%
    revenue_growth = parse_percent(metrics.get("营业收入增长率"))
    if revenue_growth is not None and revenue_growth < -20:
        rule_risks.append(f"⚠️ 营收大幅下滑：增长率{revenue_growth}%，超过-20%预警线")

    # 规则2：净利润下滑超30%
    profit_growth = parse_percent(metrics.get("净利润增长率"))
    if profit_growth is not None and profit_growth < -30:
        rule_risks.append(f"⚠️ 净利润大幅下滑：增长率{profit_growth}%，超过-30%预警线")

    # 规则3：毛利率低于20%
    gross_margin = parse_percent(metrics.get("毛利率"))
    if gross_margin is not None and gross_margin < 20:
        rule_risks.append(f"⚠️ 毛利率偏低：{gross_margin}%，低于20%警戒线")

    # 规则4：资产负债率超70%
    debt_ratio = parse_percent(metrics.get("资产负债率"))
    if debt_ratio is not None and debt_ratio > 70:
        rule_risks.append(f"⚠️ 资产负债率偏高：{debt_ratio}%，超过70%警戒线")

    # AI深度风险分析
    context = "\n\n".join([f"【第{c['page_num']}页】{c['text']}" for c in chunks])

    prompt = f"""你是专业的财务风险分析师。请根据以下财务指标和财报原文，识别潜在风险。

【已提取的财务指标】
{metrics}

【规则预警已触发】
{rule_risks if rule_risks else "无规则预警触发"}

【财报原文】
{context}

请识别以下维度的风险（如无明显风险请说明）：
1. 经营风险
2. 财务风险  
3. 行业/市场风险
4. 其他需关注事项

输出格式：每条风险一行，以"🔴高风险"、"🟡中风险"、"🟢低风险"开头标注风险等级。"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    ai_risks = response.choices[0].message.content

    print("✅ Agent2 风险识别完成")
    return {
        "rule_risks": rule_risks,
        "ai_risks": ai_risks,
        "risk_count": len(rule_risks)
    }


# ============================================================
# Agent 3：研报总结生成
# ============================================================
def agent_generate_summary(metrics: dict, risks: dict, query: str, chunks: list) -> str:
    """整合指标和风险，生成研报摘要"""

    context = "\n\n".join([f"【第{c['page_num']}页】{c['text']}" for c in chunks])

    prompt = f"""你是资深证券研究员，请根据以下信息生成一份专业的研报摘要。

【用户关注的问题】
{query}

【财务指标】
{metrics}

【风险信号】
规则预警：{risks['rule_risks']}
AI分析风险：{risks['ai_risks']}

【财报原文参考】
{context}

请生成一份200-300字的研报摘要，包含：
1. 公司经营概况
2. 核心财务数据点评
3. 主要风险提示
4. 综合评价

语言要专业、客观，数据要有出处。"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    summary = response.choices[0].message.content
    print("✅ Agent3 研报总结生成完成")
    return summary


# ============================================================
# 三个Agent串联调用
# ============================================================
def run_agents(chunks: list, query: str) -> dict:
    """
    完整的Agent流水线
    输入：检索到的chunks + 用户问题
    输出：指标、风险、研报摘要
    """
    print("\n🤖 启动多Agent分析流水线...")
    print("=" * 40)

    metrics = agent_extract_metrics(chunks)
    risks = agent_identify_risks(metrics, chunks)
    summary = agent_generate_summary(metrics, risks, query, chunks)

    print("=" * 40)
    print("✅ 多Agent分析完成\n")

    return {
        "metrics": metrics,
        "risks": risks,
        "summary": summary
    }


# 测试用
if __name__ == "__main__":
    # 先用retriever拿到chunks，再跑agents
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retriever"))
    from retriever import retrieve_chunks

    query = "公司整体经营情况如何？有哪些风险？"
    chunks = retrieve_chunks(query, top_k=5)

    result = run_agents(chunks, query)

    print("\n📊 财务指标：")
    print(result["metrics"])
    print("\n⚠️ 风险识别：")
    print(result["risks"])
    print("\n📝 研报摘要：")
    print(result["summary"])