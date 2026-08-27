import streamlit as st
import sys, os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest.pdf_parser import parse_pdf
from ingest.chunker import chunk_pages
from ingest.embedder import init_db, embed_and_store, clear_source
from retriever.retriever import retrieve_chunks, answer_question
from agents.agents import run_agents

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智能投研助手",
    page_icon="📈",
    layout="wide"
)

st.title("📈 智能投研助手")
st.caption("基于 RAG + Agent 的财报自动解读系统")

# ============================================================
# 侧边栏：上传财报
# ============================================================
with st.sidebar:
    st.header("📁 上传财报")
    uploaded_file = st.file_uploader("上传PDF年报", type=["pdf"])

    if uploaded_file:
        save_path = f"./uploads/{uploaded_file.name}"
        os.makedirs("./uploads", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🚀 开始解析财报", type="primary"):
            with st.spinner("正在解析财报..."):
                init_db()
                clear_source(uploaded_file.name)
                pages = parse_pdf(save_path)
                chunks = chunk_pages(pages)

            with st.spinner(f"正在向量化 {len(chunks)} 个chunk..."):
                embed_and_store(chunks)

            st.success(f"✅ 解析完成！共 {len(chunks)} 个chunk")
            st.session_state["current_source"] = uploaded_file.name

    st.divider()

    if "current_source" in st.session_state:
        st.info(f"📄 当前财报：\n{st.session_state['current_source']}")
    else:
        st.warning("请先上传财报")

# ============================================================
# 主区域：两个Tab
# ============================================================
tab1, tab2 = st.tabs(["💬 智能问答", "🤖 深度研报分析"])

# ---------- Tab1：智能问答 ----------
with tab1:
    st.subheader("💬 智能问答")
    st.caption("基于财报原文回答，附引用来源")

    query = st.text_input(
        "请输入您的问题",
        placeholder="例如：公司2023年营业收入是多少？同比增长了多少？"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("提问", type="primary")

    if ask_btn and query:
        if "current_source" not in st.session_state:
            st.error("请先在左侧上传并解析财报")
        else:
            with st.spinner("正在检索并生成回答..."):
                result = answer_question(query, source=st.session_state["current_source"])

            # AI回答
            st.markdown("### 🤖 AI 回答")
            st.markdown(result["answer"])

            # 人工复核区
            st.divider()
            st.markdown("### 👤 人工复核")
            review = st.text_area(
                "研究员补充意见（可选）",
                placeholder="如有异议或补充，请在此填写...",
                height=100
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("✅ 确认无误", type="primary")
            with col_b:
                st.button("⚠️ 标记存疑")

            # 引用来源
            with st.expander("📎 查看引用来源"):
                for i, s in enumerate(result["sources"]):
                    st.markdown(f"**来源{i+1}** · 第{s['page']}页 · 相似度 `{s['similarity']}`")
                    st.text(s["text"])
                    st.divider()

# ---------- Tab2：深度研报分析 ----------
with tab2:
    st.subheader("🤖 多Agent深度分析")
    st.caption("自动提取财务指标、识别风险、生成研报摘要")

    focus = st.text_input(
        "分析重点（可选）",
        placeholder="例如：重点关注盈利能力和债务风险",
        value="公司整体经营情况、盈利能力和主要风险"
    )

    if st.button("🚀 启动深度分析", type="primary"):
        if "current_source" not in st.session_state:
            st.error("请先在左侧上传并解析财报")
        else:
            chunks = retrieve_chunks(focus, top_k=8, source=st.session_state["current_source"])

            with st.spinner("Agent1 正在提取财务指标..."):
                from agents.agents import agent_extract_metrics
                metrics = agent_extract_metrics(chunks)

            with st.spinner("Agent2 正在识别风险信号..."):
                from agents.agents import agent_identify_risks
                risks = agent_identify_risks(metrics, chunks)

            with st.spinner("Agent3 正在生成研报摘要..."):
                from agents.agents import agent_generate_summary
                summary = agent_generate_summary(metrics, risks, focus, chunks)

            # 展示财务指标
            st.markdown("### 📊 财务指标")
            if "error" not in metrics:
                cols = st.columns(3)
                items = [
                    ("营业收入", metrics.get("营业收入")),
                    ("营收增长率", metrics.get("营业收入增长率")),
                    ("净利润", metrics.get("净利润")),
                    ("净利润增长率", metrics.get("净利润增长率")),
                    ("毛利率", metrics.get("毛利率")),
                    ("资产负债率", metrics.get("资产负债率")),
                ]
                for i, (label, value) in enumerate(items):
                    with cols[i % 3]:
                        st.metric(label=label, value=value or "未找到")
            else:
                st.warning("指标提取失败，请查看原始输出")
                st.text(metrics.get("raw", ""))

            st.divider()

            # 展示风险
            st.markdown("### ⚠️ 风险识别")
            if risks["rule_risks"]:
                st.error("**规则预警触发：**\n" + "\n".join(risks["rule_risks"]))
            else:
                st.success("✅ 无规则预警触发")

            st.markdown(risks["ai_risks"])

            st.divider()

            # 展示研报摘要
            st.markdown("### 📝 研报摘要")
            st.markdown(summary)

            # 人工复核
            st.divider()
            st.markdown("### 👤 人工复核")
            review = st.text_area("研究员审核意见", placeholder="请填写审核意见...", height=120)
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("✅ 审核通过", type="primary", key="approve")
            with col_b:
                st.button("🔄 需要修改", key="revise")