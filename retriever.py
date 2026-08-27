import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import sys, os, json, re
from dotenv import load_dotenv

# jieba 用于中文关键词抽取（关键词召回路径需要）
# 如果没装：pip install jieba
import jieba.analyse

# 明确指定.env的路径（在项目根目录）
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# 智谱embedding（问题也要用同一个模型向量化）
embed_client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# DeepSeek负责生成回答 + 重排序
chat_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


def get_query_embedding(text: str) -> list:
    """把用户问题向量化"""
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding


def extract_keywords(query: str, top_k: int = 5) -> list:
    """用jieba从问题中抽取关键词（TF-IDF），用于关键词召回"""
    keywords = jieba.analyse.extract_tags(query, topK=top_k)
    # 兜底：如果抽不出关键词（比如问题很短），直接用分词结果
    if not keywords:
        keywords = [w for w in jieba.lcut(query) if len(w.strip()) > 1]
    return keywords


def retrieve_by_vector(query_embedding: list, top_k: int, source: str = None) -> list:
    """向量召回路径（原有逻辑）"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    if source:
        cur.execute("""
            SELECT text, source, page_num,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM report_chunks
            WHERE source = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, source, query_embedding, top_k))
    else:
        cur.execute("""
            SELECT text, source, page_num,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM report_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    chunks = []
    for text, src, page_num, similarity in results:
        chunks.append({
            "text": text,
            "source": src,
            "page_num": page_num,
            "vector_similarity": round(similarity, 4),
            "match_type": "vector"
        })
    return chunks


def retrieve_by_keyword(query: str, top_k: int, source: str = None) -> list:
    """
    关键词召回路径。
    用jieba抽取问题中的关键词，对report_chunks做ILIKE匹配，
    按命中的关键词数量打分排序。
    注意：这是基于ILIKE的简单实现，没有依赖数据库端的中文分词扩展
    （如zhparser/pg_jieba）。如果数据量大、对召回质量要求更高，
    建议后续换成PostgreSQL全文检索 + 中文分词扩展。
    """
    keywords = extract_keywords(query, top_k=5)
    if not keywords:
        return []

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 动态拼出 SUM(CASE WHEN text ILIKE %s THEN 1 ELSE 0 END) 打分逻辑
    score_terms = " + ".join(["CASE WHEN text ILIKE %s THEN 1 ELSE 0 END"] * len(keywords))
    like_params = [f"%{kw}%" for kw in keywords]

    base_sql = f"""
        SELECT text, source, page_num, ({score_terms}) AS match_score
        FROM report_chunks
    """
    params = list(like_params)

    where_clauses = []
    if source:
        where_clauses.append("source = %s")
        params.append(source)

    sql = base_sql
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += f" HAVING ({score_terms}) > 0 ORDER BY match_score DESC LIMIT %s"
    params += like_params + [top_k]

    # HAVING 在没有 GROUP BY 时部分数据库不支持，这里用子查询更稳妥
    sql = f"""
        SELECT text, source, page_num, match_score FROM (
            SELECT text, source, page_num, ({score_terms}) AS match_score
            FROM report_chunks
            {"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
        ) t
        WHERE match_score > 0
        ORDER BY match_score DESC
        LIMIT %s
    """
    params = list(like_params) + ([source] if source else []) + [top_k]

    cur.execute(sql, params)
    results = cur.fetchall()
    cur.close()
    conn.close()

    chunks = []
    for text, src, page_num, match_score in results:
        chunks.append({
            "text": text,
            "source": src,
            "page_num": page_num,
            "keyword_score": match_score,
            "match_type": "keyword",
            "matched_keywords": keywords
        })
    return chunks


def merge_candidates(vector_chunks: list, keyword_chunks: list) -> list:
    """
    合并两路召回结果，按(source, page_num, text)去重。
    如果一个chunk同时被两路命中，match_type标记为"both"，
    并保留两边的分数信息。
    """
    merged = {}
    for c in vector_chunks:
        key = (c["source"], c["page_num"], c["text"])
        merged[key] = dict(c)

    for c in keyword_chunks:
        key = (c["source"], c["page_num"], c["text"])
        if key in merged:
            merged[key]["match_type"] = "both"
            merged[key]["keyword_score"] = c["keyword_score"]
            merged[key]["matched_keywords"] = c["matched_keywords"]
        else:
            merged[key] = dict(c)

    return list(merged.values())


def rerank_with_deepseek(query: str, candidates: list, top_k: int = 5) -> list:
    """
    用DeepSeek对候选chunk做相关性重排序，从最多10个候选中选出top_k个。
    模型只需要返回按相关性从高到低排列的候选编号（JSON数组），
    不重新生成文本，避免幻觉。
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    numbered_context = "\n\n---\n\n".join([
        f"[{i}] （第{c['page_num']}页，来源：{c['source']}）\n{c['text']}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""你是一个专业的财务文档检索重排序助手。下面是通过向量检索和关键词检索召回的多个财报片段候选（编号从0开始），请根据它们与用户问题的相关程度进行排序。

【用户问题】
{query}

【候选片段】
{numbered_context}

【要求】
- 只输出一个JSON数组，包含所有候选的编号，按相关性从高到低排列
- 例如：[3, 0, 5, 1, 2, 4]
- 不要输出任何解释文字，不要用markdown代码块包裹，只输出JSON数组本身
"""

    response = chat_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    # 兜底清理，防止模型仍然带了markdown围栏
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        order = json.loads(raw)
        order = [i for i in order if isinstance(i, int) and 0 <= i < len(candidates)]
    except (json.JSONDecodeError, TypeError):
        # 解析失败就直接按原顺序（向量在前，关键词在后）截断，保证流程不中断
        order = list(range(len(candidates)))

    # 去重并补全（防止模型漏掉某些编号）
    seen = set()
    final_order = []
    for i in order:
        if i not in seen:
            final_order.append(i)
            seen.add(i)
    for i in range(len(candidates)):
        if i not in seen:
            final_order.append(i)
            seen.add(i)

    ranked = [candidates[i] for i in final_order[:top_k]]
    return ranked


def retrieve_chunks(query: str, top_k: int = 5, source: str = None) -> list:
    """
    双路召回 + DeepSeek重排序：
    1. 向量检索召回top_k个
    2. 关键词检索召回top_k个
    3. 合并去重（最多10个左右）
    4. 用DeepSeek对合并结果重排序，返回最相关的top_k个
    """
    query_embedding = get_query_embedding(query)

    vector_chunks = retrieve_by_vector(query_embedding, top_k=top_k, source=source)
    keyword_chunks = retrieve_by_keyword(query, top_k=top_k, source=source)

    candidates = merge_candidates(vector_chunks, keyword_chunks)
    reranked = rerank_with_deepseek(query, candidates, top_k=top_k)

    # 统一输出格式，补上相似度字段（没有的用None占位，方便展示）
    chunks = []
    for c in reranked:
        chunks.append({
            "text": c["text"],
            "source": c["source"],
            "page_num": c["page_num"],
            "match_type": c.get("match_type"),
            "vector_similarity": c.get("vector_similarity"),
            "keyword_score": c.get("keyword_score"),
        })
    return chunks


def answer_question(query: str, source: str = None) -> dict:
    """
    完整的RAG问答：双路召回 + 重排序 + 生成
    返回：回答、引用来源
    """
    # 第一步：双路检索 + 重排序，得到最相关的chunk
    chunks = retrieve_chunks(query, top_k=5, source=source)

    if not chunks:
        return {
            "answer": "未找到相关内容，请确认财报已上传。",
            "sources": []
        }

    # 第二步：拼成上下文
    context = "\n\n---\n\n".join([
        f"【第{c['page_num']}页】{c['text']}" for c in chunks
    ])

    # 第三步：调用DeepSeek生成回答
    prompt = f"""你是一位专业的财务分析师。请根据以下财报原文回答问题。
只根据提供的原文回答，如果原文中没有相关信息，请明确说明"原文中未找到相关信息"，不要编造数据。

【财报原文】
{context}

【问题】
{query}

【要求】
- 回答要具体，包含具体数字
- 指出信息来自第几页
- 语言简洁专业
"""

    response = chat_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "page": c["page_num"],
                "source": c["source"],
                "match_type": c["match_type"],
                "vector_similarity": c["vector_similarity"],
                "keyword_score": c["keyword_score"],
                "text": c["text"][:100]
            }
            for c in chunks
        ]
    }


# 测试用
if __name__ == "__main__":
    question = "公司的营业收入是多少？"
    print(f"问题：{question}\n")
    result = answer_question(question)
    print(f"回答：\n{result['answer']}\n")
    print(f"引用来源：")
    for s in result["sources"]:
        print(f"  【{s['source']}】第{s['page']}页 [{s['match_type']}]（向量相似度{s['vector_similarity']}, 关键词分{s['keyword_score']}）：{s['text']}...")