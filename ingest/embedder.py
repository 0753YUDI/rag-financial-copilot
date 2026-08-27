import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG

# 智谱embedding
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding


def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS report_chunks (
            id SERIAL PRIMARY KEY,
            source TEXT,
            page_num INTEGER,
            chunk_index INTEGER,
            text TEXT,
            embedding vector(2048)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ 数据库初始化完成")


def embed_and_store(chunks: list):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    print(f"正在向量化 {len(chunks)} 个chunk...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])
        cur.execute("""
            INSERT INTO report_chunks
                (source, page_num, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            chunk["metadata"]["source"],
            chunk["metadata"]["page_num"],
            chunk["metadata"]["chunk_index"],
            chunk["text"],
            embedding
        ))
        if (i + 1) % 10 == 0:
            print(f"  进度：{i+1}/{len(chunks)}")
            conn.commit()
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ 成功存入 {len(chunks)} 个chunk")


def clear_source(source_name: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("DELETE FROM report_chunks WHERE source = %s", (source_name,))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ 已清除 {source_name} 的旧数据")
