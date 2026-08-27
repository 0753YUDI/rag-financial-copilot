import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_parser import parse_pdf
from chunker import chunk_pages
from embedder import init_db, embed_and_store, clear_source


def ingest_report(pdf_path: str, force_reload: bool = False):
    file_name = os.path.basename(pdf_path)
    print(f"\n{'='*50}")
    print(f"开始处理财报：{file_name}")
    print(f"{'='*50}")

    init_db()

    if force_reload:
        clear_source(file_name)

    print("\n[1/3] 解析PDF...")
    pages = parse_pdf(pdf_path)

    print("\n[2/3] 切块...")
    chunks = chunk_pages(pages)

    print("\n[3/3] 向量化并存入数据库...")
    embed_and_store(chunks)

    print(f"\n{'='*50}")
    print(f"✅ 财报处理完成！")
    print(f"   文件：{file_name}")
    print(f"   页数：{len(pages)}")
    print(f"   chunk数：{len(chunks)}")
    print(f"{'='*50}\n")
if __name__ == "__main__":
    ingest_report("test_report.pdf")
