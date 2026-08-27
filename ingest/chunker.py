from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    把解析好的页面列表切成小chunk
    保留每个chunk的元数据（来源、页码）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        # 中文财报按段落、句子切，不按英文空格
    )

    all_chunks = []

    for page in pages:
        # 对每页文本切块
        chunks = splitter.split_text(page["text"])

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_index": i,
                    "total_pages": page["total_pages"],
                }
            })

    print(f"✅ 切块完成：共 {len(all_chunks)} 个chunk")
    return all_chunks


# 测试用
if __name__ == "__main__":
    from pdf_parser import parse_pdf
    pages = parse_pdf("../test_report.pdf")
    chunks = chunk_pages(pages)
    print(f"\n前3个chunk预览：")
    for c in chunks[:3]:
        print(f"--- 第{c['metadata']['page_num']}页，chunk{c['metadata']['chunk_index']} ---")
        print(c["text"][:150])
        print()
