import pymupdf as fitz
import os
import re

def parse_pdf(file_path: str) -> list[dict]:
    """
    解析PDF财报，返回按页组织的文本列表
    每个元素包含：页码、文本内容、来源文件名
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    file_name = os.path.basename(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # 清洗文本
        text = clean_text(text)

        # 过滤空页或内容太少的页（比如封面图片页）
        if len(text.strip()) < 50:
            continue

        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "source": file_name,
            "total_pages": len(doc)
        })

    doc.close()
    print(f"✅ 解析完成：{file_name}，有效页数 {len(pages)} 页")
    return pages


def clean_text(text: str) -> str:
    """清洗财报文本中的噪音"""
    # 删除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 删除页眉页脚常见的页码数字独行
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    # 删除前后空白
    text = text.strip()
    return text


# 测试用
if __name__ == "__main__":
    # 把你的财报PDF放在项目根目录，改成你的文件名
    pages = parse_pdf("test_report.pdf")
    print(f"共解析 {len(pages)} 页")
    print("第一页内容预览：")
    print(pages[0]["text"][:300])
