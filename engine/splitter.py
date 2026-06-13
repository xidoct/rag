"""
智能分块器 — 按标题/章节/表格边界分块

策略:
1. 识别标题模式（中文数字、阿拉伯数字、关键词开头）
2. 表格内容独立成块
3. 普通文本按 CHUNK_SIZE 切分，保留重叠
4. 每块携带元数据（来源、章节、页码）
"""

import re
from dataclasses import dataclass

from engine.loader import Document
import config


# --- 标题识别模式 ---
HEADING_PATTERNS = [
    # 第X章、第一章 → 一级标题
    re.compile(r"^第[一二三四五六七八九十\d]+章\s*"),
    # 一、二、三、 → 一级标题
    re.compile(r"^[一二三四五六七八九十]、\s*"),
    # 1. 1.1 1.1.1 → 分级标题
    re.compile(r"^\d+(\.\d+)*\.?\s+"),
    # （一）（二） → 二级标题
    re.compile(r"^（[一二三四五六七八九十\d]+）\s*"),
    # 一、 二、 (顿号形式)
    re.compile(r"^[一二三四五六七八九十]、"),
    # 第一章 第一节
    re.compile(r"^第[一二三四五六七八九十\d]+节"),
]


def is_heading(line: str) -> bool:
    """判断一行文本是否为标题"""
    line = line.strip()
    if not line or len(line) > 60:
        return False
    for pattern in HEADING_PATTERNS:
        if pattern.match(line):
            return True
    return False


def split_documents(documents: list[Document]) -> list[Document]:
    """
    智能分块主入口
    对每个 Document 按策略分块，返回新的 Document 列表
    """
    chunks = []
    for doc in documents:
        if doc.is_table:
            # 表格独立成块
            chunks.extend(_split_table(doc))
        else:
            chunks.extend(_split_text(doc))
    return chunks


def _split_text(doc: Document) -> list[Document]:
    """按标题边界 + 长度限制分块普通文本"""
    lines = doc.content.split("\n")
    chunks = []
    current_chunk_lines: list[str] = []
    current_heading = doc.heading
    current_size = 0

    def flush_chunk():
        nonlocal current_chunk_lines, current_size
        if not current_chunk_lines:
            return
        text = "\n".join(current_chunk_lines).strip()
        if text:
            chunks.append(Document(
                content=text,
                source=doc.source,
                page=doc.page,
                heading=current_heading,
                metadata={**doc.metadata, "chunk_type": "text"}
            ))
        current_chunk_lines = []
        current_size = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_chunk_lines.append("")
            continue

        # 遇到标题 → 新分块开始
        if is_heading(stripped):
            flush_chunk()
            current_heading = stripped
            current_chunk_lines = [stripped]
            current_size = len(stripped)
            continue

        # 块大小超过阈值 → 切割
        if current_size + len(stripped) > config.CHUNK_SIZE and current_chunk_lines:
            flush_chunk()
            # 重叠：保留最后几句作为新块的上下文
            overlap_lines = _get_overlap_lines(current_chunk_lines)

        current_chunk_lines.append(stripped)
        current_size += len(stripped)

    flush_chunk()
    return chunks


def _split_table(doc: Document) -> list[Document]:
    """表格内容按大小分块，保持行完整"""
    lines = doc.content.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        if current_size + len(line) > config.TABLE_CHUNK_SIZE and current_lines:
            chunks.append(Document(
                content="\n".join(current_lines),
                source=doc.source,
                page=doc.page,
                heading=doc.heading,
                is_table=True,
                metadata={**doc.metadata, "chunk_type": "table"}
            ))
            # 表头保留作为上下文
            header_line = current_lines[0] if current_lines else line
            current_lines = [header_line, line]
            current_size = len(header_line) + len(line)
        else:
            current_lines.append(line)
            current_size += len(line)

    if current_lines:
        chunks.append(Document(
            content="\n".join(current_lines),
            source=doc.source,
            page=doc.page,
            heading=doc.heading,
            is_table=True,
            metadata={**doc.metadata, "chunk_type": "table"}
        ))

    return chunks


def _get_overlap_lines(lines: list[str], overlap_size: int = None) -> list[str]:
    """从行列表末尾取 N 个字符作为重叠"""
    if overlap_size is None:
        overlap_size = config.CHUNK_OVERLAP
    overlap = []
    total = 0
    for line in reversed(lines):
        if total >= overlap_size:
            break
        overlap.append(line)
        total += len(line)
    return list(reversed(overlap))
