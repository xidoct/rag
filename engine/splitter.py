"""
智能分块器 — 按标题/章节/表格边界分块

标题识别策略:
1. 字体/字号/位置 (PyMuPDF get_text("dict")) — 主要手段
   - 字号 > 正文字号 × 1.2 → 标题
   - 粗体 + 字号偏大 → 各级标题
   - 位置居左 + 字号最大 → 一级章节
2. 正则模式匹配 — 兜底 (OCR 页面无字体信息时)
3. 表格内容独立成块
4. 普通文本按 CHUNK_SIZE 切分，保留重叠
"""

import re
from statistics import median
from typing import Optional

from engine.loader import Document
import config


# --- 正则兜底 (OCR 页面无字体信息时用) ---
HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十\d]+章\s*"),
    re.compile(r"^第[一二三四五六七八九十\d]+节"),
    re.compile(r"^[一二三四五六七八九十]、\s*"),
    re.compile(r"^\d+(\.\d+)*\.?\s+"),
    re.compile(r"^（[一二三四五六七八九十\d]+）\s*"),
    re.compile(r"^[一二三四五六七八九十]、"),
]


def split_documents(documents: list[Document]) -> list[Document]:
    """智能分块主入口"""
    chunks = []
    for doc in documents:
        if doc.is_table:
            chunks.extend(_split_table(doc))
        else:
            chunks.extend(_split_text(doc))
    return chunks


# ---- 文本分块 ----

def _split_text(doc: Document) -> list[Document]:
    """按标题边界 + 长度限制分块，优先用字体信息判断标题"""
    lines = doc.content.split("\n")
    font_map = _build_font_map(doc)
    body_size = _get_body_size(font_map)

    chunks: list[Document] = []
    current_lines: list[str] = []
    current_heading = doc.heading
    current_size = 0

    def flush():
        nonlocal current_lines, current_size
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(Document(
                content=text,
                source=doc.source,
                page=doc.page,
                heading=current_heading,
                metadata={**doc.metadata, "chunk_type": "text"}
            ))
        current_lines = []
        current_size = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        # 判断是否为标题
        if _is_heading_smart(stripped, font_map, body_size):
            flush()
            current_heading = stripped
            current_lines = [stripped]
            current_size = len(stripped)
            continue

        # 超阈值切割
        if current_size + len(stripped) > config.CHUNK_SIZE and current_lines:
            flush()

        current_lines.append(stripped)
        current_size += len(stripped)

    flush()
    return chunks


# ---- 表格分块 ----

def _split_table(doc: Document) -> list[Document]:
    lines = doc.content.split("\n")
    chunks = []
    cur_lines, cur_size = [], 0

    for line in lines:
        if cur_size + len(line) > config.TABLE_CHUNK_SIZE and cur_lines:
            chunks.append(Document(
                content="\n".join(cur_lines),
                source=doc.source, page=doc.page,
                heading=doc.heading, is_table=True,
                metadata={**doc.metadata, "chunk_type": "table"}
            ))
            header = cur_lines[0] if cur_lines else line
            cur_lines, cur_size = [header, line], len(header) + len(line)
        else:
            cur_lines.append(line)
            cur_size += len(line)

    if cur_lines:
        chunks.append(Document(
            content="\n".join(cur_lines),
            source=doc.source, page=doc.page,
            heading=doc.heading, is_table=True,
            metadata={**doc.metadata, "chunk_type": "table"}
        ))
    return chunks


# ---- 智能标题识别 ----

def _build_font_map(doc: Document) -> dict[str, dict]:
    """
    从 Document metadata 构建字体查找表
    key = 文本行前 60 字符, value = {size, bold, y}
    """
    font_info = doc.metadata.get("font_info", [])
    if not font_info:
        return {}
    return {entry["text"][:60]: entry for entry in font_info}


def _get_body_size(font_map: dict[str, dict]) -> Optional[float]:
    """计算正文字号 (所有文本行字号的中位数)"""
    sizes = [v["size"] for v in font_map.values()]
    if not sizes:
        return None
    return median(sizes)


def _is_heading_smart(
    line: str,
    font_map: dict[str, dict],
    body_size: Optional[float],
) -> bool:
    """
    综合判断: 字体特征 > 正则模式

    标题特征:
    - 字号 >= 正文字号 × 1.2
    - 粗体 + 字号 > 正文
    - 短文本 (< 60 字符) + 有编号模式
    """
    line = line.strip()
    if not line or len(line) > 60:
        return False

    info = font_map.get(line[:60])

    # --- 有字体信息: 用视觉特征判断 ---
    if info and body_size:
        size = info["size"]
        bold = info.get("bold", False)

        # 字号明显大于正文
        if size >= body_size * 1.2:
            return True

        # 粗体 + 比正文大
        if bold and size > body_size:
            return True

        # 字号等于最大字号 → 可能是一级标题
        all_sizes = [v["size"] for v in font_map.values()]
        if all_sizes and size == max(all_sizes) and len(line) < 30:
            return True

    # --- 无字体信息 (OCR): 用正则兜底 ---
    for pattern in HEADING_PATTERNS:
        if pattern.match(line):
            return True

    return False


def _get_overlap_lines(lines: list[str], overlap_size: int = None) -> list[str]:
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
