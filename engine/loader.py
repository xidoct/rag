"""
文档加载器 — 支持 ZIP / PDF / Word 解析

输入: 文件路径 (支持 .zip / .pdf / .docx)
输出: 标准化 Document 对象列表
"""

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from docx import Document as DocxDocument

import config


@dataclass
class Document:
    """标准化的文档片段"""
    content: str                          # 文本内容
    source: str                           # 来源文件名
    page: int = 1                         # 页码 (PDF 有效)
    heading: str = ""                     # 所属章节标题
    is_table: bool = False                # 是否表格内容
    metadata: dict = field(default_factory=dict)


def load_file(file_path: str | Path) -> list[Document]:
    """
    根据文件扩展名自动选择解析器
    支持: .pdf / .docx / .zip (内含 pdf/docx)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".docx", ".doc"):
        return _load_docx(path)
    elif ext == ".zip":
        return _load_zip(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请上传 PDF / Word / ZIP")


def _load_pdf(path: Path) -> list[Document]:
    """解析 PDF，按页提取文本"""
    docs = []
    doc = fitz.open(path)
    source = path.name
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue

            # 尝试提取表格
            tables = page.find_tables()
            table_texts = _extract_table_texts(tables)

            if table_texts:
                # 表格内容单独成块
                for t_text in table_texts:
                    docs.append(Document(
                        content=t_text,
                        source=source,
                        page=page_num + 1,
                        is_table=True,
                        metadata={"type": "table"}
                    ))

            if text:
                docs.append(Document(
                    content=text,
                    source=source,
                    page=page_num + 1,
                    metadata={"type": "text"}
                ))
    finally:
        doc.close()
    return docs


def _load_docx(path: Path) -> list[Document]:
    """解析 Word 文档，按段落提取"""
    docs = []
    source = path.name
    doc = DocxDocument(str(path))

    # 提取段落
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if paragraphs:
        docs.append(Document(
            content="\n".join(paragraphs),
            source=source,
            metadata={"type": "text"}
        ))

    # 提取表格
    for idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        table_text = "\n".join(rows)
        if table_text.strip():
            docs.append(Document(
                content=table_text,
                source=source,
                is_table=True,
                metadata={"type": "table", "table_index": idx}
            ))

    return docs


def _load_zip(path: Path) -> list[Document]:
    """解压 ZIP 并递归解析其中的 PDF/Word 文件"""
    all_docs = []
    upload_dir = config.UPLOAD_DIR

    with zipfile.ZipFile(path, "r") as zf:
        # 解压到 uploads 目录
        extract_dir = upload_dir / path.stem
        zf.extractall(extract_dir)

        # 遍历解压后的文件
        for extracted in extract_dir.rglob("*"):
            if extracted.is_file():
                try:
                    docs = load_file(extracted)
                    all_docs.extend(docs)
                except ValueError:
                    # 跳过不支持的文件
                    continue

    return all_docs


def _extract_table_texts(tables) -> list[str]:
    """将 PyMuPDF 表格对象转为可读文本"""
    result = []
    for table in tables:
        try:
            data = table.extract()
            if not data:
                continue
            rows = []
            for row in data:
                # 过滤掉 None 单元格
                cells = [str(c) if c is not None else "" for c in row]
                rows.append(" | ".join(cells))
            text = "\n".join(rows)
            if text.strip():
                result.append(text)
        except Exception:
            continue
    return result


def stream_documents(file_path: str | Path) -> Iterator[Document]:
    """流式 yield Document，适合大文件处理时显示进度"""
    for doc in load_file(file_path):
        yield doc


def scan_folder(folder_path: str | Path) -> list[Path]:
    """
    递归扫描文件夹，返回所有 PDF/Word 文件路径
    用于本地大文件批量导入（不走浏览器上传）
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder_path}")

    supported = {".pdf", ".docx", ".doc"}
    files = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported:
            files.append(f)
    return sorted(files)


def load_folder(folder_path: str | Path) -> Iterator[Document]:
    """
    流式加载整个文件夹的文档
    逐个文件 yield Document，适合大文件夹处理时显示进度
    """
    files = scan_folder(folder_path)
    for file_path in files:
        try:
            for doc in load_file(file_path):
                yield doc
        except Exception:
            # 跳过解析失败的文件，继续处理后续文件
            continue
