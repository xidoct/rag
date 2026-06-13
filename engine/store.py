"""
ChromaDB 向量存储 — 文档入库、检索、管理

Collection 结构:
- documents: 文档文本
- metadatas: {source, page, heading, is_table, chunk_type, ...}
- ids: 自动生成唯一 ID
- embeddings: 预计算的向量
"""

import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from engine.loader import Document
import config


class VectorStore:
    """ChromaDB 向量存储封装"""

    _instance: Optional["VectorStore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 持久化客户端
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="graduation_thesis_docs",
            metadata={"description": "毕业论文相关文档知识库"},
        )

    @property
    def collection(self):
        return self._collection

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> int:
        """
        批量添加文档块到向量库

        Args:
            documents: 分块后的 Document 列表
            embeddings: 对应的 embedding 向量

        Returns:
            添加的块数量
        """
        if not documents:
            return 0

        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.content for doc in documents]
        metadatas = [
            {
                "source": doc.source,
                "page": doc.page,
                "heading": doc.heading or "",
                "is_table": doc.is_table,
                "chunk_type": doc.metadata.get("chunk_type", "text"),
            }
            for doc in documents
        ]

        self._collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = None,
        threshold: float = None,
    ) -> list[dict]:
        """
        相似度检索

        Returns:
            [{content, source, page, heading, distance, ...}, ...]
        """
        if top_k is None:
            top_k = config.TOP_K
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB 返回 cosine distance (0=完全相同, 1=完全不同)
            # 转换为相似度: 1 - distance
            similarity = 1.0 - distance
            if similarity < threshold:
                continue

            hits.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i].get("page", 1),
                "heading": results["metadatas"][0][i].get("heading", ""),
                "is_table": results["metadatas"][0][i].get("is_table", False),
                "similarity": round(similarity, 4),
                "chunk_id": results["ids"][0][i],
            })

        return hits

    def delete_by_source(self, source_name: str) -> int:
        """按来源文件名删除所有相关块"""
        # ChromaDB 需要先查出 ID 再删除
        results = self._collection.get(
            where={"source": source_name},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_sources(self) -> list[dict]:
        """列出所有已入库的文档来源及块数统计"""
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        # 按 source 聚合统计
        stats: dict[str, dict] = {}
        for meta in metadatas:
            src = meta["source"]
            if src not in stats:
                stats[src] = {"source": src, "chunks": 0, "pages": set()}
            stats[src]["chunks"] += 1
            page = meta.get("page", 1)
            if page:
                stats[src]["pages"].add(page)

        # 转为列表，pages 集合转范围描述
        result = []
        for src, info in stats.items():
            pages = sorted(info["pages"])
            page_range = _format_page_range(pages)
            result.append({
                "source": src,
                "chunks": info["chunks"],
                "pages": page_range,
            })
        return result

    def get_stats(self) -> dict:
        """获取知识库整体统计"""
        sources = self.list_sources()
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        table_chunks = sum(1 for m in metadatas if m.get("is_table"))
        text_chunks = len(metadatas) - table_chunks

        return {
            "total_files": len(sources),
            "total_chunks": len(metadatas),
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "sources": sources,
        }

    def clear_all(self):
        """清空知识库（危险操作）"""
        self._client.delete_collection(name="graduation_thesis_docs")
        self._collection = self._client.get_or_create_collection(
            name="graduation_thesis_docs",
            metadata={"description": "毕业论文相关文档知识库"},
        )


def _format_page_range(pages: list[int]) -> str:
    """将页码列表格式化为范围字符串，如 '1-3, 5, 8-10'"""
    if not pages:
        return ""
    ranges = []
    start = pages[0]
    end = pages[0]
    for p in pages[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append((start, end))
            start = p
            end = p
    ranges.append((start, end))

    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")
    return ", ".join(parts)


# 全局单例
store = VectorStore()
