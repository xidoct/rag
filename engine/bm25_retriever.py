"""
BM25 关键词检索器 — 中文分词 + 持久化

作为向量检索的补充，解决 "单面打印" 搜不到 "装订要求" 这类
语义相近但用词不同的问题。BM25 靠词频匹配，不怕用词偏差。

持久化: data/bm25_index.pkl  (与 ChromaDB 同步)
"""

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

import config


class BM25Retriever:
    """BM25 关键词检索器 (单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._index_path = config.DATA_DIR / "bm25_index.pkl"
        self._bm25: BM25Okapi | None = None
        self._corpus: list[list[str]] = []    # 分词后的文档
        self._chunk_ids: list[str] = []       # ChromaDB 的 chunk id
        self._contents: list[str] = []        # 原始文本
        self._metadatas: list[dict] = []      # 元数据
        self._load()

    # ---- 索引构建 ----

    def rebuild_from_store(self):
        """从 ChromaDB 读取所有块，重建 BM25 索引"""
        from engine.store import store
        col = store.collection
        result = col.get(include=["documents", "metadatas"])

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not ids or not documents:
            self._clear()
            return

        self._chunk_ids = list(ids)
        self._contents = list(documents)
        self._metadatas = list(metadatas)
        self._corpus = [list(jieba.cut(text)) for text in documents]
        self._bm25 = BM25Okapi(self._corpus)
        self._save()
        print(f"[BM25] 索引已重建: {len(ids)} 个文档块")

    def _clear(self):
        self._bm25 = None
        self._corpus = []
        self._chunk_ids = []
        self._contents = []
        self._metadatas = []
        if self._index_path.exists():
            self._index_path.unlink()

    # ---- 检索 ----

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 关键词检索

        Returns:
            [{content, source, page, heading, bm25_score, chunk_id}, ...]
        """
        if self._bm25 is None or not self._corpus:
            return []

        # 中文分词
        tokenized_query = list(jieba.cut(query))

        # BM25 打分
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top-K
        # 按分数降序排列
        indexed = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in indexed:
            if score <= 0:
                continue
            meta = self._metadatas[idx] if idx < len(self._metadatas) else {}
            results.append({
                "content": self._contents[idx] if idx < len(self._contents) else "",
                "source": meta.get("source", ""),
                "page": meta.get("page", 1),
                "heading": meta.get("heading", ""),
                "is_table": meta.get("is_table", False),
                "bm25_score": round(float(score), 4),
                "chunk_id": self._chunk_ids[idx] if idx < len(self._chunk_ids) else "",
            })

        return results

    # ---- 持久化 ----

    def _save(self):
        """保存分词语料到磁盘（BM25Okapi 对象不持久化，重建很快）"""
        data = {
            "chunk_ids": self._chunk_ids,
            "contents": self._contents,
            "metadatas": self._metadatas,
            "corpus": self._corpus,
        }
        with open(self._index_path, "wb") as f:
            pickle.dump(data, f)

    def _load(self):
        """从磁盘加载语料并重建 BM25"""
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data.get("chunk_ids", [])
            self._contents = data.get("contents", [])
            self._metadatas = data.get("metadatas", [])
            self._corpus = data.get("corpus", [])
            if self._corpus:
                self._bm25 = BM25Okapi(self._corpus)
                print(f"[BM25] 从磁盘加载: {len(self._corpus)} 个文档块")
        except Exception:
            self._clear()


# 全局单例
bm25 = BM25Retriever()
