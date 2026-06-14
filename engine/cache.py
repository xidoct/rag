"""
查询缓存 — 语义匹配 + LRU 淘汰 + 高置信度过滤

改进:
1. Key 从"问题文本 MD5"改为"问题向量 + 余弦相似度匹配"
   → "论文格式" 和 "论文排版" 指向同一缓存
2. FIFO → 真 LRU: 淘汰最近最少访问的
3. 只缓存高置信度答案: 低分/未找到答案的不缓存

存储: data/query_cache.json (文档变动时自动清空)
"""

import json
import time
import numpy as np
from pathlib import Path

import config

MAX_ENTRIES = 500
SEMANTIC_THRESHOLD = 0.92   # 余弦相似度 ≥ 此值视为相同问题
MIN_CONFIDENCE = 0.5         # Reranker 最高分 ≥ 此值才缓存


class QueryCache:
    """语义查询缓存 (单例)"""

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
        self._path = config.DATA_DIR / "query_cache.json"
        self._data: dict[str, dict] = self._load()

    # ---- 公开 API ----

    def get(self, question: str) -> dict | None:
        """
        语义匹配查找缓存
        1. 计算问题 embedding
        2. 与所有缓存条目的 embedding 做余弦相似度
        3. 最佳匹配 > SEMANTIC_THRESHOLD → 命中，更新 LRU
        4. 否则 → 未命中
        """
        if not self._data:
            return None

        q_emb = self._embed_question(question)
        if q_emb is None:
            return None

        best_key, best_sim = None, 0.0
        for k, entry in self._data.items():
            cached_emb = entry.get("embedding")
            if not cached_emb:
                continue
            sim = self._cosine_sim(q_emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = k

        if best_key and best_sim >= SEMANTIC_THRESHOLD:
            entry = self._data[best_key]
            entry["last_access"] = time.time()
            entry["access_count"] = entry.get("access_count", 0) + 1
            self._save()
            return {
                "question": entry["question"],
                "answer": entry["answer"],
                "hits": entry.get("hits", []),
                "similarity": round(best_sim, 4),
            }

        return None

    def set(self, question: str, answer: str, hits: list[dict], confidence: float):
        """
        写入缓存 (仅高置信度)
        """
        # 置信度过滤
        if confidence < MIN_CONFIDENCE:
            return

        # 拒绝无结果回答
        if "没有找到" in answer[:200] or "没有找到相关信息" in answer[:200]:
            return

        # 计算 embedding
        q_emb = self._embed_question(question)
        if q_emb is None:
            return

        # 检查是否已有高度相似的缓存，有则覆盖
        best_key, best_sim = None, 0.0
        for k, entry in self._data.items():
            cached_emb = entry.get("embedding")
            if not cached_emb:
                continue
            sim = self._cosine_sim(q_emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = k

        now = time.time()
        if best_key and best_sim >= SEMANTIC_THRESHOLD:
            # 覆盖旧缓存
            entry = self._data[best_key]
            entry["question"] = question
            entry["answer"] = answer
            entry["hits"] = self._compress_hits(hits)
            entry["embedding"] = q_emb
            entry["last_access"] = now
            entry["confidence"] = confidence
        else:
            # 新条目
            import hashlib
            k = hashlib.md5(question.strip().encode()).hexdigest()
            self._data[k] = {
                "question": question,
                "answer": answer,
                "hits": self._compress_hits(hits),
                "embedding": q_emb,
                "confidence": confidence,
                "created_at": now,
                "last_access": now,
                "access_count": 0,
            }

        # LRU 淘汰
        while len(self._data) > MAX_ENTRIES:
            lru_key = min(self._data, key=lambda k: self._data[k]["last_access"])
            del self._data[lru_key]

        self._save()

    def clear(self):
        self._data = {}
        if self._path.exists():
            self._path.unlink()

    # ---- 内部 ----

    def _embed_question(self, question: str) -> list[float] | None:
        try:
            from engine.embedder import embedder
            return embedder.embed_query(question)
        except Exception:
            return None

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        na, nb = np.array(a), np.array(b)
        norm = np.linalg.norm(na) * np.linalg.norm(nb)
        if norm < 1e-10:
            return 0.0
        return float(np.dot(na, nb) / norm)

    @staticmethod
    def _compress_hits(hits: list[dict]) -> list[dict]:
        return [
            {
                "source": h.get("source", ""),
                "page": h.get("page", 1),
                "content": h.get("content", "")[:200],
            }
            for h in hits
        ]

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self):
        return len(self._data)


cache = QueryCache()
