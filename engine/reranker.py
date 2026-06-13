"""
重排序模块 — Cross-Encoder Reranker

流程: 检索 top-N (候选) → Reranker 精排 → top-K (最终)

为什么需要:
- Embedding (bi-encoder) 把问题和文档分别编码，丢失交互信息
- Cross-encoder 把 (问题, 文档) 一起编码，准确度大幅提升
- 代价: 慢一些，所以只对 top-N 候选用，不全量跑

模型: BAAI/bge-reranker-v2-m3 (多语言, 中文优秀)
"""

from typing import Optional

import config


class Reranker:
    """Cross-Encoder 重排序器 (单例, 懒加载)"""

    _instance: Optional["Reranker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._model_name = "BAAI/bge-reranker-v2-m3"

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        import streamlit as st
        with st.spinner(f"加载重排序模型: {self._model_name} ..."):
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        对候选文档重排序

        Args:
            query: 原始问题
            documents: 候选文档列表 [{content, source, ...}, ...]
            top_k: 返回数量

        Returns:
            重排序后的文档列表，新增 rerank_score 字段
        """
        if not documents:
            return []

        self._ensure_model()

        # 组装 (query, doc) 对
        pairs = [(query, doc["content"]) for doc in documents]

        # Cross-encoder 打分
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
        )

        # 归一化到 [0, 1]
        scores = self._normalize(scores)

        # 附加分数并排序
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(float(score), 4)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

        # 更新 similarity 为 rerank_score（后续 Prompt 用）
        for d in ranked:
            d["similarity"] = d["rerank_score"]

        return ranked[:top_k]

    def _normalize(self, scores) -> list[float]:
        """将 scores 归一化到 [0, 1]"""
        import numpy as np
        s = np.array(scores)
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-8:
            return [0.5] * len(scores)
        return ((s - s_min) / (s_max - s_min)).tolist()


# 全局单例
reranker = Reranker()
