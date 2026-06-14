"""
检索引擎 — 混合检索 (向量 + BM25) + 查询扩展 + LLM 生成

核心流程:
1. 问题 → 查询扩展 (生成多个搜索词)
2. 每个搜索词 → 双路检索:
   a. Embedding 语义检索 (ChromaDB)
   b. BM25 关键词检索 (jieba 分词)
3. RRF (Reciprocal Rank Fusion) 融合排序 → top-K
4. 文档片段 + 问题 → LLM (streaming) → 回答
"""

from typing import Iterator

from engine.embedder import embedder
from engine.store import store
from engine.bm25_retriever import bm25
from engine.reranker import reranker
from engine.cache import cache
import config


# 查询扩展用的简短提示词
QUERY_EXPANSION_PROMPT = """你是一个搜索查询扩展助手。给定一个学生关于毕业论文的问题，生成3-4个不同角度的搜索查询。

规则：
- 从不同角度、不同措辞改写问题
- 使用可能的同义词和相关术语（特别是中文教育/论文领域的常用词）
- 例如："单面打印还是双面打印" → 扩展为"打印要求"、"装订规范"、"页面设置"、"排版格式"
- 每个查询一行，不要编号，不要解释
- 直接返回查询词，每行一个"""


class Retriever:
    """RAG 检索引擎 (单例, 懒加载)"""

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
        self._client = None
        self._provider = config.LLM_PROVIDER

    # ---- 公开 API ----

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """单查询 Embedding 检索（简单场景）"""
        query_embedding = embedder.embed_query(query)
        return store.query(query_embedding, top_k=top_k)

    def multi_search(self, question: str, top_k: int = None) -> list[dict]:
        """
        混合检索 + 重排序:
        1. 查询扩展
        2. 向量检索 + BM25 检索 → RRF 融合 → top-N 候选
        3. Cross-Encoder Reranker 精排 → top-K

        向量检索 (语义)    BM25 检索 (关键词)
             │                  │
             └────────┬─────────┘
                      ▼
              RRF 融合 → top-20
                      │
                      ▼
            Reranker 精排 → top-5
        """
        if top_k is None:
            top_k = config.TOP_K

        # ---- 阶段1: 混合检索（粗排，多取候选） ----
        candidate_count = top_k * 4  # 粗排取 4 倍候选
        queries = self._expand_queries(question)
        all_queries = [question] + queries

        rrf_scores: dict[str, float] = {}
        hit_map: dict[str, dict] = {}
        RRF_K = 60

        for q in all_queries:
            # 向量检索
            emb_hits = self.search(q, top_k=candidate_count)
            for rank, hit in enumerate(emb_hits, 1):
                cid = hit.get("chunk_id") or hit.get("content", "")[:50]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
                if cid not in hit_map:
                    hit_map[cid] = hit
                    hit_map[cid]["match_type"] = "语义"

            # BM25 检索
            bm25_hits = bm25.search(q, top_k=candidate_count)
            for rank, hit in enumerate(bm25_hits, 1):
                cid = hit.get("chunk_id") or hit.get("content", "")[:50]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
                if cid not in hit_map:
                    hit_map[cid] = hit
                    hit_map[cid]["match_type"] = "关键词"
                else:
                    hit_map[cid]["match_type"] = "语义+关键词"

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for cid, rrf in sorted_ids[:candidate_count]:
            hit = hit_map[cid]
            sim = hit.get("similarity", 0) or hit.get("bm25_score", 0) or 0
            candidates.append({
                **hit,
                "similarity": sim,
                "rrf_score": round(rrf, 5),
                "match_type": hit.get("match_type", "未知"),
            })

        # ---- 阶段2: Cross-Encoder 精排 ----
        if len(candidates) <= top_k:
            return candidates

        ranked = reranker.rerank(question, candidates, top_k=top_k)
        return ranked

    def ask(self, question: str, top_k: int = None) -> str:
        """检索 + 生成完整回答（非流式）"""
        self._ensure_client()
        cached = cache.get(question)
        if cached:
            return cached["answer"]
        hits = self.multi_search(question, top_k=top_k)
        system_prompt, user_prompt = self._build_messages(question, hits)
        answer = self._do_ask(system_prompt, user_prompt)
        confidence = max((h.get("rerank_score", 0) or h.get("similarity", 0)) for h in hits) if hits else 0
        cache.set(question, answer, hits, confidence=confidence)
        return answer

    def ask_stream(self, question: str, top_k: int = None) -> Iterator[tuple[str, list[dict]]]:
        """检索 + 流式生成。命中缓存则直接返回。"""
        self._ensure_client()

        # 查缓存
        cached = cache.get(question)
        if cached:
            yield ("", cached["hits"])
            yield (cached["answer"], cached["hits"])
            return

        hits = self.multi_search(question, top_k=top_k)
        system_prompt, user_prompt = self._build_messages(question, hits)

        yield ("", hits)

        full_answer = ""
        for delta in self._do_ask_stream(system_prompt, user_prompt):
            full_answer += delta
            yield (delta, hits)

        # 写入缓存 (传入最高 rerank 分作为置信度)
        confidence = max((h.get("rerank_score", 0) or h.get("similarity", 0)) for h in hits) if hits else 0
        cache.set(question, full_answer, hits, confidence=confidence)

    # ---- 内部: 查询扩展 ----

    def _expand_queries(self, question: str) -> list[str]:
        """
        用 LLM 将学生问题扩展为多个搜索词

        例如: "论文是单面打印还是双面打印"
        → ["打印要求", "装订规范", "页面设置格式", "排版打印规定"]
        """
        self._ensure_client()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=150,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            text = response.choices[0].message.content.strip()
            # 按行拆分，过滤空行
            queries = [q.strip() for q in text.split("\n") if q.strip()]
            # 去掉可能的编号前缀 "1. " "2. " 等
            import re
            queries = [re.sub(r"^\d+[\.\、\)]\s*", "", q) for q in queries]
            return queries[:4]  # 最多4个
        except Exception:
            # 扩展失败时退回原始问题
            return []

    # ---- 内部: 客户端管理 ----

    def _ensure_client(self):
        if self._client is not None:
            return

        if self._provider == "siliconflow":
            self._init_siliconflow()
        elif self._provider == "deepseek":
            self._init_deepseek()
        elif self._provider == "anthropic":
            self._init_anthropic()
        elif self._provider == "openai":
            self._init_openai()
        else:
            raise ValueError(
                f"不支持的 LLM Provider: {self._provider}。"
                "可选: 'siliconflow' | 'deepseek' | 'anthropic' | 'openai'"
            )

    def _init_siliconflow(self):
        if not config.SILICONFLOW_API_KEY:
            raise RuntimeError(
                "未配置 SILICONFLOW_API_KEY。"
                "请在 .streamlit/secrets.toml 中设置。"
                "获取地址: https://www.siliconflow.cn/"
            )
        from openai import OpenAI
        self._client = OpenAI(
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL,
        )
        self._model = config.SILICONFLOW_LLM_MODEL

    def _init_deepseek(self):
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。"
                "请在 .streamlit/secrets.toml 中设置。"
                "获取地址: https://platform.deepseek.com/"
            )
        from openai import OpenAI
        self._client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._model = config.DEEPSEEK_MODEL

    def _init_anthropic(self):
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "未配置 ANTHROPIC_API_KEY。"
                "请在 .streamlit/secrets.toml 中设置。"
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.CLAUDE_MODEL

    def _init_openai(self):
        import os
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", config._get_config("OPENAI_API_KEY")),
            base_url=os.getenv("OPENAI_BASE_URL", config._get_config("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        )
        self._model = config._get_config("OPENAI_MODEL", "gpt-4o")

    # ---- 内部: Prompt 构建 ----

    def _build_messages(self, question: str, hits: list[dict]) -> tuple[str, str]:
        """构建 system prompt + user prompt"""
        if not hits:
            user_prompt = f"""【学生问题】
{question}

---
注意：知识库中没有找到与该问题相关的文档。请告知学生没有找到相关信息，并建议咨询教务老师。"""
            return config.SYSTEM_PROMPT, user_prompt

        context_parts = []
        for i, hit in enumerate(hits, 1):
            heading_info = f" > {hit['heading']}" if hit.get("heading") else ""
            sim = hit.get("similarity", 0)
            match = hit.get("match_type", "未知")
            confidence = "🟢 高相关" if sim > 0.7 else ("🟡 中相关" if sim > 0.5 else "🔴 低相关")
            context_parts.append(
                f"### 文档片段 {i} ({confidence} | 匹配: {match})\n"
                f"**来源**: {hit['source']}（第{hit['page']}页{heading_info}）\n"
                f"```\n{hit['content']}\n```"
            )

        context_text = "\n\n".join(context_parts)
        user_prompt = f"""以下是与学生问题相关的学校官方文档内容。
**注意：低相关的片段可能不包含答案，请优先依据高相关片段回答。**

{context_text}

---

【学生问题】
{question}

请根据以上文档内容回答。要求：
- 如果文档中有明确原文，用「」引用
- 如果只有低相关片段且无法确定答案，请如实说明
- 标注信息来源（文件名、页码）"""

        return config.SYSTEM_PROMPT, user_prompt

    # ---- 内部: LLM 调用 ----

    def _do_ask(self, system_prompt: str, user_prompt: str) -> str:
        """非流式调用 LLM"""
        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        else:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=2048,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

    def _do_ask_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """流式调用 LLM"""
        if self._provider == "anthropic":
            with self._client.messages.stream(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            stream = self._client.chat.completions.create(
                model=self._model,
                max_tokens=2048,
                temperature=0.3,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content


# 全局单例
retriever = Retriever()
