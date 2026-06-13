"""
向量化模块 — 将文本转为 embedding 向量

支持的 Provider:
- siliconflow — 硅基流动 (默认, OpenAI 兼容, 一个 Key 搞定)
- local      — 本地 BGE 模型 (离线免费)
- voyage     — Voyage AI API
- openai     — OpenAI / 任何兼容接口

单例 + 懒加载。
"""

import time
from typing import Optional

import config


class Embedder:
    """文本向量化器 (单例, 懒加载)"""

    _instance: Optional["Embedder"] = None

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
        self._provider = config.EMBEDDING_PROVIDER

    # ---- 公开 API ----

    def embed(self, texts: str | list[str], input_type: str = "document") -> list[list[float]]:
        self._ensure_model()
        if isinstance(texts, str):
            texts = [texts]
        return self._do_embed(texts, input_type)

    def embed_query(self, question: str) -> list[float]:
        return self.embed(question, input_type="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, input_type="document")

    # ---- 内部: 模型加载 ----

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._provider == "siliconflow":
            self._init_openai_compatible(
                api_key=config.SILICONFLOW_API_KEY,
                base_url=config.SILICONFLOW_BASE_URL,
                model=config.SILICONFLOW_EMBEDDING_MODEL,
                name="硅基流动",
            )
        elif self._provider == "openai":
            self._init_openai_compatible(
                api_key=config._get_config("OPENAI_API_KEY"),
                base_url=config._get_config("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=config._get_config("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                name="OpenAI",
            )
        elif self._provider == "local":
            self._init_local()
        elif self._provider == "voyage":
            self._init_voyage()
        else:
            raise ValueError(
                f"不支持的 Embedding Provider: {self._provider}。"
                "可选: 'siliconflow' | 'local' | 'voyage' | 'openai'"
            )

    def _init_openai_compatible(self, api_key: str, base_url: str, model: str, name: str):
        """通用 OpenAI 兼容 Embedding 初始化（硅基流动 / OpenAI 等）"""
        if not api_key:
            raise RuntimeError(
                f"{name} API Key 未配置。"
                "请在 .streamlit/secrets.toml 中设置 SILICONFLOW_API_KEY。"
                f"\n获取地址: {'https://www.siliconflow.cn/' if '硅基' in name else 'https://platform.openai.com/'}"
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._provider_type = "openai_compat"

    def _init_local(self):
        from sentence_transformers import SentenceTransformer
        import streamlit as st
        model_name = config.LOCAL_EMBEDDING_MODEL
        with st.spinner(f"加载本地模型: {model_name} ..."):
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
        self._provider_type = "local"

    def _init_voyage(self):
        if not config.VOYAGE_API_KEY:
            raise RuntimeError(
                "未配置 VOYAGE_API_KEY。获取地址: https://www.voyageai.com/"
            )
        import voyageai
        voyageai.api_key = config.VOYAGE_API_KEY
        self._model = voyageai
        self._model_name = config._get_config("VOYAGE_MODEL", "voyage-3")
        self._provider_type = "voyage"

    # ---- 内部: 向量化执行 ----

    def _do_embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._provider_type == "openai_compat":
                    # 硅基流动 / OpenAI 兼容 Embedding API
                    resp = self._client.embeddings.create(
                        model=self._model,
                        input=texts,
                    )
                    return [d.embedding for d in resp.data]

                elif self._provider_type == "local":
                    embeddings = self._model.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    return embeddings.tolist()

                elif self._provider_type == "voyage":
                    result = self._model.embed(
                        texts=texts,
                        model=self._model_name,
                        input_type=input_type,
                    )
                    return result.embeddings

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding 失败 (已重试 {max_retries} 次): {e}") from e


embedder = Embedder()
