"""
RAG 知识问答系统 — 配置文件

配置读取优先级:
1. 环境变量 (os.environ)
2. .streamlit/secrets.toml (直接解析 TOML)
3. 默认值

默认 Provider: 硅基流动 (SiliconFlow)
- 一个 API Key 同时管 LLM + Embedding
- 兼容 OpenAI 接口格式
"""

import os
from pathlib import Path


# --- 项目路径 ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"

for d in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- 缓存 TOML 解析结果 ---
_secrets_cache: dict | None = None


def _load_secrets_file() -> dict:
    global _secrets_cache
    if _secrets_cache is not None:
        return _secrets_cache
    if not SECRETS_FILE.exists():
        _secrets_cache = {}
        return _secrets_cache
    try:
        import sys
        if sys.version_info >= (3, 11):
            import tomllib
            with open(SECRETS_FILE, "rb") as f:
                _secrets_cache = tomllib.load(f)
        else:
            import toml
            _secrets_cache = toml.load(str(SECRETS_FILE))
    except ImportError:
        _secrets_cache = _parse_toml_simple(str(SECRETS_FILE))
    except Exception:
        _secrets_cache = {}
    return _secrets_cache


def _parse_toml_simple(filepath: str) -> dict:
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                result[key] = val
    return result


def _get_config(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    secrets = _load_secrets_file()
    if key in secrets:
        val = secrets[key]
        if isinstance(val, str):
            return val.strip()
        return str(val)
    try:
        import streamlit as st
        st_val = st.secrets.get(key)
        if st_val and isinstance(st_val, str):
            return st_val.strip()
    except Exception:
        pass
    return default


# ============================================================
#  硅基流动 (SiliconFlow) — 默认 Provider
# ============================================================

SILICONFLOW_API_KEY = _get_config("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = _get_config(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)

# --- LLM ---
LLM_PROVIDER = _get_config("LLM_PROVIDER", "siliconflow")
# 可选: "siliconflow" | "deepseek" | "anthropic" | "openai"
SILICONFLOW_LLM_MODEL = _get_config(
    "SILICONFLOW_LLM_MODEL", "deepseek-ai/DeepSeek-V3"
)

# --- Embedding ---
EMBEDDING_PROVIDER = _get_config("EMBEDDING_PROVIDER", "siliconflow")
# 可选: "siliconflow" | "local" | "voyage" | "openai"
SILICONFLOW_EMBEDDING_MODEL = _get_config(
    "SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
)

# --- 向下兼容: 保留其他 Provider 的配置 ---
DEEPSEEK_API_KEY = _get_config("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _get_config("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _get_config("DEEPSEEK_MODEL", "deepseek-chat")
ANTHROPIC_API_KEY = _get_config("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _get_config("CLAUDE_MODEL", "claude-sonnet-4-6")
VOYAGE_API_KEY = _get_config("VOYAGE_API_KEY")
LOCAL_EMBEDDING_MODEL = _get_config(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
)


# ============================================================
#  分块配置
# ============================================================

CHUNK_SIZE = int(_get_config("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get_config("CHUNK_OVERLAP", "150"))
TABLE_CHUNK_SIZE = int(_get_config("TABLE_CHUNK_SIZE", "600"))


# ============================================================
#  检索配置
# ============================================================

TOP_K = int(_get_config("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(_get_config("SIMILARITY_THRESHOLD", "0.3"))


# ============================================================
#  OCR 配置 (扫描件 PDF)
# ============================================================

OCR_LANG = _get_config("OCR_LANG", "ch")                # PaddleOCR 语言
OCR_DPI = int(_get_config("OCR_DPI", "200"))           # 渲染分辨率


# ============================================================
#  系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一所大学的毕业论文政策问答助手。你的职责是**严格依据学校官方文档**，准确回答学生关于毕业论文的问题。

## 核心铁律
1. **只使用文档中明确写出的内容**，不得推测、联想或补全。
2. 文档片段标注了相关度（🟢高/🟡中/🔴低），优先采信高相关片段。
3. **🔴低相关片段仅供参考**，不能作为唯一答案依据。如果只有低相关片段，必须说明"信息不确定，建议查阅原文或咨询教务老师"。
4. 如果所有片段都没有答案，明确说："根据现有文档，没有找到关于[具体问题]的规定。"

## 回答格式
1. 📝 用「」引用文档原文
2. ✅ 基于原文的结论
3. ⚠️ 如有不确定，明确说明
4. 📄 标注来源（文件名、页码）"""
