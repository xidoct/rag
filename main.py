"""
RAG 毕业论文知识问答系统 — Streamlit 入口

启动方式:
    cd 论文知识查询
    streamlit run main.py

首次使用:
    1. 编辑 .streamlit/secrets.toml，填入 API Key
    2. pip install -r requirements.txt
    3. streamlit run main.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python Path 中
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# 页面配置（必须是第一个 Streamlit 命令）
st.set_page_config(
    page_title="毕业论文知识问答",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import upload, chat, manage


# --- 侧边栏 ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.title("🎓 论文知识问答")
    st.markdown("基于学校官方文档的毕业论文 RAG 问答系统")

    st.divider()

    # 页面导航
    page = st.radio(
        "📋 导航",
        ["💬 知识问答", "📤 文档上传", "📚 文档管理"],
        label_visibility="collapsed",
    )

    st.divider()

    # 状态提示
    try:
        from engine.store import store
        stats = store.get_stats()
        st.caption(f"📊 知识库: {stats['total_files']} 个文档, {stats['total_chunks']} 个知识块")
    except Exception:
        st.caption("⚠️ 知识库未初始化")

    st.caption("🔑 服务状态检查...")

    import config

    # LLM
    llm_p = config.LLM_PROVIDER
    if llm_p == "siliconflow":
        if config.SILICONFLOW_API_KEY:
            st.caption(f"  ✅ LLM: 硅基流动 ({config.SILICONFLOW_LLM_MODEL})")
        else:
            st.caption("  ❌ SILICONFLOW_API_KEY 未配置")
    elif llm_p == "deepseek":
        if config.DEEPSEEK_API_KEY:
            st.caption(f"  ✅ LLM: DeepSeek ({config.DEEPSEEK_MODEL})")
        else:
            st.caption("  ❌ DeepSeek API Key 未配置")
    elif llm_p == "anthropic":
        if config.ANTHROPIC_API_KEY:
            st.caption(f"  ✅ LLM: Claude ({config.CLAUDE_MODEL})")
        else:
            st.caption("  ❌ Anthropic API Key 未配置")
    else:
        st.caption(f"  ⚙️ LLM: {llm_p}")

    # Embedding
    emb_p = config.EMBEDDING_PROVIDER
    if emb_p == "siliconflow":
        if config.SILICONFLOW_API_KEY:
            st.caption(f"  ✅ Embedding: 硅基流动 ({config.SILICONFLOW_EMBEDDING_MODEL})")
        else:
            st.caption("  ❌ SILICONFLOW_API_KEY 未配置")
    elif emb_p == "local":
        st.caption("  ✅ Embedding: 本地 BGE 模型")
    else:
        st.caption(f"  ⚙️ Embedding: {emb_p}")

    st.divider()
    st.caption("💡 提示: 先在「文档上传」导入文档，再到「知识问答」提问")


# --- 页面路由 ---
if "💬 知识问答" in page:
    chat.render()
elif "📤 文档上传" in page:
    upload.render()
elif "📚 文档管理" in page:
    manage.render()
