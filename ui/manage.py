"""
文档管理页面 — 查看已入库文档、删除、统计
"""

import streamlit as st

from engine.store import store
from engine.bm25_retriever import bm25
from engine.cache import cache


def render():
    """渲染管理页面"""
    st.title("📚 文档库管理")
    st.markdown("查看已入库的文档，支持删除和统计。")

    # --- 统计概览 ---
    try:
        stats = store.get_stats()
    except Exception as e:
        st.error(f"无法访问知识库: {e}")
        return

    # 指标卡片
    cols = st.columns(4)
    cols[0].metric("📄 文档总数", stats["total_files"])
    cols[1].metric("🧩 知识块总数", stats["total_chunks"])
    cols[2].metric("📝 文本块", stats.get("text_chunks", 0))
    cols[3].metric("📊 表格块", stats.get("table_chunks", 0))

    st.divider()

    # --- 文档列表 ---
    sources = stats.get("sources", [])

    if not sources:
        st.info("📭 知识库为空。请先到「文档上传」页面导入文档。")
        return

    st.subheader(f"📋 已入库文档 ({len(sources)})")

    for src in sources:
        with st.expander(
            f"📄 {src['source']} — {src['chunks']} 块, 页码: {src['pages']}",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"""
            | 属性 | 值 |
            |------|-----|
            | 文件名 | `{src['source']}` |
            | 知识块数 | {src['chunks']} |
            | 涉及页码 | {src['pages']} |
            """)

            if col2.button("🗑️ 删除", key=f"del_{src['source']}", type="secondary"):
                _delete_source(src['source'])


    st.divider()

    # --- 危险操作 ---
    with st.expander("⚠️ 危险操作", expanded=False):
        st.warning("以下操作不可撤销，请谨慎使用。")
        if st.button("🧹 清空全部知识库", type="primary"):
            if st.session_state.get("confirm_clear"):
                store.clear_all()
                cache.clear()
                bm25._clear()
                st.success("知识库已清空。")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.error("再次点击按钮确认清空。此操作不可撤销！")


def _delete_source(source_name: str):
    """删除指定文档来源"""
    try:
        count = store.delete_by_source(source_name)
        cache.clear()
        bm25.rebuild_from_store()
        st.success(f"已删除 '{source_name}'（{count} 个知识块）")
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")
