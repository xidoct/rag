"""
文档上传页面 — 两种导入方式:
1. 浏览器上传 (适合小文件, <200MB)
2. 本地文件夹导入 (适合大批量/大文件, 无大小限制)
"""

from pathlib import Path

import streamlit as st

from engine.loader import load_file, scan_folder, load_folder
from engine.splitter import split_documents
from engine.embedder import embedder
from engine.store import store
from engine.bm25_retriever import bm25
from engine.cache import cache
import config


def render():
    """渲染上传页面"""
    st.title("📤 文档导入")
    st.markdown("将学校关于毕业论文的 PDF、Word 文档导入知识库。")

    _show_current_stats()

    # --- 两种导入方式 ---
    tab1, tab2 = st.tabs(["📁 本地文件夹导入（推荐大文件）", "🌐 浏览器上传（小文件）"])

    with tab1:
        _render_folder_import()

    with tab2:
        _render_browser_upload()


def _show_current_stats():
    """显示知识库当前状态"""
    try:
        stats = store.get_stats()
    except Exception:
        stats = {"total_files": 0, "total_chunks": 0}

    cols = st.columns(3)
    cols[0].metric("📄 文档数", stats.get("total_files", 0))
    cols[1].metric("🧩 知识块", stats.get("total_chunks", 0))
    cols[2].metric("📊 表格块", stats.get("table_chunks", 0))


# ============================================================
#  方式一: 本地文件夹导入（无大小限制）
# ============================================================

def _render_folder_import():
    st.subheader("📁 从本地文件夹导入")
    st.markdown(
        "将 PDF/Word 文件放到一个文件夹里，输入路径即可批量导入。"
        "**不走浏览器，无文件大小限制**，适合大批量文档。"
    )

    col1, col2 = st.columns([3, 1])
    folder_path = col1.text_input(
        "文件夹路径",
        placeholder="例如: D:\\论文资料\\2024年规范\\",
        help="系统会递归扫描该文件夹下的所有 PDF 和 Word 文件",
        key="folder_path",
    )
    auto_scan = col2.checkbox("自动扫描已解压的 ZIP", value=True,
                              help="如果之前已经解压过 ZIP，勾选后系统自动识别")

    if folder_path:
        _do_folder_scan(folder_path)


def _do_folder_scan(folder_path: str):
    """扫描文件夹并显示文件列表"""
    try:
        files = scan_folder(folder_path)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except NotADirectoryError as e:
        st.error(str(e))
        return

    if not files:
        st.warning(f"在 `{folder_path}` 中未找到 PDF/Word 文件。")
        return

    # 显示扫描结果
    file_count = len(files)
    # 估算总大小
    total_size = sum(f.stat().st_size for f in files)
    total_size_mb = total_size / (1024 * 1024)

    st.success(f"🔍 扫描到 **{file_count}** 个文件，共约 {total_size_mb:.1f} MB")

    # 只显示前 20 个文件名，其余折叠
    with st.expander(f"📋 查看文件列表 ({file_count} 个)", expanded=False):
        for f in files[:50]:
            size_kb = f.stat().st_size / 1024
            st.caption(f"  • {f.name} ({size_kb:.0f} KB)")
        if file_count > 50:
            st.caption(f"  ... 还有 {file_count - 50} 个文件")

    # 导入按钮
    if st.button("🚀 开始导入全部文件", type="primary", use_container_width=True, key="btn_folder"):
        _process_folder(files)


def _process_folder(files: list[Path]):
    """批量处理文件夹中的文档 → 分块 → 向量化 → 入库"""
    total_files = len(files)
    progress_bar = st.progress(0, text=f"准备处理 {total_files} 个文件...")
    status = st.status(f"处理文档 (共 {total_files} 个)...", expanded=True)
    log_placeholder = st.empty()

    all_chunks = []
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(files):
        file_name = file_path.name
        status.update(label=f"[{i+1}/{total_files}] {file_name}")

        try:
            # 解析
            documents = load_file(file_path)
            chunks = split_documents(documents)
            all_chunks.extend(chunks)
            success_count += 1
            status.write(f"  ✅ {file_name} → {len(chunks)} 块")

        except Exception as e:
            fail_count += 1
            status.write(f"  ❌ {file_name} — {e}")
            continue

        progress_bar.progress(
            (i + 1) / total_files,
            text=f"📖 解析: {i+1}/{total_files} (成功 {success_count}, 失败 {fail_count})"
        )

    if not all_chunks:
        status.update(label="没有可处理的内容", state="error")
        return

    # 向量化
    status.update(label=f"🧮 向量化 {len(all_chunks)} 个知识块...")
    progress_bar.progress(0.85, text="🧮 正在向量化...")

    try:
        texts = [chunk.content for chunk in all_chunks]
        batch_size = 50
        all_embeddings = []

        for j in range(0, len(texts), batch_size):
            batch = texts[j:j + batch_size]
            embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
            progress_bar.progress(
                0.85 + 0.1 * (j / len(texts)),
                text=f"🧮 向量化: {min(j + batch_size, len(texts))}/{len(texts)}"
            )

        # 入库
        progress_bar.progress(0.98, text="💾 写入知识库...")
        count = store.add_documents(all_chunks, all_embeddings)

        progress_bar.progress(1.0, text="✅ 完成！")
        status.update(
            label=f"✅ 导入完成！{success_count} 个文件 → {count} 个知识块",
            state="complete"
        )
        cache.clear()
        bm25.rebuild_from_store()
        st.success(f"🎉 成功导入 {count} 个知识块！切换到「知识问答」开始提问。")
        st.rerun()

    except Exception as e:
        status.update(label="向量化失败", state="error")
        st.error(f"向量化出错: {e}")


# ============================================================
#  方式二: 浏览器上传（小文件）
# ============================================================

def _render_browser_upload():
    st.subheader("🌐 浏览器上传")
    st.markdown("直接上传 PDF、Word 或 ZIP 文件。单文件限制 200MB，适合小批量。")
    st.info("💡 如果文件超过 200MB 或数量很多，请使用左侧「本地文件夹导入」。")

    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "doc", "zip"],
        accept_multiple_files=True,
        help="支持 PDF、Word (.docx)、ZIP 压缩包。",
        key="browser_uploader",
    )

    if uploaded_files:
        _process_browser_uploads(uploaded_files)


def _process_browser_uploads(uploaded_files):
    """处理浏览器上传的文件"""
    if st.button("🚀 开始处理上传文件", type="primary", use_container_width=True, key="btn_upload"):
        progress_bar = st.progress(0, text="准备中...")
        status = st.status("处理文档...", expanded=True)

        all_chunks = []
        total_files = len(uploaded_files)

        for i, uf in enumerate(uploaded_files):
            file_name = uf.name
            status.update(label=f"[{i+1}/{total_files}] {file_name}")

            temp_path = config.UPLOAD_DIR / file_name
            temp_path.write_bytes(uf.read())

            try:
                documents = load_file(temp_path)
                chunks = split_documents(documents)
                all_chunks.extend(chunks)
                status.write(f"  ✅ {file_name} → {len(chunks)} 块")
            except Exception as e:
                status.write(f"  ❌ {file_name} — {e}")
                continue

            progress_bar.progress((i + 1) / total_files, text=f"📖 {file_name}")

        if not all_chunks:
            status.update(label="没有可处理的内容", state="error")
            return

        _do_vectorize_and_store(all_chunks, progress_bar, status, total_files)


def _do_vectorize_and_store(all_chunks, progress_bar, status, file_count):
    """向量化 + 入库的公共逻辑"""
    texts = [chunk.content for chunk in all_chunks]
    status.update(label=f"🧮 向量化 {len(all_chunks)} 个知识块...")

    try:
        batch_size = 50
        all_embeddings = []
        for j in range(0, len(texts), batch_size):
            batch = texts[j:j + batch_size]
            embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
            progress_bar.progress(
                0.8 + 0.15 * (j / len(texts)),
                text=f"🧮 向量化: {min(j + batch_size, len(texts))}/{len(texts)}"
            )

        progress_bar.progress(0.98, text="💾 写入知识库...")
        count = store.add_documents(all_chunks, all_embeddings)

        progress_bar.progress(1.0, text="✅ 完成！")
        status.update(
            label=f"✅ 完成！{count} 个知识块入库",
            state="complete"
        )
        cache.clear()
        bm25.rebuild_from_store()
        st.success(f"🎉 成功入库 {count} 个知识块！")
        st.rerun()

    except Exception as e:
        status.update(label="向量化失败", state="error")
        st.error(f"向量化出错: {e}")
