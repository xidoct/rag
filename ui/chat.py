"""
知识问答页面 — 聊天界面，学生提问关于毕业论文的问题
"""

import streamlit as st

from engine.retriever import retriever


def render():
    """渲染问答页面"""
    st.title("💬 毕业论文知识问答")
    st.markdown("向我提问关于毕业论文的任何问题，我会基于学校官方文档为你解答。")

    # --- 提示示例 ---
    with st.expander("💡 点击查看示例问题"):
        st.markdown("""
        - 毕业论文的查重率要求是多少？
        - 论文格式有什么具体要求（字体、行距、页边距）？
        - 毕业论文的开题报告截止日期是哪天？
        - 论文答辩需要准备什么材料？
        - 参考文献的引用格式要求是什么？
        - 字数要求是多少？有没有最少页数限制？
        """)

    # --- 初始化聊天记录 ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是毕业论文知识问答助手 👋\n\n你可以问我关于毕业论文的任何问题，比如查重率要求、格式规范、截止日期、答辩流程等。我会严格依据已上传的学校文档为你解答。"}
        ]

    # --- 渲染历史消息 ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 输入框 ---
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            _stream_answer(prompt)


def _stream_answer(question: str):
    """流式生成并展示回答 + 引用来源"""
    try:
        # 使用流式 API
        stream = retriever.ask_stream(question)

        # 第一个 yield 是检索结果
        first = next(stream)
        _, hits = first

        # 展示检索来源（折叠）
        if hits:
            _show_sources(hits)

        # 流式显示回答
        response_container = st.empty()
        full_response = ""

        for delta, _ in stream:
            full_response += delta
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

        # 保存到历史
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    except Exception as e:
        error_msg = f"❌ 出错了: {str(e)}\n\n请确保:\n1. 已配置 API Key\n2. 已上传文档到知识库\n3. 网络连接正常"
        st.error(error_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )


def _show_sources(hits: list[dict]):
    """显示检索到的来源文档"""
    with st.expander(f"📚 参考来源 ({len(hits)} 个相关片段)", expanded=False):
        for i, hit in enumerate(hits, 1):
            heading = f" > {hit['heading']}" if hit.get("heading") else ""
            st.markdown(
                f"**{i}.** `{hit['source']}` (第{hit['page']}页{heading}) "
                f"— 相关度: {hit['similarity']:.2%}"
            )
