import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 这就是解决问题的魔法
import sys, streamlit as st
st.write("Python:", sys.executable)

# app.py
import streamlit as st
import os
from core.rag import init_rag, ask_question
from core.ticket import create_ticket
from core.feedback import save_feedback
from core.stats import get_ticket_stats

st.title("🏢 企业IT智能助手（架构分离版）")

# 状态
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_q" not in st.session_state:
    st.session_state.last_q = None
if "last_a" not in st.session_state:
    st.session_state.last_a = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

# 上传
uploaded = st.file_uploader("上传IT文档", type=["pdf", "txt"])
if uploaded:
    tmp = f"./tmp.{uploaded.name.split('.')[-1]}"
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("构建知识库"):
        with st.spinner("处理中..."):
            try:
                st.session_state.chain, st.session_state.retriever = init_rag(tmp)
                st.success("? 知识库构建完成")
            except Exception as e:
                st.error(f"? 构建失败，错误详情：{str(e)}")
                st.stop()  # 遇到错误就停止后续运行，避免卡住

# 提问
q = st.text_input("问题：")
if st.button("发送") and q and st.session_state.chain:
    with st.spinner("检索中..."):
        a, sources = ask_question(st.session_state.chain, st.session_state.retriever, q)
        st.session_state.last_q = q
        st.session_state.last_a = a
        st.session_state.last_sources = sources

# 显示回答
if st.session_state.last_a:
    st.subheader("💡 回答")
    st.write(st.session_state.last_a)

    st.subheader("📄 来源")
    for idx, s in enumerate(st.session_state.last_sources[:2]):
        st.info(f"片段 {idx+1}: {s.page_content[:300]}")

    # 无结果 → 工单
    if not st.session_state.last_sources:
        st.warning("⚠️ 未找到答案，已自动创建工单")
        create_ticket(st.session_state.last_q)

    # 反馈
    st.subheader("评价")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("有用 👍"):
            save_feedback(st.session_state.last_q, st.session_state.last_a, "useful")
            st.success("已记录")
    with c2:
        if st.button("无用 👎"):
            save_feedback(st.session_state.last_q, st.session_state.last_a, "useless")
            st.info("已记录")

# 统计面板
st.divider()
st.subheader("📊 IT管理面板")
stats = get_ticket_stats()
st.write("工单总数：", stats["total"])
st.write("最近无法回答：", stats["recent"])