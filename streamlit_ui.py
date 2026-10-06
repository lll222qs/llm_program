import streamlit as st
import requests

# 配置页面
st.set_page_config(page_title="清华知识库问答", page_icon="🎓")
st.title("🎓 清华大学知识库问答系统")
st.caption("基于 RAG + Rerank 技术构建，支持引用溯源")

# 后端 API 地址
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# 初始化会话状态中的聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 处理用户输入
if prompt := st.chat_input("请输入问题，例如：清华大学建于哪一年？"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 存入历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 调用后端 API
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={
                        "messages": st.session_state.messages,
                        "model": "qwen2.5:3b-instruct-q8_0",
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                ai_reply = response.json()["choices"][0]["message"]["content"]
                
                # 显示 AI 回复
                st.markdown(ai_reply)
                
                # 存入历史
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            # 修改 streamlit_ui.py 中的 except 部分
            except Exception as e:
                # 【调试用】打印出原始响应内容
                if 'response' in locals():
                    st.error(f"状态码: {response.status_code}")
                    st.error(f"原始响应: {response.text[:500]}") # 只看前500字
                else:
                    st.error(f"连接失败: {str(e)}")