import os
# 在导入 chromadb 之前，通过环境变量关闭遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import httpx
import chromadb
from rank_bm25 import BM25Okapi
import uvicorn
from sentence_transformers import CrossEncoder
import time

app = FastAPI()

# ================= 多轮对话记忆管理 =================
# 这里我们用一个简单的字典来保存每个会话的历史记录
# 在实际生产中，通常会存入 Redis 或数据库
chat_histories = {}

def get_history(session_id: str):
    """获取指定会话的历史记录，如果没有则初始化"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]

# ================= RAG 核心组件初始化 =================
# 初始化重排序模型（第一次运行会自动下载，约 300MB）
print("🔄 正在加载重排序模型 bge-reranker-base...")
rerank_model = CrossEncoder('BAAI/bge-reranker-base')
print("✅ 重排序模型加载完成！")

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name="company_docs")

all_docs = collection.get()['documents']
tokenized_docs = [list(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:3b-instruct-q8_0"

# ================= 检索与融合算法 =================
def vector_search(query, top_k=8):
    """向量检索（语义匹配），同时获取元数据"""
    results = collection.query(
        query_texts=[query], 
        n_results=top_k, 
        include=["documents", "metadatas"]  # 【新增】：获取元数据
    )
    # 返回文档和对应的元数据
    return results['documents'][0], results['metadatas'][0]

def bm25_search(query, top_k=5):
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_k]
    return [all_docs[i] for i in top_indices]

def rrf_fusion(vector_results, bm25_results, k_rrf=60, top_k=8):
    """RRF 倒数排名融合算法（支持返回元数据）"""
    score_map = {}
    doc_map = {}
    meta_map = {}  # 【新增】：用来存储元数据
    
    # 处理向量检索结果（假设 vector_results 是 (docs, metas) 的元组）
    v_docs, v_metas = vector_results
    for rank, doc in enumerate(v_docs, 1):
        key = hash(doc)
        score_map[key] = score_map.get(key, 0) + 1.0 / (rank + k_rrf)
        doc_map[key] = doc
        meta_map[key] = v_metas[rank - 1]  # 【新增】：绑定元数据
        
    # 处理 BM25 检索结果（BM25 没有元数据，给个默认的）
    for rank, doc in enumerate(bm25_results, 1):
        key = hash(doc)
        score_map[key] = score_map.get(key, 0) + 1.0 / (rank + k_rrf)
        doc_map[key] = doc
        if key not in meta_map:  # 如果向量检索没捞到，给个默认来源
            meta_map[key] = {"source": "BM25关键词检索"}
            
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    
    # 【核心改动】：同时返回文本和对应的元数据
    top_docs = [doc_map[key] for key, _ in ranked[:top_k]]
    top_metas = [meta_map[key] for key, _ in ranked[:top_k]]
    return top_docs, top_metas

def rerank_results(query, docs, top_k=3):
    """使用 CrossEncoder 对文档进行精排序"""
    if not docs:
        return []
    
    # 构造 (query, doc) 配对
    pairs = [[query, doc] for doc in docs]
    
    # 模型打分
    scores = rerank_model.predict(pairs)
    
    # 按分数从高到低排序
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # 返回重排后的前 top_k 个文档
    return [docs[i] for i in ranked_indices[:top_k]]

# ================= RAG Prompt 构建 =================
def build_rag_prompt(user_query, context_docs, metadatas=None):
    """将检索到的资料和用户问题拼装成 RAG 标准 Prompt（带引用溯源）"""
    context = ""
    for i, doc in enumerate(context_docs):
        # 给每段资料加上编号和来源标签
        source = metadatas[i].get('source', '未知来源') if metadatas else '未知来源'
        context += f"\n\n---\n\n[来源 {i+1}: {source}]\n{doc}"
        
    system_prompt = """你是一个专业的企业知识库问答助手。
请严格根据下面提供的【参考资料】来回答用户的问题。

【回答原则】：
1. **直接给出答案**：如果资料中包含答案，请直接陈述事实，不要使用"推测"、"推断"、"大约"等不确定的词汇。
2. **同义词匹配**：如果资料中提到的是"前身"、"学堂"等与问题主体相关的历史名称，且时间明确，请直接视为有效答案。
3. **拒绝废话**：不要复述你的推理过程，不要说"根据资料显示"，直接说结果。
4. **兜底策略**：只有当所有资料完全无法支撑答案时，才回答"抱歉，知识库中没有找到相关信息"。

【引用要求】：
在回答的末尾，必须列出你参考了哪些资料的编号，格式为："📚 参考来源：[1], [2]"。"""
    
    user_prompt = f"""【参考资料】：
{context}

【用户问题】：
{user_query}"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ================= FastAPI 路由与流式输出 =================
class ChatRequest(BaseModel):
    messages: List[dict]
    stream: bool = True

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 【调试】打印收到的消息格式
    print(f"\n🔍 [DEBUG] 收到消息数量: {len(request.messages)}")
    if request.messages:
        print(f"🔍 [DEBUG] 最后一条消息类型: {type(request.messages[-1])}")
        print(f"🔍 [DEBUG] 最后一条消息内容: {request.messages[-1]}")
    
    # 1. 提取用户的最新提问
    user_query = request.messages[-1]["content"]
    
    # 2. RAG 检索流程 (保持不变)
    session_id = "default_session"
    history = get_history(session_id)
    
    v_results = vector_search(user_query, top_k=20) 
    b_results = bm25_search(user_query, top_k=20)   
    top_docs, top_metas = rrf_fusion(v_results, b_results, top_k=20)
    final_docs = rerank_results(user_query, top_docs, top_k=3)
    final_metas = top_metas[:3] 
    
    rag_messages = build_rag_prompt(user_query, final_docs, final_metas)
    full_messages = [rag_messages[0]] + history + [rag_messages[1]]
    
    # 【新增判断】如果前端请求非流式 (stream=False)
    if not request.stream:
        print("⚡ [DEBUG] 检测到非流式请求，正在同步生成完整回复...")
        
        collected_response = ""
        payload = {
            "model": MODEL_NAME,
            "messages": full_messages,
            "stream": False  # 告诉 Ollama 也不要流式输出，直接给结果
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as ac:
                resp = await ac.post(OLLAMA_URL, json=payload)
                data = resp.json()
                
                # 提取 Ollama 返回的完整内容
                collected_response = data['choices'][0]['message']['content']
                
            # 保存历史
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": collected_response})
            if len(history) > 10:
                history[:] = history[-10:]
                
            # 构造标准的 OpenAI 格式非流式响应
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": collected_response},
                    "finish_reason": "stop"
                }]
            }
        except Exception as e:
            print(f"❌ [ERROR] 非流式生成失败: {e}")
            return {"error": str(e)}

    # 【原有逻辑】如果前端请求流式 (stream=True)
    else:
        print("🌊 [DEBUG] 检测到流式请求，正在开启 StreamingResponse...")
        collected_response = "" 
        
        async def generate_stream():
            nonlocal collected_response
            payload = {
                "model": MODEL_NAME,
                "messages": full_messages,
                "stream": True
            }
            async with httpx.AsyncClient(timeout=60.0) as ac:
                async with ac.stream("POST", OLLAMA_URL, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                import json
                                chunk = json.loads(line[6:])
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    collected_response += content
                            except:
                                pass
                            yield line + "\n\n"
                            
        async def save_history_task():
            if collected_response:
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": collected_response})
                if len(history) > 10:
                    history[:] = history[-10:]
                print(f"💾 [DEBUG] 已保存对话，当前历史轮数: {len(history)//2}")

        from starlette.background import BackgroundTask
        background_task = BackgroundTask(save_history_task)
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream", background=background_task)

# 【关键修复】：显式启动 Uvicorn 服务器
if __name__ == "__main__":
    print("🚀 RAG 知识库服务正在启动...")
    uvicorn.run(app, host="127.0.0.1", port=8000)