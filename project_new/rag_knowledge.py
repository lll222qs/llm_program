
# rag_knowledge.py
# RAG 知识库模块：本地 Embedding + Qdrant 向量数据库
# 【修复】模型加载改为懒加载，避免模块导入时崩溃

import os
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, models
from config import settings
from langchain_core.tools import tool 


# ================= 1. 知识库内容定义 =================
COLLECTION_NAME = "steel_manual"

manual_content = [
    "crazing (裂纹): 表面出现细小裂纹。建议：检查轧辊温度，降低冷却水压力，必要时更换轧辊。",
    "inclusion (夹杂): 表面有非金属夹杂物。建议：检查钢水纯净度，清理中间包，优化保护渣性能。",
    "patches (斑块): 表面出现不规则斑块。建议：检查除鳞系统是否正常工作，调整除鳞水压。",
    "pitted_surface (麻点): 表面有密集小坑。建议：检查酸洗工艺参数，增加酸洗时间或提高酸液浓度。",
    "rolled-in_scale (氧化皮): 氧化皮压入表面。建议：检查加热炉温度均匀性，优化除鳞喷嘴角度。",
    "scratches (划痕): 表面有线性划痕。建议：检查导卫装置是否磨损，清理传输辊道异物。"
]

# ================= 2. 本地 Embedding 模型（懒加载）==================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "text2vec-base-chinese")

# 模块级变量，延迟初始化
_embedding_model = None
_model_load_error = None


def _load_embedding_model():
    """懒加载 Embedding 模型，首次调用时加载"""
    global _embedding_model, _model_load_error
    
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        word_embedding_model = models.Transformer(
            MODEL_PATH,
            max_seq_length=256,
            model_args={"ignore_mismatched_sizes": True}
        )
        
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        _embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(f"[RAG] Embedding 模型加载成功: {MODEL_PATH}")
        return _embedding_model
        
    except Exception as e:
        _model_load_error = e
        print(f"[RAG] 警告：Embedding 模型加载失败: {e}")
        print(f"[RAG] 模型路径: {MODEL_PATH}")
        print(f"[RAG] 请确保已下载 text2vec-base-chinese 模型到对应路径")
        raise


class LocalEmbeddings:
    """LangChain 兼容的本地 Embedding 包装器"""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# ================= 3. Qdrant 连接与初始化 =================
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

_vectorstore = None


def _get_vectorstore():
    """
    懒加载向量数据库连接 + 自动初始化数据
    - 首次启动：Collection 不存在 -> 创建并写入数据
    - 后续启动：Collection 已存在 -> 直接复用，不重复写入
    """
    global _vectorstore
    
    if _vectorstore is not None:
        return _vectorstore
    
    # 先加载模型（如果还没加载）
    try:
        model = _load_embedding_model()
    except Exception as e:
        print(f"[RAG] 模型加载失败，跳过 Qdrant 初始化: {e}")
        return None
    
    try:
        client = QdrantClient(
            host=settings.VECTOR_DB_HOST,
            port=settings.VECTOR_DB_PORT,
        )
        
        # 检查 Collection 是否已存在
        collections = [c.name for c in client.get_collections().collections]
        
        if COLLECTION_NAME not in collections:
            print(f"[RAG] Collection '{COLLECTION_NAME}' 不存在，正在初始化...")
            docs = [Document(page_content=c) for c in manual_content]
            vectorstore = QdrantVectorStore.from_documents(
                documents=docs,
                embedding=LocalEmbeddings(model),
                collection_name=COLLECTION_NAME,
                client=client,
            )
            print(f"[RAG] 成功写入 {len(docs)} 条维修手册记录到 Qdrant")
        else:
            print(f"[RAG] Collection '{COLLECTION_NAME}' 已存在，跳过初始化")
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=LocalEmbeddings(model),
            )
        
        _vectorstore = vectorstore
        return vectorstore
        
    except Exception as e:
        print(f"[RAG] Qdrant 连接失败: {e}")
        print(f"[RAG] 请确保 Qdrant 服务正在运行于 {settings.VECTOR_DB_HOST}:{settings.VECTOR_DB_PORT}")
        return None


# ================= 4. 检索接口（对外暴露）=================
@tool
def search_manual(query: str) -> str:
    """
    当用户询问关于工业缺陷的维修建议、处置流程、操作规范或技术参数时调用此工具。
    输入参数 query 应该是具体的缺陷名称（如 'inclusion', '划痕', '裂纹'）或故障现象。
    不要直接回答维修建议，必须先调用此工具获取权威文档内容。
    """
    try:
        # 这里的逻辑保持不变，利用你现有的 _get_vectorstore
        store = _get_vectorstore() 
        
        if store is None:
            # 注意：在 Docker 环境中，如果 Qdrant 没起，这里会返回兜底数据
            # 这正好符合你"先集成逻辑，后启动依赖"的策略
            return (
                "[系统提示] 知识库服务暂时不可用，使用本地兜底策略：\n"
                "- 针对一般性缺陷：建议停机检查，拍照记录，并联系高级工程师。\n"
                "- 针对 Inclusion (夹杂)：参考标准流程 - 1.确认位置 2.评估深度 3.打磨或更换。\n"
                "- 针对 Crack (裂纹)：严禁运行，必须立即更换部件。"
            )

        # 设置检索参数，k=2 表示取最相关的两条，给 LLM 更多上下文
        retriever = store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(query)

        if not docs:
            return f"未在知识库中检索到关于 '{query}' 的相关维修建议。"

        # 将检索到的文档内容拼接成字符串返回给 LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"[知识库检索结果]:\n{context}"

    except Exception as e:
        # 捕获异常并返回错误信息，防止整个 Agent 崩溃
        return f"[RAG工具执行出错]: {str(e)}"