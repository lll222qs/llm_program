import chromadb
from rank_bm25 import BM25Okapi
import warnings
import logging

# 关闭 ChromaDB 遥测警告
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 1. 连接数据库
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name="company_docs")

# 2. 获取所有文档（用于 BM25 关键词检索）
all_docs = collection.get()['documents']

# 3. 构建 BM25 关键词检索器
def tokenize(text):
    return list(text)

tokenized_docs = [tokenize(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, top_k=10):
    """BM25 关键词检索，返回 (doc, score) 列表"""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_k]
    return [(all_docs[i], scores[i]) for i in top_indices]

def vector_search(query, top_k=10):
    """向量检索，返回 (doc, distance) 列表"""
    results = collection.query(
        query_texts=[query], n_results=top_k,
        include=["documents", "distances"]
    )
    return [(doc, dist) for doc, dist in zip(results['documents'][0], results['distances'][0])]

def rrf_fusion(vector_results, bm25_results, k_rrf=60, top_k=5):
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    
    核心思想：
    - 每个结果根据排名获得分数：score = 1 / (rank + k)
    - 如果一个文档在两个检索器中都出现，分数累加
    - 最终按总分排序，取 top_k
    
    k 值越大，排名的影响越小（k=60 是业界常用值）
    """
    score_map = {}  # doc -> total_score
    doc_map = {}    # doc -> full_content (去重用)
    
    # 向量检索结果（语义匹配）
    for rank, (doc, dist) in enumerate(vector_results, 1):
        key = hash(doc)  # 用哈希去重
        score = 1.0 / (rank + k_rrf)
        score_map[key] = score_map.get(key, 0) + score
        doc_map[key] = doc
    
    # BM25 关键词检索结果（字面匹配）
    for rank, (doc, score) in enumerate(bm25_results, 1):
        key = hash(doc)
        # BM25 的原始分数也加上去，作为加权因子
        rrf_score = 1.0 / (rank + k_rrf)
        score_map[key] = score_map.get(key, 0) + rrf_score
        doc_map[key] = doc
    
    # 按融合分数排序
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:top_k]]

# ==================== 测试 ====================

def test_query(query):
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")
    
    # 纯向量检索
    print("\n【纯向量检索（语义匹配）】")
    v_results = vector_search(query, top_k=3)
    for i, (doc, dist) in enumerate(v_results):
        print(f"[{i+1}] (距离:{dist:.4f}) {doc[:200]}...")
    
    # 纯 BM25 检索
    print("\n【纯 BM25 关键词检索】")
    b_results = bm25_search(query, top_k=3)
    for i, (doc, score) in enumerate(b_results):
        print(f"[{i+1}] (分数:{score:.4f}) {doc[:200]}...")
    
    # 混合检索（RRF 融合）
    print("\n【RRF 混合检索（向量 + BM25 融合）】")
    v_all = vector_search(query, top_k=5)
    b_all = bm25_search(query, top_k=5)
    merged = rrf_fusion(v_all, b_all, k_rrf=60, top_k=5)
    for i, doc in enumerate(merged):
        print(f"[{i+1}] {doc[:200]}...")

# 4. 自动测试几个问题
test_queries = ["学校简介", "学校简介是什么", "清华大学简介"]
for q in test_queries:
    test_query(q)

# 5. 交互式测试
print("\n" + "="*60)
print("请输入你想测试的问题（输入 quit 退出）:")
print("="*60)
while True:
    user_query = input("\n请输入问题: ").strip()
    if user_query.lower() == 'quit':
        break
    test_query(user_query)