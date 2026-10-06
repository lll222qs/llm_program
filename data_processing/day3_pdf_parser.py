import pymupdf  # 【修复1】：使用新版导入，消除 fitz 的废弃警告
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import re  # 在文件开头确保导入了正则表达式模块


# ================= 配置区 =================
PDF_PATH = "test_doc.pdf"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
DB_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"
# =========================================

# 【修复2】：在全局初始化 ChromaDB 客户端，避免作用域问题
client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False)
)

def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 提取 PDF 中的所有文本"""
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"❌ PDF 读取失败: {e}")
        return ""


def clean_text(text):
    """清洗 PDF 提取出的文本，去除页眉、页脚、页码等干扰信息"""
    # 1. 去除独立的纯数字页码（比如单独一行的 47, 58 等）
    text = re.sub(r'\n\d{1,3}\n', '\n', text)
    
    # 2. 去除常见的页脚网址和版权声明
    text = re.sub(r'www\.tsinghua\.edu\.cn', '', text)
    text = re.sub(r'Tsinghua University', '', text)
    text = re.sub(r'2023 清华概览', '', text)
    text = re.sub(r'编辑出版：清华大学校长办公室', '', text)
    text = re.sub(r'电话：010-\d+', '', text)
    text = re.sub(r'传真：010-\d+', '', text)
    text = re.sub(r'网址：http[s]?://[^\s]+', '', text)
    
    # 3. 去除多余的空行（把连续3个以上的换行符替换为2个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def chunk_text(text):
    """使用 LangChain 进行语义智能切分文本"""
    text_splitter = RecursiveCharacterTextSplitter(
        # 【核心改动】：定义切分优先级！
        # 1. 优先按双换行（段落）切分
        # 2. 其次按单换行切分
        # 3. 再次按中文句号、感叹号、问号切分
        # 4. 最后才按英文标点切分
        # 5. 万不得已，才按空格或字符硬切
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        chunk_size=400,        # 【优化】稍微调大一点，保证语义完整性
        chunk_overlap=80,      # 【优化】增加重叠，防止关键信息在边界丢失
        length_function=len,   # 按字符长度计算
    )
    return text_splitter.split_text(text)

def save_to_chromadb(chunks):
    """将切分后的文本块存入 ChromaDB，并附带元数据"""
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
        
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 【核心改动】：给每个文本块生成一个来源标签
    # 实际生产中，这里可以提取 PDF 的标题或页码
    metadatas = [{"source": "清华大学2023概览.pdf"} for _ in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        ids=[f"pdf_chunk_{i}" for i in range(len(chunks))],
        metadatas=metadatas  # 【新增】：把标签一起存入数据库
    )
    return len(chunks)

if __name__ == "__main__":
    print(f"📖 正在读取 PDF: {PDF_PATH} ...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    
    if not raw_text:
        print("⚠️ 没有提取到任何文本，请检查 PDF 是否为纯文本格式（非扫描件）。")
    else:
        print(f"✅ 成功提取 {len(raw_text)} 个字符！")
        
        # 【新增】：进行文本清洗
        print("🧹 正在进行文本清洗（去除页眉页脚）...")
        raw_text = clean_text(raw_text)
        print(f"✅ 清洗完成，剩余 {len(raw_text)} 个字符！")
        
        print("🔪 正在进行智能切分...")
        chunks = chunk_text(raw_text)
        # ... 后面的代码保持不变
        print(f"✅ 文本被切分成了 {len(chunks)} 个片段！")
        
        print("📦 正在写入向量数据库...")
        count = save_to_chromadb(chunks)
        print(f"🎉 成功将 {count} 个片段存入 [{COLLECTION_NAME}]！")
        
        # 【修复3】：现在可以直接使用全局的 client 进行测试了
        test_collection = client.get_or_create_collection(name=COLLECTION_NAME)
        test_query = input("\n🔍 请输入一个问题测试检索: ")
        results = test_collection.query(query_texts=[test_query], n_results=2)
        
        print("\n📚 数据库找到的相关依据：")
        for i, doc in enumerate(results['documents'][0]):
            print(f"[{i+1}] {doc[:100]}...")