# 功能说明：将企业知识库PDF文件进行向量计算并持久化存储到向量数据库（chroma）
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from utils import pdfSplitTest_Ch
from utils import pdfSplitTest_En

# 加载 .env 文件
load_dotenv()

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 嵌入模型配置 — 从环境变量读取
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

ONEAPI_API_BASE = os.getenv("ONEAPI_API_BASE", "")
ONEAPI_EMBEDDING_API_KEY = os.getenv("ONEAPI_API_KEY", "")
ONEAPI_EMBEDDING_MODEL = os.getenv("ONEAPI_EMBEDDING_MODEL", "text-embedding-v1")

QWen_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWen_EMBEDDING_API_KEY = os.getenv("QWEN_API_KEY", "")
QWen_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
OLLAMA_EMBEDDING_API_KEY = "ollama"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")


# 优先使用 EMBEDDING_PROVIDER，否则回退到 LLM_TYPE
llmType = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_TYPE", "ollama"))
if llmType == "deepseek":
    logger.info("DeepSeek 无嵌入模型，使用 Ollama 嵌入")
    llmType = "ollama"

# 设置测试文本类型 Chinese 或 English
TEXT_LANGUAGE = 'Chinese'
INPUT_PDF = "input/裕兴集团员工手册.pdf"

# 指定文件中待处理的页码，全部页码则填None
PAGE_NUMBERS=None

# 指定向量数据库chromaDB的存储位置和集合 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = "chromaDB"
CHROMADB_COLLECTION_NAME = "yuxing_handbook"


# get_embeddings方法计算向量
def get_embeddings(texts):
    global llmType
    global ONEAPI_API_BASE, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    global QWen_API_BASE, QWen_EMBEDDING_API_KEY, QWen_EMBEDDING_MODEL
    global OLLAMA_API_BASE, OLLAMA_EMBEDDING_API_KEY, OLLAMA_EMBEDDING_MODEL
    if llmType == 'oneapi':
        try:
            client = OpenAI(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=ONEAPI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'qwen':
        try:
            client = OpenAI(
                base_url=QWen_API_BASE,
                api_key=QWen_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=QWen_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'ollama':
        try:
            client = OpenAI(
                base_url=OLLAMA_API_BASE,
                api_key=OLLAMA_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OLLAMA_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    else:
        try:
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []


# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        response = get_embeddings(batch)
        results.extend(response)
    return results


# 封装向量数据库chromadb类，提供两种方法
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        global CHROMADB_DIRECTORY
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[str(uuid.uuid4()) for i in range(len(documents))]
        )

    def search(self, query, top_n):
        try:
            results = self.collection.query(
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.info(f"检索向量数据库时出错: {e}")
            return []


# 封装文本预处理及灌库方法  提供外部调用
def vectorStoreSave():
    global TEXT_LANGUAGE, CHROMADB_COLLECTION_NAME, INPUT_PDF, PAGE_NUMBERS

    if TEXT_LANGUAGE == 'Chinese':
        paragraphs = pdfSplitTest_Ch.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        vector_db.add_documents(paragraphs)
        user_query = "裕兴集团的考勤制度是什么"
        search_results = vector_db.search(user_query, 5)
        logger.info(f"检索向量数据库的结果: {search_results}")

    elif TEXT_LANGUAGE == 'English':
        paragraphs = pdfSplitTest_En.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        vector_db.add_documents(paragraphs)
        user_query = "deepseek-r1性能如何？"
        search_results = vector_db.search(user_query, 5)
        logger.info(f"检索向量数据库的结果: {search_results}")


if __name__ == "__main__":
    vectorStoreSave()
