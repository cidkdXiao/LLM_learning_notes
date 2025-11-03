from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import bs4

# 1) Load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content","post_title","post-header")))
)
docs = loader.load()

# 2) Split (token-aware)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400,
    chunk_overlap=60
)
splits = splitter.split_documents(docs)

# 3) Index (persist)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=emb,
    persist_directory="./chroma_db"
)
vectorstore.persist()

# 4) Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 示例：检索最相近的片段
# docs = retriever.get_relevant_documents("What is an autonomous agent?")
# for d in docs:
#     print(d.metadata.get("source"), d.page_content[:120], "...")


# 5) Multi Query: Different Perspective
# Generate multiple query reformulations from different perspectives
template = """
你是一款 AI 语言模型助手。你的任务是为用户提供的问题生成五个不同版本的改写，以便从向量数据库中检索相关文档。通过从多个角度改写用户问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。请将这些改写问题用换行符分隔。
原始问题：{question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(template)
post_clean = RunnableLambda(
    lambda lines: [s.strip(" •-*0123456789.").strip() for s in lines if s.strip()]
)

def ensure_five(lines):
    lines = [l for l in lines if l.strip()]
    return (lines + lines[:5])[:5]  # 不足则循环补齐

generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: ensure_five(x.splitlines()))
    | post_clean
)


# 6) Retrieval QA with Multi-Query
# def get_unique_union(documents: list[list]):
#     """Unique union of retrieved docs"""
#     # Flatten list of lists, and convert each Document to string
#     flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
#     # Get unique documents
#     unique_docs = list(set(flattened_docs))
#     # Return
#     return [loads(doc) for doc in unique_docs]

def unique_union_ordered(doc_lists: list[list]):
    seen = set()
    merged = []
    for sub in doc_lists:                 # 保留先出现的优先级
        for d in sub:
            key = (d.page_content,)       # 仅按内容去重，或加 source 等
            if key not in seen:
                seen.add(key)
                merged.append(d)
    return merged

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | unique_union_ordered
docs = retrieval_chain.invoke({"question":question})


# 7) Responding with Retrieved Context
# RAG
QA_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(QA_template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question": question})