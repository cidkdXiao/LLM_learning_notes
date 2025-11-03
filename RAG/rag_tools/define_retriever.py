from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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