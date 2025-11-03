from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from rag_tools.define_retriever import retriever
from rag_tools.multi_query import generate_queries


# 6) Retrieval QA with Multi-Query
def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# def unique_union_ordered(doc_lists: list[list]):
#     seen = set()
#     merged = []
#     for sub in doc_lists:                 # 保留先出现的优先级
#         for d in sub:
#             key = (d.page_content,)       # 仅按内容去重，或加 source 等
#             if key not in seen:
#                 seen.add(key)
#                 merged.append(d)
#     return merged

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# retrieval_chain = generate_queries | retriever.map() | unique_union_ordered
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