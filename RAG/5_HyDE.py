from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag_tools.define_retriever import retriever

# HyDE document generation
template = """请撰写一段科学论文内容来回答以下问题。
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

llm = ChatOpenAI(temperature=0)

# Run
question = "LLM代理的任务分解是什么？"
generate_docs_for_retrieval.invoke({"question": question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
retrieved_docs

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context": retrieved_docs, "question": question})
