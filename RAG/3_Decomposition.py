from typing import List
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from rag_tools.define_retriever import retriever

# -------- 0. 工具函数 --------
def format_docs(docs: List[Document], max_docs=6, max_chars_per_doc=800, sep="\n\n---\n\n") -> str:
    chunks = []
    for d in docs[:max_docs]:
        text = (d.page_content or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + " ..."
        chunks.append(text)
    return sep.join(chunks)

def format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"

# -------- 1. 组合提示（修正拼写）--------
template = """
Here is the question you need to answer:
\n---\n{question}\n---\n
Here are any available background question + answer pairs:
\n---\n{q_a_pairs}\n---\n
Here is additional context relevant to the question:
\n---\n{context}\n---\n
Use ONLY the above context and background Q/A pairs to answer the question:
{question}
If the answer is not in the context, say "I don't know".
"""
decomposition_prompt = ChatPromptTemplate.from_template(template)

# -------- 2. RAG 基础组件 --------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt_rag = hub.pull("rlm/rag-prompt")  # 这个模板一般需要 {context} 是文本

# -------- 2.1 子问题生成链 --------
sub_question_template = """
你是一名问题分解助手。你的任务是把用户的问题拆解为几个可以独立回答的子问题。

要求：
1. 子问题应覆盖原问题的全部关键点。
2. 每个子问题尽量具体、可检索、无二义性。
3. 子问题之间逻辑独立但互相关联。
4. 输出格式：每行一个子问题，不要编号或其他符号。

原始问题：
{question}
"""
sub_question_prompt = ChatPromptTemplate.from_template(sub_question_template)
sub_question_generator_chain = (
    sub_question_prompt
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.splitlines() if q.strip()][:6])  # 上限 6 条
)

def retrieval_and_rag(question: str, prompt_rag, sub_question_generator_chain):
    """对每个子问题执行一次：检索 -> RAG 回答"""
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    rag_answers = []

    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ctx_text = format_docs(docs)  # ★ 把 Document 列表转为可读上下文文本
        answer = (prompt_rag | llm | StrOutputParser()).invoke({
            "context": ctx_text,
            "question": sq
        })
        rag_answers.append(answer)

    return rag_answers, sub_questions

# 调用
question = "What is task decomposition for LLM agents?"
answers, questions = retrieval_and_rag(question, prompt_rag, sub_question_generator_chain)

# -------- 3. 迭代累积 Q/A，再次用组合提示做“汇总式”回答 --------
q_a_pairs = ""
for q, a in zip(questions, answers):
    q_a_pairs = (q_a_pairs + "\n---\n" if q_a_pairs else "") + format_qa_pair(q, a)

# 最后一轮：把所有子问题的 Q/A + 针对“总问题”的检索上下文，一起总结成最终答案
final_docs = retriever.get_relevant_documents(question)
final_context = format_docs(final_docs)

final_chain = (
    decomposition_prompt
    | llm
    | StrOutputParser()
)
final_answer = final_chain.invoke({
    "question": question,
    "q_a_pairs": q_a_pairs,
    "context": final_context
})

print(final_answer)
