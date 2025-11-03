from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

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