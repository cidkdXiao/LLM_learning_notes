# Few shot Example
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag_tools.define_retriever import retriever

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal background?",
    },
]

# Now transform these into message prompts
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """你是一位世界领域的专家。你的任务是退一步，将问题改写为更通用的、便于回答的“退一步”问题。以下是一些示例：""",),
        # Few shot examples
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

question = "What is task decomposition for LLM agents?"
generated_query = generate_queries_step_back.invoke({"question": question})

# Response prompt
response_prompt_template = """
你是一位世界知识领域的专家。我将向你提问一个问题。你的回答应当全面，并且在相关情况下不得与以下内容矛盾。如果这些内容与问题无关，请忽略它们。

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:
"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(
            lambda x: "Normal context for: " + x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

chain.invoke({"question": question})