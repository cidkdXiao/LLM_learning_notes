"""
与 multi_query 类似，但将多个查询的结果使用 RRF (Retrieval Result Fusion) 方法进行融合，以提升检索效果。
"""
from collections import defaultdict
from langchain.load import dumps, loads

def reciprocal_rank_fusion(retrieved_docs_lists: list[list], k: int = 60) -> list:
    """
    使用 RRF (Reciprocal Rank Fusion) 方法融合多个查询的检索结果。

    :param retrieved_docs_lists:
        多个查询的检索结果列表，每个元素是一个查询的检索结果列表。
    :param k:
        RRF 中的常数参数，通常取值在 60 左右。
    :return:
        融合后的检索结果列表，按得分从高到低排序。
    """
    # Initialize score dictionary to hold fused scores for each document
    score_dict = defaultdict(float)

    # Iterate through each list of retrieved documents
    for docs in retrieved_docs_lists:
        # Iterate through documents and their ranks
        for rank, doc in enumerate(docs):
            # Convert document to string to use as a key (assumes documents can be serialized)
            doc_str = dumps(doc)
            if doc_str not in score_dict:
                score_dict[doc_str] = 0.0
            # Retrieve the current score of the document, if any
            previouse_score = score_dict[doc_str]
            # Update score using RRF formula
            score_dict[dumps(doc)] += 1 / (k + rank + 1)

    # 根据得分排序
    sorted_docs = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

    # 返回融合后的文档列表
    fused_docs = [loads(doc_str) for doc_str, score in sorted_docs]

    return fused_docs

# retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
# docs = retrieval_chain.invoke({"question":question})