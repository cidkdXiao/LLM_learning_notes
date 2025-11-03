"""
与 multi_query 类似，但将多个查询的结果使用 RRF (Retrieval Result Fusion) 方法进行融合，以提升检索效果。
"""
from collections import defaultdict
from typing import Callable, List, Sequence, Tuple, Optional
from langchain.schema import Document

from rag_tools.define_retriever import retriever
from rag_tools.multi_query import generate_queries

def rrf_fusion(
    retrieved_docs_lists: Sequence[Sequence[Document]],
    k: int = 60,
    top_m_per_list: Optional[int] = None,           # 每路只取前 M 做融合（加速）
    channel_weights: Optional[Sequence[float]] = None,  # 每路权重，长度需等于路数
    key_fn: Optional[Callable[[Document], Tuple]] = None,  # 自定义去重键
    top_k_final: Optional[int] = None               # 只返回前 K
) -> List[Document]:
    """
    Reciprocal Rank Fusion with options:
    - top_m_per_list: 截断每路的候选长度
    - channel_weights: 每路的权重
    - key_fn: 自定义文档去重键（默认用 (page_content, source)）
    - top_k_final: 只返回前 K 个融合结果
    """
    if key_fn is None:
        key_fn = lambda d: (d.page_content, d.metadata.get("source"))

    if channel_weights is None:
        channel_weights = [1.0] * len(retrieved_docs_lists)

    score_dict = defaultdict(float)
    repr_doc = {}  # key -> Document（第一次遇到的原对象，用于回传）

    for ch_idx, docs in enumerate(retrieved_docs_lists):
        weight = channel_weights[ch_idx]
        if top_m_per_list is not None:
            docs = docs[:top_m_per_list]
        for rank, doc in enumerate(docs):
            key = key_fn(doc)
            if key not in repr_doc:
                repr_doc[key] = doc
            score_dict[key] += weight * (1.0 / (k + rank + 1))

    # 排序并裁剪
    items = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    if top_k_final is not None:
        items = items[:top_k_final]

    # 回收原始文档对象（保留第一次看到的）
    fused_docs = [repr_doc[key] for key, _ in items]
    return fused_docs

# 多路检索：generate_queries | retriever.map()
# 然后用 rrf 融合
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | (lambda lists: rrf_fusion(
    lists,
    k=60,
    top_m_per_list=50,
    channel_weights=None,     # 或 [1.0, 0.8, 1.2] 等
    key_fn=lambda d: (d.page_content, d.metadata.get("source")),
    top_k_final=50
))
docs = retrieval_chain.invoke({"question": question})
