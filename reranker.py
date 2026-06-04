"""
Reranker 精排模块
支持本地 Cross-Encoder 模型和远程 API 两种方式，对候选记忆进行相关性重排序。
"""
import os
import asyncio
from typing import List, Dict

RERANKER_TYPE = os.getenv("RERANKER_TYPE", "local")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-minicpm-layerwise")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "https://api.cohere.ai/v1/rerank")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
RERANKER_TIMEOUT = int(os.getenv("RERANKER_TIMEOUT", "10"))

_local_reranker = None

def get_local_reranker():
    global _local_reranker
    if _local_reranker is None:
        from sentence_transformers import CrossEncoder
        _local_reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"✅ 本地 Reranker 已加载: {RERANKER_MODEL_NAME}")
    return _local_reranker

async def rerank(query: str, candidates: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
    if not candidates or not query.strip():
        return candidates[:top_k]

    if RERANKER_TYPE == "api":
        return await _rerank_api(query, candidates, top_k)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _rerank_local, query, candidates, top_k)

def _rerank_local(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    model = get_local_reranker()
    pairs = [(query, cand["content"]) for cand in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

async def _rerank_api(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    if not RERANKER_API_KEY:
        print("⚠️  Reranker API Key 未设置，回退到原始排序")
        return candidates[:top_k]

    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {RERANKER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": [cand["content"] for cand in candidates],
            "top_n": top_k,
            "return_documents": False
        }
        async with httpx.AsyncClient(timeout=RERANKER_TIMEOUT) as client:
            resp = await client.post(RERANKER_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            score_map = {r["index"]: r["relevance_score"] for r in results}
            for idx, cand in enumerate(candidates):
                cand["rerank_score"] = score_map.get(idx, 0.0)
            reranked = [candidates[r["index"]] for r in results if r["index"] < len(candidates)]
            return reranked[:top_k]
    except Exception as e:
        print(f"⚠️  Reranker API 调用失败: {e}，回退到原始排序")
        return candidates[:top_k]
