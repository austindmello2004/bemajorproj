from typing import List, Dict
import math


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks_with_embeddings(agent, text: str) -> List[Dict]:
    """Chunk text and attach embeddings using the provided agent embedder."""
    chunks = chunk_text(text)
    results: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        embedding = agent.embed_text(chunk) if agent else []
        results.append({"chunk_index": idx, "content": chunk, "embedding": embedding})
    return results


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
