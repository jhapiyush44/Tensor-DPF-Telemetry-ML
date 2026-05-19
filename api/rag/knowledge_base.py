"""
DPF RAG Knowledge Base
Stores domain knowledge as embeddings in FAISS for context retrieval.
Uses sentence-transformers for local embeddings (no API key needed).
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple

# ── Domain Knowledge Chunks ──────────────────────────────────────────────────
DPF_KNOWLEDGE = [
    # Soot load ranges
    "DPF soot load below 30% is considered low and normal. No action required.",
    "DPF soot load between 30% and 60% is moderate. Monitor driving conditions and avoid excessive idling.",
    "DPF soot load between 60% and 80% is high. Passive regeneration may be insufficient. Plan a highway drive.",
    "DPF soot load above 80% is critical. Active regeneration is urgently needed to prevent filter damage.",
    "DPF soot load at 100% indicates a fully blocked filter. Immediate forced regeneration or service required.",

    # RPM and regeneration
    "High RPM above 2500 combined with sustained load promotes passive DPF regeneration through elevated exhaust temperatures.",
    "Low RPM below 1200 during extended idling prevents regeneration and accelerates soot accumulation.",
    "Engine RPM between 1800 and 2500 is the optimal range for passive regeneration during highway driving.",

    # Exhaust temperature
    "Exhaust temperature above 550°C enables passive DPF regeneration by burning off accumulated soot.",
    "Exhaust temperature below 300°C during city driving prevents regeneration and causes soot buildup.",
    "Exhaust temperature differential (pre minus post DPF) greater than 50°C indicates active soot combustion.",
    "Sustained exhaust temperature above 600°C during active regeneration burns soot efficiently in 20-30 minutes.",

    # Speed and driving patterns
    "Highway driving above 80 km/h sustains high exhaust temperatures necessary for passive DPF regeneration.",
    "Urban stop-and-go driving below 40 km/h with frequent idling is the primary cause of rapid soot accumulation.",
    "Short trips under 15 minutes do not allow the engine to reach regeneration temperatures.",

    # Engine load
    "Engine load above 70% generates sufficient heat for passive regeneration under most conditions.",
    "Engine load below 20% for extended periods (idle ratio) is strongly correlated with high soot accumulation rates.",
    "High load ratio above 50% combined with high exhaust temperature is the optimal condition for soot burn-off.",

    # Flow rate
    "Low exhaust flow rate below 30 g/s indicates low engine activity and insufficient energy for regeneration.",
    "High exhaust flow rate above 80 g/s combined with high temperature accelerates passive regeneration significantly.",

    # Rolling temperature features
    "Rolling mean exhaust temperature over 10 minutes above 480°C suggests ongoing passive regeneration.",
    "Rolling mean exhaust temperature over 60 minutes below 350°C indicates predominantly cold driving conditions.",
    "Large temperature delta (instantaneous minus rolling mean) suggests recent acceleration or load spike.",

    # Recommendations
    "To reduce soot load: drive at highway speeds above 80 km/h for at least 30 minutes continuously.",
    "To prevent rapid soot buildup: avoid extended idling; turn off engine if stationary for more than 2 minutes.",
    "After a warning light for DPF: do not ignore it. Drive at 60-80 km/h on an open road for 20-30 minutes.",
    "Frequent short city trips require periodic planned highway drives to maintain DPF health.",
    "Using low-quality diesel fuel with high sulfur content accelerates DPF clogging and inhibits regeneration.",
    "Regular engine oil changes are important as oil ash is a secondary contributor to DPF blockage.",
]

# ── Embedding Model (local, no API needed) ───────────────────────────────────
_embedder = None
_index = None
_chunks = DPF_KNOWLEDGE


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _embedder


def build_index() -> faiss.IndexFlatL2:
    """Build FAISS index from DPF knowledge chunks."""
    embedder = _get_embedder()
    embeddings = embedder.encode(_chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity on normalized vecs
    index.add(embeddings.astype(np.float32))
    return index


def get_index() -> faiss.IndexFlatL2:
    global _index
    if _index is None:
        cache_path = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                _index = pickle.load(f)
        else:
            _index = build_index()
            with open(cache_path, "wb") as f:
                pickle.dump(_index, f)
    return _index


def retrieve(query: str, top_k: int = 5) -> List[str]:
    """Retrieve top-k relevant knowledge chunks for a query."""
    embedder = _get_embedder()
    index = get_index()
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(query_vec.astype(np.float32), top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]
