"""
ChromaDB vector store for resume chunks.

Design decisions:
- In-memory (EphemeralClient) per analysis session — no disk state to manage,
  no conflicts between concurrent Streamlit users.
- Each chunk stored with metadata: section, chunk_id, word_count.
- Supports section-filtered queries so we can ask "what does the Experience
  section say about Kubernetes?" separately from the Skills section.
"""
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings

# ── Collection builder ────────────────────────────────────────────────────────

def build_collection(
    chunks: list[dict],
    embeddings: np.ndarray,
    collection_name: str | None = None,
) -> chromadb.Collection:
    """
    Create an ephemeral ChromaDB collection for one resume session.

    Args:
        chunks:     list of {chunk_id, section, text}
        embeddings: numpy array (n_chunks, dim), pre-normalised
        collection_name: optional; auto-generated if None

    Returns:
        Populated ChromaDB Collection ready for querying.
    """
    client = chromadb.EphemeralClient()
    name = collection_name or f"resume_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    ids       = [str(c["chunk_id"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "section":    c["section"],
            "chunk_id":   c["chunk_id"],
            "word_count": len(c["text"].split()),
        }
        for c in chunks
    ]
    # ChromaDB expects plain Python lists, not numpy arrays
    emb_list = embeddings.tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=emb_list,
        metadatas=metadatas,
    )
    return collection


# ── Query helpers ─────────────────────────────────────────────────────────────

def query_top_k(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    n_results: int = 5,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve top-k chunks most similar to query_embedding.

    Args:
        section_filter: if provided, restrict results to that section only.

    Returns:
        list of {text, section, chunk_id, distance, similarity}
        sorted by similarity descending.
    """
    where = {"section": section_filter} if section_filter else None
    kwargs = dict(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    res = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        # ChromaDB cosine distance = 1 - cosine_similarity
        similarity = round(1.0 - float(dist), 4)
        results.append({
            "text":       doc,
            "section":    meta["section"],
            "chunk_id":   meta["chunk_id"],
            "similarity": similarity,
        })

    return sorted(results, key=lambda x: -x["similarity"])


def query_all_sections(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    n_per_section: int = 2,
) -> dict[str, list[dict]]:
    """
    For a given query, retrieve the best n_per_section chunks for each section.
    Returns {section_name: [chunk, ...]} dict.
    """
    # Get all unique sections
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sections  = list({m["section"] for m in all_meta})

    section_results = {}
    for sec in sections:
        hits = query_top_k(collection, query_embedding, n_results=n_per_section, section_filter=sec)
        if hits:
            section_results[sec] = hits

    return section_results


def get_section_chunks(
    collection: chromadb.Collection,
    section: str,
) -> list[dict]:
    """Return all stored chunks for a given section."""
    res = collection.get(
        where={"section": section},
        include=["documents", "metadatas"],
    )
    return [
        {"text": doc, "section": meta["section"], "chunk_id": meta["chunk_id"]}
        for doc, meta in zip(res["documents"], res["metadatas"])
    ]
