"""Embed policy clauses and expose cosine-similarity search over them."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import config

load_dotenv()

BATCH_SIZE = 20


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_index() -> None:
    """Embed every clause as a retrieval document and save vectors + metadata."""
    clauses = json.loads(config.CLAUSES_JSON.read_text())
    client = _client()
    contents = [f"{c['title']}\n{c['text']}" for c in clauses]

    vectors = []
    for i in range(0, len(contents), BATCH_SIZE):
        batch = contents[i : i + BATCH_SIZE]
        response = client.models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(e.values for e in response.embeddings)

    np.savez_compressed(
        config.POLICY_INDEX,
        clause_ids=np.array([c["clause_id"] for c in clauses]),
        titles=np.array([c["title"] for c in clauses]),
        texts=np.array([c["text"] for c in clauses]),
        embeddings=np.array(vectors, dtype=np.float32),
    )


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k clauses by cosine similarity to the query."""
    data = np.load(config.POLICY_INDEX, allow_pickle=False)
    embeddings = data["embeddings"]

    client = _client()
    response = client.models.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vec = np.array(response.embeddings[0].values, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    scores = embeddings @ query_vec / norms
    top_k = np.argsort(-scores)[:k]

    return [
        {
            "clause_id": str(data["clause_ids"][i]),
            "title": str(data["titles"][i]),
            "text": str(data["texts"][i]),
            "score": float(scores[i]),
        }
        for i in top_k
    ]


if __name__ == "__main__":
    build_index()
    print(f"Saved {config.POLICY_INDEX}")
