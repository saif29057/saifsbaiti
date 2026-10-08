"""
qdrant_store.py
Accès à Qdrant : client, collection "invoices", embeddings et conversion des filtres.

Par défaut Qdrant tourne en mode local (dossier ./qdrant_db, sans serveur).
Pour utiliser un serveur Qdrant (Docker / Cloud), définir QDRANT_URL (+ QDRANT_API_KEY) dans .env.
"""

import atexit
import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = "invoices"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

_client = None
_embedder = None


def get_client():
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        _client = (QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY")) if url
                   else QdrantClient(path=str(BASE_DIR / "qdrant_db")))
        atexit.register(_client.close)
    return _client


def embed(texts):
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return [v.tolist() for v in _embedder.embed(list(texts))]


def create_collection(reset=False):
    client = get_client()
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    return client


def to_qdrant_filter(clauses):
    """[{field, op, value}] (opérateurs $eq/$ne/$gt/$gte/$lt/$lte/$in/$nin) -> models.Filter"""
    must, must_not = [], []
    for c in clauses or []:
        field, op, value = c["field"], c["op"], c["value"]
        if op == "$eq":
            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
        elif op == "$ne":
            must_not.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
        elif op == "$in":
            must.append(models.FieldCondition(key=field, match=models.MatchAny(any=value)))
        elif op == "$nin":
            must_not.append(models.FieldCondition(key=field, match=models.MatchAny(any=value)))
        else:  # $gt $gte $lt $lte
            must.append(models.FieldCondition(key=field, range=models.Range(**{op[1:]: value})))
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def count():
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    return client.count(COLLECTION_NAME, exact=True).count


def scroll_all(qfilter=None, with_payload=True):
    """Tous les points qui respectent le filtre (pour les statistiques exactes)."""
    client, points, offset = get_client(), [], None
    while True:
        batch, offset = client.scroll(COLLECTION_NAME, scroll_filter=qfilter, limit=1000,
                                      offset=offset, with_payload=with_payload, with_vectors=False)
        points.extend(batch)
        if offset is None:
            return points


def search(query, qfilter=None, limit=10):
    res = get_client().query_points(COLLECTION_NAME, query=embed([query])[0],
                                    query_filter=qfilter, limit=limit, with_payload=True)
    return res.points
