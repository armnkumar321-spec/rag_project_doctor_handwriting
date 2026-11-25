"""Tiny vector database: NumPy cosine search with save/load. Optional Chroma backend."""
import json, pickle
import numpy as np
from . import config


class NumpyStore:
    def __init__(self):
        self.vecs = np.zeros((0, 0), np.float32); self.items = []

    def add(self, items, vecs):
        self.vecs = vecs if not len(self.items) else np.vstack([self.vecs, vecs])
        self.items += items

    def search(self, qvec, k=4, where=None):
        if not self.items:
            return []
        scores = self.vecs @ qvec
        order = np.argsort(-scores)
        out = []
        for i in order:
            it = self.items[i]
            if where and any(it["meta"].get(a) != b for a, b in where.items()):
                continue
            out.append({**it, "score": float(scores[i])})
            if len(out) == k:
                break
        return out

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vecs)
        json.dump(self.items, open(path / "items.json", "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        s = cls(); s.vecs = np.load(path / "vectors.npy")
        s.items = json.load(open(path / "items.json", encoding="utf-8")); return s


class ChromaStore:
    def __init__(self, path=None, name="rxreadable"):
        import chromadb
        self.c = chromadb.PersistentClient(path=str(path or config.INDEX_DIR / "chroma"))
        self.col = self.c.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, items, vecs):
        self.col.upsert(ids=[i["id"] for i in items], documents=[i["text"] for i in items],
                        metadatas=[i["meta"] for i in items], embeddings=vecs.tolist())

    def search(self, qvec, k=4, where=None):
        r = self.col.query(query_embeddings=[qvec.tolist()], n_results=k, where=where or None)
        return [{"id": i, "text": t, "meta": m, "score": 1 - d}
                for i, t, m, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def save(self, path): pass
