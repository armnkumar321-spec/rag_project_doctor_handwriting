"""Embedding backends: Gemini API, sentence-transformers (local), or TF-IDF char n-grams (offline fallback,
robust to OCR typos)."""
import numpy as np
from . import config


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
        self.fitted = False

    def fit(self, texts):
        self.vec.fit(texts); self.fitted = True

    def embed(self, texts):
        if not self.fitted:
            self.fit(texts)
        m = self.vec.transform(texts).toarray().astype(np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class STEmbedder:
    name = "sentence-transformers"

    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)

    def fit(self, texts): pass

    def embed(self, texts):
        return np.asarray(self.m.encode(texts, normalize_embeddings=True), dtype=np.float32)


class GeminiEmbedder:
    name = "gemini"

    def __init__(self):
        self.client = config.gemini_client()

    def fit(self, texts): pass

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 90):
            r = self.client.models.embed_content(model=config.GEMINI_EMBED_MODEL, contents=texts[i:i + 90])
            out += [e.values for e in r.embeddings]
        m = np.asarray(out, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def get_embedder(name=None):
    name = (name or config.EMBEDDINGS).lower()
    if name == "auto":
        if config.GEMINI_API_KEY:
            name = "gemini"
        else:
            try:
                import sentence_transformers  # noqa
                name = "sentence-transformers"
            except ImportError:
                name = "tfidf"
    return {"gemini": GeminiEmbedder, "sentence-transformers": STEmbedder, "tfidf": TfidfEmbedder}[name]()
