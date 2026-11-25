"""Retrieval-augmented explanations of prescriptions and medical terms."""
import pickle, re
from . import config
from .chunking import kb_documents, STRATEGIES, chunk_prescription
from .embeddings import get_embedder, TfidfEmbedder
from .vectorstore import NumpyStore, ChromaStore

SYSTEM = """You explain handwritten prescriptions to patients and students in simple English.
Use ONLY the context provided. If the context does not contain the answer, say you don't know.
Never change, recommend or invent doses. Point out look-alike names and illegible parts when relevant.
End with one short line reminding the reader to confirm with their doctor or pharmacist."""


class RxRAG:
    def __init__(self, strategy="per_entity", embedder=None, store=None):
        self.strategy = strategy
        self.embedder = embedder or get_embedder()
        backend = store or config.VECTOR_STORE
        self.store = ChromaStore() if backend == "chroma" else NumpyStore()
        self.path = config.INDEX_DIR / f"{self.embedder.name}_{strategy}"

    # ---------- indexing
    def build(self, save=True):
        chunks = STRATEGIES[self.strategy](kb_documents())
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.store.add(chunks, self.embedder.embed(texts))
        if save and isinstance(self.store, NumpyStore):
            self.store.save(self.path)
            if isinstance(self.embedder, TfidfEmbedder):
                pickle.dump(self.embedder, open(self.path / "tfidf.pkl", "wb"))
        return len(chunks)

    @classmethod
    def load_or_build(cls, strategy="per_entity", embedder_name=None):
        rag = cls(strategy, get_embedder(embedder_name))
        if isinstance(rag.store, NumpyStore) and (rag.path / "items.json").exists():
            rag.store = NumpyStore.load(rag.path)
            if (rag.path / "tfidf.pkl").exists():
                rag.embedder = pickle.load(open(rag.path / "tfidf.pkl", "rb"))
        else:
            rag.build()
        return rag

    def add_prescription(self, pid, parsed, ocr_text):
        chunks = chunk_prescription(pid, parsed, ocr_text)
        self.store.add(chunks, self.embedder.embed([c["text"] for c in chunks]))

    # ---------- retrieval
    def retrieve(self, query, k=4, where=None):
        return self.store.search(self.embedder.embed([query])[0], k, where)

    def context_for(self, parsed, question=""):
        hits, seen = [], set()
        for m in parsed.get("medicines", []):
            for h in self.retrieve(f"Drug: {m['name']}", k=1, where={"type": "drug"}):
                if h["id"] not in seen: hits.append(h); seen.add(h["id"])
            for tok in re.findall(r"[A-Za-z/]+|\d-\d-\d", f"{m['form']} {m['frequency']} {m['raw_line']}"):
                for h in self.retrieve(f"Abbreviation {tok.upper()} means", k=1, where={"type": "abbreviation"}):
                    if h["meta"]["name"].upper() == tok.upper() and h["id"] not in seen:
                        hits.append(h); seen.add(h["id"])
        if question:
            for h in self.retrieve(question, k=4):
                if h["id"] not in seen: hits.append(h); seen.add(h["id"])
        return hits

    # ---------- generation
    def answer(self, question, parsed=None, k=4):
        hits = self.context_for(parsed, question) if parsed else self.retrieve(question, k)
        ctx = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
        rx = ""
        if parsed:
            rx = "Prescription as read by OCR:\n" + "\n".join(
                f"- {m['raw_line']}  => {m['plain_english']}  warnings: {'; '.join(m['warnings']) or 'none'}"
                for m in parsed["medicines"])
        client = config.gemini_client()
        if client:
            from google.genai import types
            prompt = f"{rx}\n\nContext:\n{ctx}\n\nQuestion: {question}"
            r = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt,
                                               config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2))
            text = r.text
        else:  # offline: extractive answer from retrieved context
            text = "Relevant information found (offline mode, no LLM):\n" + "\n".join(f"- {h['text']}" for h in hits[:k + 2]) + \
                   "\n\nPlease confirm with your doctor or pharmacist."
        return {"answer": text, "sources": hits}

    def explain(self, parsed):
        return self.answer("Explain each medicine on this prescription: what it is generally used for, "
                           "how to read the dosing instructions, and anything to double-check.", parsed)
