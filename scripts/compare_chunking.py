"""Compare chunking strategies x embedding backends on retrieval hit@k.
Queries include OCR-style misspellings to test robustness.   python scripts/compare_chunking.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rxreadable.rag import RxRAG
from rxreadable.embeddings import get_embedder

QUERIES = [  # (query, expected substring in a top-k chunk)
    ("What is Pantoprazol used for", "Pantoprazole"), ("paracetamal side effects", "Paracetamol"),
    ("what does TDS mean", "three times daily"), ("meaning of SOS on prescription", "only if needed"),
    ("Losarten blood pressure", "Losartan"), ("Cefixim antibiotic", "Cefixime"),
    ("what is 1-0-1", "morning and night"), ("medicine for allergy sneezing", "Antihistamine"),
    ("drug for type 2 diabetes", "Metformin"), ("what does AC mean", "before food"),
    ("Hydroxyzin itching", "Hydroxyzine"), ("cholesterol lowering statin", "Atorvastatin"),
    ("Montelucast asthma", "Montelukast"), ("HS abbreviation", "bedtime"), ("x 5/7 duration", "5 days"),
]


def main(k=3, backends=("tfidf",)):
    print(f"{'embeddings':22}{'strategy':12}{'hit@1':>7}{f'hit@{k}':>7}")
    for b in backends:
        for strat in ("per_entity", "sentences", "fixed"):
            rag = RxRAG(strat, get_embedder(b), store="numpy"); rag.build(save=False)
            h1 = hk = 0
            for q, exp in QUERIES:
                hits = rag.retrieve(q, k)
                h1 += exp.lower() in hits[0]["text"].lower()
                hk += any(exp.lower() in h["text"].lower() for h in hits)
            print(f"{b:22}{strat:12}{h1 / len(QUERIES):>7.0%}{hk / len(QUERIES):>7.0%}")


if __name__ == "__main__":
    main(backends=sys.argv[1:] or ("tfidf",))
