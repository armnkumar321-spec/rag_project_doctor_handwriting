"""Chunking strategies. Compare them with scripts/compare_chunking.py."""
import json
from . import config


def kb_documents():
    """Knowledge base -> list of {id, text, meta}: one doc per drug and per abbreviation."""
    docs = []
    for d in json.load(open(config.KB_DIR / "drugs.json", encoding="utf-8"))["drugs"]:
        text = (f"Drug: {d['name']}" + (f" (also called {', '.join(d['aliases'])})" if d["aliases"] else "") +
                f". Class: {d['class']}. Commonly used for: {d['used_for']}. "
                f"Common side effects: {d['common_side_effects']}. " + (f"Caution: {d['caution']}. " if d["caution"] else "") +
                (f"Look-alike names: {', '.join(d['lookalikes'])}." if d["lookalikes"] else ""))
        docs.append({"id": f"drug:{d['name']}", "text": text, "meta": {"type": "drug", "name": d["name"]}})
    for k, v in json.load(open(config.KB_DIR / "abbreviations.json", encoding="utf-8")).items():
        text = f"Abbreviation {k} means {v['expansion']}" + (f" (Latin: {v['latin']})" if v.get("latin") else "") + \
               f". Category: {v['category']}."
        docs.append({"id": f"abbr:{k}", "text": text, "meta": {"type": "abbreviation", "name": k}})
    return docs


def chunk_per_entity(docs):
    return [dict(d) for d in docs]


def chunk_fixed(docs, size=200, overlap=40):
    """Concatenate everything and cut fixed-size character windows (the naive baseline)."""
    full = "\n".join(d["text"] for d in docs)
    out, i, n = [], 0, 0
    while i < len(full):
        out.append({"id": f"fixed:{n}", "text": full[i:i + size], "meta": {"type": "fixed"}})
        i += size - overlap; n += 1
    return out


def chunk_sentences(docs, max_sent=2):
    out = []
    for d in docs:
        sents = [s.strip() + "." for s in d["text"].split(". ") if s.strip()]
        for j in range(0, len(sents), max_sent):
            out.append({"id": f"{d['id']}#{j}", "text": " ".join(sents[j:j + max_sent]), "meta": d["meta"]})
    return out


def chunk_prescription(pid, parsed, ocr_text):
    """A processed prescription -> one chunk per medicine + one for the whole text."""
    out = [{"id": f"rx:{pid}", "text": ocr_text, "meta": {"type": "prescription", "prescription_id": pid}}]
    for i, m in enumerate(parsed["medicines"]):
        out.append({"id": f"rx:{pid}:m{i}", "text": f"Prescription {pid}: {m['plain_english']}. Written as: {m['raw_line']}",
                    "meta": {"type": "prescription_medicine", "prescription_id": pid, "name": m["name"]}})
    return out


STRATEGIES = {"per_entity": chunk_per_entity, "fixed": chunk_fixed, "sentences": chunk_sentences}
