"""End-to-end pipeline: image -> OCR -> parse -> RAG explanation."""
import time
from PIL import Image
from .ocr import run_ocr
from .parser import parse_prescription
from .rag import RxRAG

_rag = None


def get_rag():
    global _rag
    if _rag is None:
        _rag = RxRAG.load_or_build()
    return _rag


def process(image, engine=None, explain=True, pid="uploaded", index=False):
    t0 = time.time()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    ocr = run_ocr(img, engine)
    parsed = parse_prescription(ocr["text"])
    out = {"prescription_id": pid, "ocr": ocr, "parsed": parsed}
    if explain:
        rag = get_rag()
        if index:
            rag.add_prescription(pid, parsed, ocr["text"])
        out["explanation"] = rag.explain(parsed)
    out["seconds"] = round(time.time() - t0, 2)
    return out
