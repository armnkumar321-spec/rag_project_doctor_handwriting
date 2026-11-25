"""Command line interface.

  python cli.py read path/to/prescription.png [--engine gemini|tesseract]
  python cli.py ask "What does TDS mean?"
  python cli.py build-index [--strategy per_entity|sentences|fixed]
"""
import argparse, json
from rxreadable import config


def main():
    ap = argparse.ArgumentParser(description="RxReadable CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read"); r.add_argument("image"); r.add_argument("--engine"); r.add_argument("--json", action="store_true")
    a = sub.add_parser("ask"); a.add_argument("question")
    b = sub.add_parser("build-index"); b.add_argument("--strategy", default="per_entity"); b.add_argument("--embeddings")
    args = ap.parse_args()

    if args.cmd == "read":
        from rxreadable.pipeline import process
        res = process(args.image, args.engine)
        if args.json:
            print(json.dumps(res, indent=2, default=str)); return
        print(f"== OCR ({res['ocr']['engine']}) ==\n{res['ocr']['text']}\n")
        p = res["parsed"]
        print(f"== Parsed ==\nPatient: {p['patient']}  Age/Sex: {p['age_sex']}  Date: {p['date']}")
        for i, m in enumerate(p["medicines"], 1):
            print(f"{i}. {m['plain_english']}")
            for w in m["warnings"]:
                print(f"     ! {w}")
        print(f"\n== Explanation ==\n{res['explanation']['answer']}\n\n{config.DISCLAIMER}")
    elif args.cmd == "ask":
        from rxreadable.pipeline import get_rag
        res = get_rag().answer(args.question)
        print(res["answer"]); print("\nSources:", ", ".join(h["id"] for h in res["sources"]))
    elif args.cmd == "build-index":
        from rxreadable.rag import RxRAG
        from rxreadable.embeddings import get_embedder
        rag = RxRAG(args.strategy, get_embedder(args.embeddings))
        print(f"Indexed {rag.build()} chunks with {rag.embedder.name} -> {rag.path}")


if __name__ == "__main__":
    main()
