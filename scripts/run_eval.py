"""Evaluate the full pipeline on the RxReadable test dataset.

  python scripts/run_eval.py --engine tesseract            # OCR + parsing on all 50 pages
  python scripts/run_eval.py --engine gemini --limit 10
  python scripts/run_eval.py --parser-only                 # parse the ground-truth text (upper bound for the parser)

Writes outputs/preds_<engine>.json, outputs/report_<engine>.json and prints CER/WER plus field accuracy.
"""
import argparse, json, re, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rxreadable import config
from rxreadable.ocr import run_ocr
from rxreadable.parser import parse_prescription, clean_text

try:
    import jiwer
except ImportError:
    jiwer = None


def norm(t):
    t = re.sub(r"\bRX\d{3}\b|SYNTHETIC TEST SAMPLE - NOT A REAL PRESCRIPTION|\[(ILLEGIBLE|UNK)\]", " ", t)
    return re.sub(r"\s+", " ", t.replace("~~", "")).strip()


def ncanon(s):
    return re.sub(r"[^a-z0-9./-]", "", s.lower())


def field_scores(gt_meds, pred_meds):
    """Match predicted medicines to ground-truth medicines by name; score each field."""
    s = defaultdict(lambda: [0, 0])
    pred_by = {m["name"].lower(): m for m in pred_meds}
    for g in gt_meds:
        if g["name"] == "[ILLEGIBLE]":
            continue
        p = pred_by.get(g["name"].lower())
        s["name"][0] += bool(p); s["name"][1] += 1
        if not p:
            for f in ("strength", "frequency", "duration"): s[f][1] += 1
            continue
        s["strength"][0] += ncanon(g["strength"].replace(" ", "")) == ncanon(p["strength"]) or "[ILLEGIBLE]" in g["strength"]
        s["strength"][1] += 1
        gf = g["frequency"].split()[0] if g["frequency"] else ""
        s["frequency"][0] += (ncanon(gf) == ncanon(p["frequency"]) or "[ILLEGIBLE]" in g["frequency"]
                              or (gf.lower() in ("once", "twice", "three", "at", "when") and gf.lower() in p["frequency"].lower()))
        s["frequency"][1] += 1
        gd = re.findall(r"\d+", g["duration"])
        s["duration"][0] += (bool(gd) and gd[0] == (re.findall(r"\d+", p["duration"]) or [""])[0]) or "[ILLEGIBLE]" in g["duration"]
        s["duration"][1] += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_DIR))
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--parser-only", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ds = Path(a.dataset)
    gt = json.load(open(ds / "ground_truth.json", encoding="utf-8"))["prescriptions"][: a.limit]
    name = "parser_only" if a.parser_only else a.engine
    preds, agg = {}, defaultdict(lambda: defaultdict(list))
    fields = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in gt:
        pid, lvl = r["prescription_id"], r["difficulty_level"]
        if a.parser_only:
            text = r["original_text"]
        else:
            t0 = time.time()
            res = run_ocr(ds / "images" / f"{pid}.png", a.engine)
            text, name = res["text"], res["engine"]
            if res["engine"] == "gemini":
                time.sleep(1)   # stay under free-tier rate limits
        preds[pid] = text
        parsed = parse_prescription(text)
        ref, hyp = norm(r["original_text"]), norm(text)
        c = jiwer.cer(ref, hyp) if jiwer else float("nan")
        w = jiwer.wer(ref, hyp) if jiwer else float("nan")
        for L in (lvl, "ALL"):
            agg[L]["cer"].append(c); agg[L]["wer"].append(w)
            for f, (ok, n) in field_scores(r["medicines"], parsed["medicines"]).items():
                fields[L][f][0] += ok; fields[L][f][1] += n
        print(f"{pid} {lvl:6} CER {c:.3f}  meds found {len(parsed['medicines'])}/{len(r['medicines'])}", flush=True)

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    json.dump(preds, open(out / f"preds_{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    mean = lambda x: sum(x) / len(x)
    report = {}
    print(f"\nengine={name}\n{'level':7}{'n':>4}{'CER':>7}{'WER':>7}{'name':>8}{'strength':>10}{'freq':>7}{'dur':>7}")
    for L in ("easy", "medium", "hard", "ALL"):
        if L not in agg: continue
        f = {k: v[0] / v[1] if v[1] else 0 for k, v in fields[L].items()}
        report[L] = {"n": len(agg[L]["cer"]), "cer": mean(agg[L]["cer"]), "wer": mean(agg[L]["wer"]), **f}
        print(f"{L:7}{len(agg[L]['cer']):>4}{report[L]['cer']:>7.3f}{report[L]['wer']:>7.3f}"
              f"{f.get('name', 0):>8.0%}{f.get('strength', 0):>10.0%}{f.get('frequency', 0):>7.0%}{f.get('duration', 0):>7.0%}")
    json.dump(report, open(out / f"report_{name}.json", "w"), indent=2)
    print(f"\nSaved outputs/preds_{name}.json and outputs/report_{name}.json")


if __name__ == "__main__":
    main()
