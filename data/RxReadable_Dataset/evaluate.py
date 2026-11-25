"""Score OCR/HTR predictions against the RxReadable ground truth.

Usage:
    python evaluate.py predictions.json            # {"RX001": "predicted text", ...}
    python evaluate.py predictions.csv             # columns: prescription_id,predicted_text
    python evaluate.py predictions.json --mode clean --no-header

Reference modes (--mode):
    visible (default)  every word that has visible, readable ink: crossed-out words are KEPT
                       as plain words (they are on the page), corrections kept, [ILLEGIBLE] removed.
    clean              the doctor's intended text: crossed-out words removed, [ILLEGIBLE] removed.
In both modes, [ILLEGIBLE] / [UNK] tokens in the prediction are also removed, so a model is
neither rewarded nor punished for flagging illegible spots; inventing a word there costs one insertion.
"""
import argparse, csv, json, re, sys, difflib
from collections import defaultdict
from pathlib import Path

try:
    import jiwer
except ImportError:
    jiwer = None

STRIP = [r"\bRX\d{3}\b", r"SYNTHETIC TEST SAMPLE - NOT A REAL PRESCRIPTION"]


def norm(t, header=True, lines_skip=3):
    if not header:
        t = "\n".join(t.split("\n")[lines_skip:])
    for p in STRIP:
        t = re.sub(p, " ", t)
    t = re.sub(r"\[(ILLEGIBLE|UNK)\]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def reference(text, mode):
    if mode == "clean":
        text = re.sub(r"~~.+?~~\s*", "", text)
    else:
        text = text.replace("~~", "")
    return text


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(r, h):
    return jiwer.cer(r, h) if jiwer else lev(r, h) / max(1, len(r))


def wer(r, h):
    return jiwer.wer(r, h) if jiwer else lev(r.split(), h.split()) / max(1, len(r.split()))


def med_hit(name, pred, fuzzy=0.8):
    words = pred.lower().split()
    n = len(name.split())
    grams = [" ".join(words[i:i + n]) for i in range(len(words))]
    exact = name.lower() in grams
    best = max((difflib.SequenceMatcher(None, name.lower(), g).ratio() for g in grams), default=0)
    return exact, best >= fuzzy


def load_preds(path):
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, list):
            return {x["prescription_id"]: x.get("predicted_text", x.get("text", "")) for x in d}
        return d
    with open(p, newline="", encoding="utf-8") as f:
        return {r["prescription_id"]: r["predicted_text"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--gt", default=str(Path(__file__).with_name("ground_truth.json")))
    ap.add_argument("--mode", choices=["visible", "clean"], default="visible")
    ap.add_argument("--no-header", action="store_true", help="score only below the letterhead (use when your OCR input was cropped below the header line)")
    ap.add_argument("--per-item", action="store_true")
    a = ap.parse_args()
    gt = json.load(open(a.gt, encoding="utf-8"))["prescriptions"]
    preds = load_preds(a.predictions)
    agg = defaultdict(lambda: defaultdict(list))
    for r in gt:
        pid = r["prescription_id"]
        if pid not in preds:
            print(f"warning: no prediction for {pid}", file=sys.stderr); continue
        ref = norm(reference(r["original_text"], a.mode), header=not a.no_header)
        hyp = norm(preds[pid], header=True)
        c, w = cer(ref, hyp), wer(ref, hyp)
        names = [n for n in r["medicine_names"] if n != "[ILLEGIBLE]"]
        hits = [med_hit(n, hyp) for n in names]
        for lvl in (r["difficulty_level"], "ALL"):
            agg[lvl]["cer"].append(c); agg[lvl]["wer"].append(w)
            agg[lvl]["med_exact"] += [h[0] for h in hits]; agg[lvl]["med_fuzzy"] += [h[1] for h in hits]
        if a.per_item:
            print(f"{pid:6} {r['difficulty_level']:6} CER {c:.3f} WER {w:.3f} meds {sum(h[0] for h in hits)}/{len(hits)}")
    mean = lambda x: sum(x) / len(x) if x else float("nan")
    print(f"\nmode={a.mode}  header={'excluded' if a.no_header else 'included'}  jiwer={'yes' if jiwer else 'no (built-in)'}")
    print(f"{'level':8}{'n':>4}{'CER':>8}{'WER':>8}{'Med exact':>11}{'Med fuzzy':>11}")
    for lvl in ("easy", "medium", "hard", "ALL"):
        d = agg.get(lvl)
        if not d: continue
        print(f"{lvl:8}{len(d['cer']):>4}{mean(d['cer']):>8.3f}{mean(d['wer']):>8.3f}"
              f"{mean(d['med_exact']):>11.1%}{mean(d['med_fuzzy']):>11.1%}")


if __name__ == "__main__":
    main()
