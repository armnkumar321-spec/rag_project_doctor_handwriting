"""Turn raw OCR text into structured medicine entries.

Steps: clean text -> find medicine lines -> fuzzy-match drug names against the knowledge base
(fixes OCR misspellings, flags look-alikes) -> extract strength, dose, frequency, timing, duration
-> expand abbreviations into plain English.
"""
import json, re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from rapidfuzz import fuzz, process
from . import config

FORMS = {"tab": "Tab", "tablet": "Tab", "cap": "Cap", "capsule": "Cap", "syp": "Syp", "syrup": "Syp",
         "inj": "Inj", "injection": "Inj", "sachet": "Sachet", "drops": "Drops", "oint": "Oint", "gel": "Gel"}
FREQ_ABBR = ["OD", "BD", "BID", "TDS", "TID", "QID", "QDS", "HS", "SOS", "PRN", "STAT"]
FREQ_WORDS = ["once a day", "twice a day", "three times a day", "four times a day", "at night",
              "when needed", "in the morning"]
TIMING = {"AC": "before food", "PC": "after food", "after food": "after food", "before food": "before food"}

RE_STRENGTH = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|ml|g|iu)\b", re.I)
RE_PATTERN = re.compile(r"\b([0-2](?:/2)?)\s*-\s*([0-2](?:/2)?)\s*-\s*([0-2](?:/2)?)\b")
RE_DURATION = re.compile(r"(?:\bx\s*|\bfor\s+)?(\d+)\s*(/7|d\b|days?\b|wks?\b|weeks?\b|m\b|months?\b)", re.I)
RE_DOSE = re.compile(r"(?<![\d.-])(1/2|½|\d)\s*(tab(?:let)?s?|cap(?:sule)?s?|sachets?|tsp|puffs?)?(?![\d./-])(?!\s*(?:mg|ml|mcg|g|iu|d\b|days?|wk|m\b|/7))", re.I)


@lru_cache(maxsize=1)
def load_kb():
    drugs = json.load(open(config.KB_DIR / "drugs.json", encoding="utf-8"))["drugs"]
    abbr = json.load(open(config.KB_DIR / "abbreviations.json", encoding="utf-8"))
    names = {}
    for d in drugs:
        names[d["name"].lower()] = d
        for a in d.get("aliases", []):
            names[a.lower()] = d
    return drugs, abbr, names


@dataclass
class Medicine:
    raw_line: str
    name_ocr: str = ""
    name: str = ""               # matched canonical name ("" if not matched)
    match_score: float = 0.0
    form: str = ""
    strength: str = ""
    dose: str = ""
    frequency: str = ""
    timing: str = ""
    duration: str = ""
    plain_english: str = ""
    warnings: list = field(default_factory=list)
    drug_info: dict = field(default_factory=dict)


def clean_text(text):
    """Resolve crossed-out text (~~old~~ new -> new), normalise spaces and common OCR confusions."""
    text = re.sub(r"~~.+?~~\s*", "", text)
    text = text.replace("½", "1/2").replace("—", "-").replace("–", "-")
    text = re.sub(r"(?<=\d)\s*[oO](?=\s*mg)", "0", text)      # 5o0mg -> 500mg
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def match_drug(words, min_score=78):
    """Best fuzzy match over 1-2 word windows. Returns (span_text, kb_entry, score)."""
    _, _, names = load_kb()
    best = ("", None, 0.0)
    choices = list(names.keys())
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            span = " ".join(words[i:i + n])
            s = re.sub(r"[^a-z0-9 ]", "", span.lower())
            if len(s) < 3 or s in FORMS or s.upper() in FREQ_ABBR:
                continue
            hit = process.extractOne(s, choices, scorer=fuzz.ratio)
            if hit and hit[1] > best[2]:
                best = (span, names[hit[0]], hit[1])
    return best if best[2] >= min_score else ("", None, best[2])


def expand_frequency(freq):
    _, abbr, _ = load_kb()
    out = []
    for tok in freq.split():
        e = abbr.get(tok.upper())
        out.append(e["expansion"] if e else tok)
    return " ".join(out)


def parse_line(line):
    words = line.replace(",", " ").split()
    span, drug, score = match_drug(words)
    if not drug:
        return None
    m = Medicine(raw_line=line, name_ocr=span, name=drug["name"], match_score=round(score, 1))
    lw = [w.lower().strip(".)") for w in words]
    for w in lw:
        if w in FORMS:
            m.form = FORMS[w]; break
    rest = line.replace(span, " ", 1)
    if (s := RE_STRENGTH.search(rest)):
        m.strength = f"{s.group(1).replace(',', '.')}{s.group(2).lower().replace('iu', 'IU')}"
        rest = rest.replace(s.group(0), " ", 1)
    if (p := RE_PATTERN.search(rest)):
        m.frequency = "-".join(p.groups()); rest = rest.replace(p.group(0), " ", 1)
    else:
        for f in FREQ_ABBR:
            if re.search(rf"\b{f}\b", rest, re.I):
                m.frequency = f; rest = re.sub(rf"\b{f}\b", " ", rest, count=1, flags=re.I); break
        else:
            for f in FREQ_WORDS:
                if f in rest.lower():
                    m.frequency = f; rest = re.sub(f, " ", rest, flags=re.I); break
    for t, v in TIMING.items():
        if re.search(rf"\b{t}\b", rest, re.I if " " in t else 0):
            m.timing = v; rest = re.sub(rf"\b{t}\b", " ", rest, flags=re.I); break
    if (d := RE_DURATION.search(rest)):
        num, unit = d.group(1), d.group(2).lower()
        unit = {"/7": "days", "d": "days", "day": "days", "wk": "weeks", "wks": "weeks", "week": "weeks",
                "m": "months", "month": "months"}.get(unit, unit)
        m.duration = f"{num} {unit}"
        rest = rest.replace(d.group(0), " ", 1)
    rest = re.sub(r"^\s*\d+\s*[.)]", " ", rest)                     # list number "1)" / "2."
    if (q := RE_DOSE.search(rest)):
        m.dose = (q.group(1) + (" " + q.group(2) if q.group(2) else "")).strip()

    # warnings
    if score < 90:
        m.warnings.append(f"Low-confidence name match ('{span}' read as {drug['name']}, {score:.0f}%). Verify.")
    if drug.get("lookalikes"):
        m.warnings.append(f"Look-alike drug name: {', '.join(drug['lookalikes'])}. Double-check the spelling.")
    if "[ILLEGIBLE]" in line:
        m.warnings.append("Part of this line is illegible.")
    for key in ("strength", "frequency", "duration"):
        if not getattr(m, key):
            m.warnings.append(f"{key} not found")
    if re.search(r"\b0?\.\d|\b[17]\d?mg", m.strength or ""):
        m.warnings.append("Strength contains digits that are easy to misread (1/7, decimal point).")
    m.plain_english = describe(m)
    m.drug_info = {k: drug.get(k, "") for k in ("class", "used_for", "common_side_effects", "caution")}
    return m


def describe(m):
    form = {"Tab": "tablet", "Cap": "capsule", "Syp": "syrup", "Inj": "injection", "Sachet": "sachet"}.get(m.form, m.form.lower())
    dose = m.dose
    if dose and not re.search(r"[a-z]", dose, re.I) and form:
        dose = f"{dose} {form}"
    what = f"{dose} of {m.name}" if dose else (f"{m.name} {form}".strip())
    if m.strength:
        what += f" {m.strength}"
    parts = [what]
    if m.frequency:
        parts.append(expand_frequency(m.frequency).split(" (")[0])
    if m.timing:
        parts.append(m.timing)
    if m.duration:
        n, u = m.duration.split(" ", 1)
        parts.append(f"for {n} {u[:-1] if n == '1' and u.endswith('s') else u}")
    return ", ".join(parts)


def parse_prescription(text):
    """Return dict with medicines, header info, advice and leftover lines."""
    text = clean_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meds, other = [], []
    info = {"date": "", "patient": "", "age_sex": "", "complaint": "", "diagnosis": "", "advice": []}
    for line in lines:
        low = line.lower()
        if re.match(r"^date\s*[:.-]", low):
            info["date"] = line.split(":", 1)[-1].strip(); continue
        if re.match(r"^name\s*[:.-]", low):
            info["patient"] = re.split(r"age", line.split(":", 1)[-1], flags=re.I)[0].strip()
            if "age" not in low:
                continue
        if "age/sex" in low or re.search(r"\bage\b", low):
            info["age_sex"] = line.split(":")[-1].strip()
            continue
        if re.match(r"^(c/o|complaint)", low):
            info["complaint"] = line; continue
        if re.match(r"^dx\b", low):
            info["diagnosis"] = line.split(":", 1)[-1].strip(); continue
        if re.match(r"^(adv|advice|review|rv|avoid|drink|take rest|come back)", low):
            info["advice"].append(line); continue
        med = parse_line(line) if not re.match(r"^rx$", low) else None
        if med:
            meds.append(med)
        elif meds and (RE_DURATION.search(line) or re.match(r"^(days?|weeks?|months?|wks?)\b", low) or any(
                re.search(rf"\b{f}\b", line, re.I) for f in FREQ_ABBR + FREQ_WORDS)) and len(line.split()) <= 6:
            meds[-1] = parse_line(meds[-1].raw_line + " " + line)    # wrapped continuation line
        else:
            other.append(line)
    # fallbacks when OCR garbled the labels ("pate:", "agelsex")
    if not info["date"] and (d := re.search(r"\b\d{1,2}\s*[/.-]\s*(?:\d{1,2}|[A-Za-z]{3})\s*[/.-]\s*\d{2,4}\b", text)):
        info["date"] = d.group(0)
    if not info["age_sex"] and (a := re.search(r"\b(\d{1,3})\s*(?:yrs?|y)?\w?\s*/\s*([MF])\b", text)):
        info["age_sex"] = f"{a.group(1)}/{a.group(2)}"
    return {"medicines": [asdict(m) for m in meds], **info, "other_lines": other}
