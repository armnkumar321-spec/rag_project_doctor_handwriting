# RxReadable – Synthetic Handwritten Prescription Test Set (v1.0)

A 50-page **synthetic** test set for evaluating handwriting recognition (OCR/HTR) and the downstream
RAG pipeline of *RxReadable – Doctor Handwriting Recognition and Prescription Understanding System*.

> **Disclaimer.** Every clinic, doctor, registration number, address and patient on these pages is
> invented. Nothing here is a real patient record. The dataset is **not clinically validated**, and
> the drug/dose combinations are test strings only. They are **not treatment advice**. Each page carries the
> footer "SYNTHETIC TEST SAMPLE - NOT A REAL PRESCRIPTION".

## 1. Contents

```
RxReadable_Dataset/
├── prescriptions.pdf      50 pages, one prescription per page (A5 portrait, 150 dpi, JPEG q90)
├── images/RX001.png …     the same 50 pages as lossless PNG (874 × 1240 px). Use these for OCR.
├── ground_truth.csv       one row per prescription, list fields joined with " | "
├── ground_truth.json      same data plus per-medicine objects, corrections, layout, debug info
├── evaluate.py            CER / WER / medicine-name accuracy per difficulty level
├── generator/           data.py, render.py, build.py: the scripts that produced everything (reproducible, seeded)
└── README.md
```

IDs (`RX001` to `RX050`) are the same across the PDF page order, image file names, CSV and JSON.
Page *n* of the PDF is `RX{n:03d}`.

## 2. Difficulty distribution

| Level  | IDs           | Count | What makes it hard |
|--------|---------------|------:|--------------------|
| easy   | RX001 to RX015 | 15 | Neat print-style hands (Patrick Hand, Kalam, Caveat), dark ink, almost no tilt, instructions in full words ("twice a day for 5 days") |
| medium | RX016 to RX030 | 15 | Cursive hands, tighter word spacing, heavy abbreviations (OD/BD/TDS/QID/HS/SOS/AC/PC, 1-0-1 patterns, x 5/7), mild tilt, some corrections |
| hard   | RX031 to RX050 | 20 | Messiest cursive hands, faint and uneven ink, 3–6° line tilt plus page rotation, blur, ink smudges, printed labels mixed with handwriting, crossed-out words, **≥1 `[ILLEGIBLE]` segment on every page** |

### Special test cases (see the `special_cases` field to filter)

| Tag | Pages | Meaning |
|-----|------:|---------|
| `date` | 50 | Varied formats: `12/03/26`, `12-Mar-2026`, `12.3.26`, `12 Mar 26` |
| `crossed_out_word` + `handwritten_correction` | 28 | Struck-through word with the correction written above it |
| `crossed_out_date` | 3 | Date itself corrected |
| `illegible_segment` | 26 | Word rendered and then deliberately destroyed |
| `mixed_case` | 25 | e.g. `TAB`, `cAP`, `PARACETAMOL`, `metformin` |
| `numeric_values` | 19 | BP, temperature, weight, SpO2 |
| `ambiguous_digits` | 14 | Strengths such as `0.5mg`, `1.5mg`, `17mg`, `75mg` (1↔7, 5↔6, decimal points) |
| `lookalike_name:A~B` | 18 | Contains drug A, which is easily confused with B (Losartan~Valsartan, Cefixime~Cefuroxime, Hydroxyzine~Hydralazine, Metformin~Metronidazole, Amlodipine~Amiloride, Prednisolone~Prednisone) |
| `lookalike_corrected` | 9 | Doctor wrote B, crossed it out, and wrote A above it |

## 3. Ground-truth conventions

* `original_text` is **every line of text on the page in reading order**, with `\n` line breaks that
  match the rendered line breaks. This includes the 3 printed letterhead lines, printed labels
  (`Name:`, `Date:`, `Dx:`) and the printed caption `Signature`.
  **Excluded:** the small corner ID (`RX001`), the footer disclaimer, and the signature scribble
  (it is not text). `evaluate.py` strips the ID and footer from predictions automatically.
* Two-column header pages (`layout = two_column`): the left block is first, then the right-hand `Date:` line.
* **Crossed-out text:** `~~old~~ new`. `old` is the struck word and `new` the handwritten correction.
  In field columns (`medicine_names`, `dosage`, `frequency`, `duration`) only the **correction** appears.
* **Illegible text:** `[ILLEGIBLE]`, everywhere it occurs. No reading is guessed. The word that was
  rendered before being destroyed is kept **only** under `_render_source_hidden` in the JSON, with its
  pixel box (before the small page rotation on hard pages). That is for debugging the renderer, **not**
  ground truth, and should never be used for scoring.
* `dosage` = strength + dose count as written (e.g. `500 mg 1 tablet`, `20mg 1/2`).
  `frequency` includes AC/PC/after food. `duration` keeps the written form (`x 5/7`, `for 10 days`).
* `abbreviations` lists abbreviation tokens found in the handwritten body, plus dose patterns such as `1-0-1 (dose pattern)`.
* `unclear_segments` lists every line that contains `[ILLEGIBLE]`.

CSV columns: `prescription_id, difficulty_level, original_text, medicine_names, dosage, frequency,
duration, abbreviations, unclear_segments, special_cases`.
JSON adds: `medicines` (objects `{form, name, strength, dose, frequency, duration}`), `patient`,
`corrections`, `layout`, `handwriting_font`, `_render_source_hidden`.

## 4. How the pages were made

Text was generated **first** as structured data. The images were rendered from that same data, and
the line breaks in `original_text` are recorded by the renderer as it wraps. Handwriting is simulated
with open-licence Google Fonts handwriting faces (Patrick Hand, Kalam, Caveat, Nothing You Could Do,
Cedarville Cursive, Reenie Beanie, Dawning of a New Day, La Belle Aurore). On top of those the renderer applies:
per-character jitter and rotation (print-style fonts), sinusoidal baseline warping, size variation,
horizontal squeeze, per-line tilt, low-frequency ink-opacity variation (faint ink), Gaussian blur,
smudges, paper tint and noise, optional ruled lines, page rotation (hard), and hand-drawn
strike-through strokes. Letterhead and labels are a printed font, so text is mixed printed and handwritten.

**Limitation:** font-based handwriting is more regular than a real doctor's hand. Treat scores here as
an upper bound, and add real photographed samples (written by you, with fictional content) for a harder test.

## 5. Using it for OCR / HTR testing

1. Run your OCR on `images/RX*.png`, preferably over the whole page, or after your own
   preprocessing (deskew, binarise, line segmentation). The PDF is for viewing or for testing PDF ingestion.
2. Save the predictions as JSON `{"RX001": "full predicted text", ...}` or CSV `prescription_id,predicted_text`.
   Line breaks do not matter: whitespace is normalised. If your model can flag unreadable words,
   output `[ILLEGIBLE]` or `[UNK]`.
3. Score:

```bash
pip install jiwer           # optional; evaluate.py has a built-in fallback
python evaluate.py preds.json                 # 'visible' reference
python evaluate.py preds.json --mode clean    # doctor's intended text (struck words removed)
python evaluate.py preds.json --no-header --per-item   # if you cropped below the letterhead
```

Example `pytesseract` baseline:

```python
import json, glob, os, pytesseract
from PIL import Image
preds = {os.path.basename(p)[:-4]: pytesseract.image_to_string(Image.open(p))
         for p in sorted(glob.glob("images/RX*.png"))}
json.dump(preds, open("preds_tesseract.json", "w"), indent=1)
```

### How predictions are compared with ground truth

| Metric | How it's computed |
|--------|-------------------|
| **CER** | Character edit distance ÷ reference length, after whitespace normalisation (jiwer.cer) |
| **WER** | Word edit distance ÷ reference word count (jiwer.wer) |
| **Med exact** | Share of ground-truth medicine names found as an exact (case-insensitive) word n-gram in the prediction |
| **Med fuzzy** | Same, with `difflib` similarity ≥ 0.80. Catches near-misses such as `Paracetamal` |

`[ILLEGIBLE]` handling: removed from the reference **and** from the prediction. A model that invents a
word there pays one insertion. A model that leaves it blank or flags it pays nothing.
Illegible medicine names are excluded from medicine accuracy.

Suggested extra checks you can build from `ground_truth.json`:
* **Field-level F1** for `frequency` and `duration` after your parser extracts them from the OCR text.
* **Look-alike confusion:** on `lookalike_name:*` pages, count how often the prediction contains the *partner* name instead.
* **Correction handling:** on `crossed_out_word` pages, check that your parser keeps the correction, not the struck word.
* **Digit accuracy:** on `ambiguous_digits` pages, compare the strengths exactly.

## 6. Suggested RAG pipeline tests

* **Preprocessing:** abbreviation expansion (OD → once daily, TDS → three times daily, 1-0-1 → morning and night,
  x 5/7 → for 5 days). Compare against `abbreviations` and `frequency`.
* **Chunking:** one chunk per prescription vs per medicine line (`medicines` gives the gold split).
* **Embeddings / retrieval:** query by medicine name with OCR noise (e.g. "Pantoprazol 40") and check
  the right RX is retrieved. Query with a look-alike partner name to test for false matches.
* **Terminology explanation:** ask the RAG system to explain each abbreviation and drug class, and
  judge the answers against your reference knowledge base, not against this dataset.
  The dataset deliberately contains no clinical guidance.

## 7. Licence and credits

Synthetic data generated for the RxReadable project. Handwriting fonts: Google Fonts, SIL Open Font Licence / Apache 2.0.
