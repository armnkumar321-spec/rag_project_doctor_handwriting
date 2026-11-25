# 💊 RxReadable

**Doctor handwriting recognition and prescription understanding.** Upload a photo of a prescription.
RxReadable reads it (OCR/HTR), pulls out every medicine with its strength, dose, frequency and duration,
flags risky readings (look-alike drug names, ambiguous digits, illegible words), and explains it in
plain English using RAG over a medical-terms knowledge base.

> ⚠️ Academic project. Readings and explanations can be wrong and are **not medical advice**.
> Always confirm a prescription with the doctor or a pharmacist.

## Live demo

Browser version (parser + knowledge-base search in JavaScript, 8 dataset samples): deployed on Vercel from `web/`.
The full pipeline with live OCR and Gemini runs locally with `streamlit run app.py` (see Setup).

## Architecture

```
image ──► preprocess (OpenCV) ──► OCR engine ──► text cleanup ──► parser ──► structured medicines
            denoise, shadow        Gemini vision     ~~struck~~       fuzzy drug match   + warnings
            removal, CLAHE,        or Tesseract      removal, OCR     (rapidfuzz), regex  + plain English
            deskew, binarise                         fixes            dose/freq/duration
                                                                          │
knowledge base (drugs + abbreviations) ─► chunking ─► embeddings ─► vector store ─► retrieve ─► Gemini answer
                                          per_entity   Gemini / ST /   NumPy or       (or offline
                                          sentences    TF-IDF char     Chroma          extractive answer)
                                          fixed        n-grams
```

| Folder / file | What it does |
|---|---|
| `rxreadable/preprocess.py` | Grayscale, denoise, background (shadow) removal, CLAHE, deskew, adaptive threshold |
| `rxreadable/ocr.py` | `gemini` (vision LLM, best for handwriting) and `tesseract` (offline baseline); `auto` picks the best available |
| `rxreadable/parser.py` | Medicine lines → `{name, form, strength, dose, frequency, timing, duration}`, wrapped-line merging, abbreviation expansion, warnings |
| `rxreadable/chunking.py` | Three chunking strategies + per-prescription chunks |
| `rxreadable/embeddings.py` | Gemini embeddings, sentence-transformers, or TF-IDF character n-grams (works offline, tolerant of OCR typos) |
| `rxreadable/vectorstore.py` | Small NumPy cosine vector DB with save/load; optional ChromaDB |
| `rxreadable/rag.py` | Retrieval + grounded generation with Gemini; offline extractive fallback |
| `app.py` | Streamlit web app |
| `web/index.html` | Static browser demo deployed on Vercel (`vercel.json`) |
| `cli.py` | Command-line interface |
| `scripts/run_eval.py` | OCR + parsing evaluation on the dataset (CER, WER, field accuracy per difficulty) |
| `scripts/compare_chunking.py` | Retrieval hit@1 / hit@3 for each chunking strategy × embedding backend |
| `data/kb/` | Knowledge base: 33 drugs (general info, look-alike pairs) and 43 abbreviations |
| `data/RxReadable_Dataset/` | 50 synthetic handwritten prescriptions with ground truth |
| `tests/` | pytest unit tests for the parser |

## Setup (Windows / Mac / Linux)

```bash
cd RxReadable
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # Mac/Linux: cp .env.example .env
```

Then choose **at least one** OCR engine:

* **Gemini (recommended, much better on handwriting):** get a free API key at
  https://aistudio.google.com/apikey and set `GEMINI_API_KEY=...` in `.env`. This also turns on
  Gemini embeddings and LLM explanations.
* **Tesseract (offline):** install it (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki),
  then set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env` if it isn't on PATH.

Without a Gemini key, everything still runs: Tesseract OCR, TF-IDF embeddings, and extractive answers.

## Run

```bash
streamlit run app.py                                        # web app at http://localhost:8501
python cli.py read data/RxReadable_Dataset/images/RX020.png # read one prescription
python cli.py ask "What does TDS mean?"                     # RAG question
python cli.py build-index --strategy sentences              # rebuild the vector index
python -m pytest -q                                         # tests
```

In the app, pick a dataset sample from the sidebar or upload your own photo. You can correct the OCR text
in the text box and the parsed table updates live. Then press **Explain this prescription**.

## Evaluate

```bash
python scripts/run_eval.py --parser-only        # parser on ground-truth text (upper bound)
python scripts/run_eval.py --engine tesseract   # full OCR + parsing on all 50 pages
python scripts/run_eval.py --engine gemini      # same with Gemini (about 1–2 min)
python scripts/compare_chunking.py tfidf gemini # chunking × embeddings retrieval test
```

Results are written to `outputs/`. You can also score any OCR output with the dataset's own script:
`python data/RxReadable_Dataset/evaluate.py outputs/preds_tesseract.json`.

### Baseline results (measured)

| Setup | Level | CER | WER | Name | Strength | Freq | Duration |
|---|---|---:|---:|---:|---:|---:|---:|
| Parser only (perfect text) | all | 0 | 0 | 100% | 100% | 97% | 97% |
| Tesseract + preprocessing | easy | 0.083 | 0.153 | 100% | 89% | 85% | 81% |
| | medium | 0.203 | 0.479 | 64% | 13% | 26% | 13% |
| | hard | 0.486 | 0.823 | 0% | 0% | 0% | 0% |

| Retrieval (TF-IDF, 15 queries with typos) | hit@1 | hit@3 |
|---|---:|---:|
| per_entity chunks | 87% | 100% |
| fixed 200-char chunks | 87% | 93% |
| 2-sentence chunks | 60% | 87% |

Tesseract is a printed-text OCR, so it collapses on cursive handwriting. That is exactly the gap a
handwriting-capable model (Gemini vision, TrOCR, etc.) should close. Run the Gemini evaluation and add
its row to this table for your report.

## Ideas to extend

* Add a TrOCR engine (`microsoft/trocr-base-handwritten`) with line segmentation, and compare it with Gemini.
* Fine-tune on the synthetic dataset, then test on real photos you write yourself (fictional content).
* Confidence-based human review: highlight low-confidence medicine lines in red in the app.
* Swap the NumPy store for Chroma (`VECTOR_STORE=chroma`, `pip install chromadb`).
