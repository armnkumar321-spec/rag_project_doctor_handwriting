import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_DIR = ROOT / "data" / "kb"
INDEX_DIR = ROOT / "index"
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "data" / "RxReadable_Dataset"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").lower()
EMBEDDINGS = os.getenv("EMBEDDINGS", "auto").lower()
VECTOR_STORE = os.getenv("VECTOR_STORE", "numpy").lower()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

DISCLAIMER = ("RxReadable is an academic project. Its readings and explanations can be wrong and are "
              "not medical advice. Always confirm a prescription with the doctor or a pharmacist.")


def gemini_client():
    if not GEMINI_API_KEY:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)
