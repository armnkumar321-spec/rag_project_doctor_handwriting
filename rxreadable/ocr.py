"""OCR / HTR engines. Gemini (vision LLM) is far better on handwriting; Tesseract is the offline baseline."""
import io
from PIL import Image
from . import config
from .preprocess import preprocess

GEMINI_OCR_PROMPT = """You are an OCR engine for handwritten medical prescriptions.
Transcribe ALL text in the image exactly as written, top to bottom, one output line per written line.
Rules:
- Do not correct spelling, expand abbreviations, or reorder anything. Keep case as written.
- If a word is crossed out and a correction is written near it, write: ~~crossed~~ correction
- If a word is unreadable, write [ILLEGIBLE]. Never guess an unreadable word.
- Ignore signature scribbles. Output only the transcription, no commentary."""


def available_engines():
    engines = []
    if config.GEMINI_API_KEY:
        engines.append("gemini")
    try:
        import pytesseract
        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        pytesseract.get_tesseract_version()
        engines.append("tesseract")
    except Exception:
        pass
    return engines


def resolve_engine(name=None):
    name = (name or config.OCR_ENGINE).lower()
    avail = available_engines()
    if name == "auto":
        if not avail:
            raise RuntimeError("No OCR engine available. Set GEMINI_API_KEY in .env or install Tesseract.")
        return avail[0]
    if name not in avail:
        raise RuntimeError(f"OCR engine '{name}' not available (available: {avail or 'none'}).")
    return name


def ocr_tesseract(img, do_preprocess=True):
    import pytesseract
    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    info = {}
    if do_preprocess:
        img, info = preprocess(img)
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 4")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines), info


def ocr_gemini(img):
    from google.genai import types
    client = config.gemini_client()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), GEMINI_OCR_PROMPT],
        config=types.GenerateContentConfig(temperature=0),
    )
    return (resp.text or "").strip(), {"model": config.GEMINI_MODEL}


def run_ocr(img, engine=None, do_preprocess=True):
    """img: PIL image or path. Returns dict(text, engine, info)."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    eng = resolve_engine(engine)
    text, info = ocr_gemini(img) if eng == "gemini" else ocr_tesseract(img, do_preprocess)
    return {"text": text, "engine": eng, "info": info}
