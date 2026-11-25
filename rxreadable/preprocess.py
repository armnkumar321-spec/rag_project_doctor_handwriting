"""Image preprocessing for handwritten prescriptions: grayscale, denoise, deskew, contrast, binarise."""
import cv2
import numpy as np
from PIL import Image


def to_cv(img):
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def estimate_skew(gray):
    """Estimate page skew (degrees) from the dominant angle of ink pixels."""
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle > 45:
        angle -= 90
    return float(angle) if abs(angle) < 15 else 0.0


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(img, binarize=True, deskew=True, upscale=1.5):
    """Return (processed PIL image, info dict). Good default input for Tesseract-style OCR."""
    bgr = to_cv(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # remove uneven background (shadows, smudges, paper tint)
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    angle = estimate_skew(norm) if deskew else 0.0
    if angle:
        norm = rotate(norm, angle)
    out = norm
    if binarize:
        out = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(out), {"skew_deg": round(angle, 2), "upscale": upscale, "binarized": binarize}
