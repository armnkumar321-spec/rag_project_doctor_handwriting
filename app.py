"""RxReadable web app.  Run:  streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from PIL import Image

from rxreadable import config
from rxreadable.ocr import available_engines, run_ocr
from rxreadable.parser import parse_prescription
from rxreadable.preprocess import preprocess
from rxreadable.pipeline import get_rag

st.set_page_config(page_title="RxReadable", page_icon="💊", layout="wide")
st.title("💊 RxReadable")
st.caption("Doctor handwriting recognition and prescription understanding: OCR → parsing → RAG explanations")
st.warning(config.DISCLAIMER, icon="⚠️")

engines = available_engines()
with st.sidebar:
    st.header("Settings")
    if not engines:
        st.error("No OCR engine found. Add GEMINI_API_KEY to .env or install Tesseract.")
    engine = st.selectbox("OCR engine", engines or ["none"])
    show_pre = st.checkbox("Show preprocessed image", value=False)
    st.markdown(f"**Embeddings:** `{get_rag().embedder.name}`  \n**LLM:** "
                f"`{config.GEMINI_MODEL if config.GEMINI_API_KEY else 'none (offline extractive mode)'}`")
    st.divider()
    sample_dir = config.DATASET_DIR / "images"
    samples = sorted(p.name for p in sample_dir.glob("RX*.png")) if sample_dir.exists() else []
    sample = st.selectbox("…or try a dataset sample", ["—"] + samples)

tab_read, tab_ask = st.tabs(["📄 Read a prescription", "💬 Ask about medical terms"])

with tab_read:
    up = st.file_uploader("Upload a prescription photo or scan", type=["png", "jpg", "jpeg", "webp"])
    img, pid = None, None
    if up:
        img, pid = Image.open(up), up.name
    elif sample != "—":
        img, pid = Image.open(sample_dir / sample), sample[:-4]

    if img is not None and engines:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption=pid, width="stretch")
            if show_pre:
                st.image(preprocess(img)[0], caption="Preprocessed (used by Tesseract)", width="stretch")
        with c2:
            with st.spinner(f"Running OCR with {engine}…"):
                key = f"{pid}-{engine}"
                if st.session_state.get("key") != key:
                    st.session_state.ocr = run_ocr(img, engine)
                    st.session_state.key = key
                    st.session_state.pop("expl", None)
            text = st.text_area("OCR text (you can correct it, parsing updates live)",
                                st.session_state.ocr["text"], height=260)
            parsed = parse_prescription(text)
            st.markdown(f"**Patient:** {parsed['patient'] or '—'} &nbsp; **Age/Sex:** {parsed['age_sex'] or '—'} "
                        f"&nbsp; **Date:** {parsed['date'] or '—'}")
            if parsed["diagnosis"]:
                st.markdown(f"**Diagnosis:** {parsed['diagnosis']}")

        st.subheader("Medicines found")
        if parsed["medicines"]:
            df = pd.DataFrame([{k: m[k] for k in ("name", "form", "strength", "dose", "frequency", "timing",
                                                    "duration", "match_score")} for m in parsed["medicines"]])
            st.dataframe(df, width="stretch", hide_index=True)
            for m in parsed["medicines"]:
                with st.expander(f"{m['name']}: {m['plain_english']}"):
                    st.write(f"Written as: `{m['raw_line']}`")
                    for w in m["warnings"]:
                        st.warning(w)
                    st.json(m["drug_info"])
        else:
            st.info("No medicines recognised. Try another OCR engine or correct the text above.")
        if parsed["advice"]:
            st.markdown("**Advice:** " + " · ".join(parsed["advice"]))

        if st.button("🧠 Explain this prescription (RAG)", type="primary", disabled=not parsed["medicines"]):
            with st.spinner("Retrieving and explaining…"):
                st.session_state.expl = get_rag().explain(parsed)
        if "expl" in st.session_state:
            st.markdown(st.session_state.expl["answer"])
            with st.expander("Retrieved sources"):
                for h in st.session_state.expl["sources"]:
                    st.write(f"`{h['id']}` ({h['score']:.2f}) {h['text']}")
        st.download_button("Download result (JSON)", json.dumps({"ocr": st.session_state.ocr, "parsed": parsed},
                                                                indent=2, default=str), f"{pid}_rxreadable.json")

with tab_ask:
    q = st.text_input("Ask, e.g. 'What does TDS mean?' or 'What is pantoprazole used for?'")
    if q:
        res = get_rag().answer(q)
        st.markdown(res["answer"])
        with st.expander("Retrieved sources"):
            for h in res["sources"]:
                st.write(f"`{h['id']}` ({h['score']:.2f}) {h['text']}")
