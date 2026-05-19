from __future__ import annotations

from pathlib import Path
import json
import streamlit as st
import fitz
import cv2
import numpy as np

from cleaner.preview import render_pdf_page
from cleaner.vector_cleaner import clean_vector_pdf_bytes
from cleaner.raster_cleaner import clean_raster_image
from cleaner.export import save_outputs

st.set_page_config(page_title="PW-PLAN-CLEANER-01", layout="wide")
st.title("PW-PLAN-CLEANER-01 — Architectural Hatch Cleaner")

if "state" not in st.session_state:
    st.session_state.state = {}

uploaded = st.file_uploader("Upload PDF plan", type=["pdf"])
sensitivity = st.slider("Sensitivity (conservative default)", 0.0, 1.0, 0.35, 0.01)
mode = st.radio("Mode", ["Auto", "Vector", "Raster"], horizontal=True)

col_a, col_b = st.columns(2)

if uploaded:
    pdf_bytes = uploaded.read()
    with col_a:
        st.subheader("Before")
        before = render_pdf_page(pdf_bytes, page_index=0, zoom=2.0)
        st.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), use_container_width=True)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first = doc[0]
    drawings = first.get_drawings()
    is_vectorish = len(drawings) > 20

    do_vector = (mode == "Vector") or (mode == "Auto" and is_vectorish)

    if st.button("Run Cleaner"):
        if do_vector:
            cleaned_pdf, debug_data, _ = clean_vector_pdf_bytes(pdf_bytes, sensitivity=sensitivity)
            cleaned_preview = render_pdf_page(cleaned_pdf, page_index=0, zoom=2.0)
        else:
            cleaned_preview, overlay = clean_raster_image(before, sensitivity=sensitivity)
            tmp = fitz.open()
            p = tmp.new_page(width=before.shape[1], height=before.shape[0])
            ok, buf = cv2.imencode(".png", cleaned_preview)
            if not ok:
                st.error("Failed to encode raster preview")
                st.stop()
            p.insert_image(p.rect, stream=buf.tobytes())
            cleaned_pdf = tmp.tobytes()
            debug_data = [{"mode": "raster", "note": "fallback mode used"}]

        st.session_state.state = {
            "cleaned_pdf": cleaned_pdf,
            "cleaned_preview": cleaned_preview,
            "debug_data": debug_data,
            "filename": Path(uploaded.name).stem,
        }

if st.session_state.state:
    data = st.session_state.state
    with col_b:
        st.subheader("After")
        st.image(cv2.cvtColor(data["cleaned_preview"], cv2.COLOR_BGR2RGB), use_container_width=True)

    outputs = save_outputs(Path("pw-plan-cleaner-01/outputs"), data["filename"], data["cleaned_pdf"], data["cleaned_preview"], data["debug_data"])

    st.download_button("Download cleaned PDF", data=Path(outputs["pdf"]).read_bytes(), file_name=Path(outputs["pdf"]).name, mime="application/pdf")
    st.download_button("Download cleaned PNG", data=Path(outputs["png"]).read_bytes(), file_name=Path(outputs["png"]).name, mime="image/png")
    st.download_button("Download debug JSON", data=Path(outputs["debug"]).read_text(encoding="utf-8"), file_name=Path(outputs["debug"]).name, mime="application/json")

    st.expander("Debug JSON").json(data["debug_data"])

    if st.button("Undo / Reset"):
        st.session_state.state = {}
        st.rerun()
