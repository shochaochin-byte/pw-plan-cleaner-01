from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import streamlit as st

from cleaner.export import save_outputs
from cleaner.geometry_parser import parse_pdf_geometry
from cleaner.masking import colorize_overlay, mask_to_transparent_png
from cleaner.preview import render_pdf_page
from cleaner.raster_cleaner import clean_raster_image
from cleaner.segmentation import build_segmentation_masks
from cleaner.svg_export import export_debug_svg
from cleaner.vector_cleaner import clean_vector_pdf_bytes

st.set_page_config(page_title="PW-PLAN-CLEANER-01", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root {
  --pupa-blue:#3542FA; --pupa-red:#FF574A; --ink:#111111; --paper:#F7F3EA; --concrete:#B8B8B0;
}
.stApp {background: var(--paper); color: var(--ink); font-family:'IBM Plex Sans', sans-serif;}
h1,h2,h3{font-family:'IBM Plex Sans',sans-serif; letter-spacing:0.02em; text-transform:uppercase}
.small-label{font-family:'IBM Plex Mono', monospace; font-size:.8rem; letter-spacing:.08em; color:#333}
.tech-box{border:1px solid #888; padding:.75rem; background:rgba(255,255,255,.55)}
.red-marker{border-left:4px solid var(--pupa-red); padding-left:.6rem; margin:.35rem 0}
.blue-chip{display:inline-block; background:rgba(53,66,250,.12); color:var(--pupa-blue); border:1px solid var(--pupa-blue); padding:.1rem .4rem; font-family:'IBM Plex Mono'; font-size:.75rem}
.stButton>button{border:1px solid var(--ink); background:#fff; border-radius:0; color:var(--ink)}
hr{border:none; border-top:1px solid #999}
</style>
""",
    unsafe_allow_html=True,
)

st.title("PW-PLAN-CLEANER-01")
st.caption("AI-ASSISTED ARCHITECTURAL PLAN CLEANING SYSTEM — HALFTONE DUOTONE ARCHITECTURAL INTELLIGENCE")

if "state" not in st.session_state:
    st.session_state.state = {}

left, center, right = st.columns([1.1, 3.7, 1.2])

with left:
    st.markdown("#### Upload")
    uploaded = st.file_uploader("DROP PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown("#### Layers / Overlays")
    st.markdown("- **BLACK**: immutable architecture")
    st.markdown("- **BLUE**: intelligence / preserved zones")
    st.markdown("- **RED**: affected / removable hatch")

    st.markdown("#### Tools")
    mode = st.radio("Mode", ["Auto", "Vector", "Raster"], horizontal=False, label_visibility="collapsed")
    show_debug = st.toggle("DETECTION PREVIEW", value=True)
    sensitivity = st.slider("SENSITIVITY", 0.0, 1.0, 0.35, 0.01)

with right:
    st.markdown("#### Status")
    st.markdown("<span class='blue-chip'>PUPA BLUE INTELLIGENCE LAYER</span>", unsafe_allow_html=True)
    st.markdown("<div class='small-label'>RED is only used for changed/affected zones and warnings.</div>", unsafe_allow_html=True)

if uploaded:
    pdf_bytes = uploaded.read()
    before = render_pdf_page(pdf_bytes, 0, 2.0)
    gmeta = parse_pdf_geometry(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_drawings = len(doc[0].get_drawings())
    do_vector = (mode == "Vector") or (mode == "Auto" and first_drawings > 20)

    with center:
        st.markdown("#### Plan Workspace")
        st.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), use_container_width=True)
        if st.button("RUN PREPROCESSING"):
            if do_vector:
                cleaned_pdf, debug_data, decisions_by_page = clean_vector_pdf_bytes(pdf_bytes, sensitivity)
                cleaned = render_pdf_page(cleaned_pdf, 0, 2.0)
                decision_page = decisions_by_page[0] if decisions_by_page else []
                red = np.zeros(before.shape[:2], dtype=np.uint8)
                blue = np.zeros(before.shape[:2], dtype=np.uint8)
                red_boxes = []
                for d in decision_page:
                    x0, y0, x1, y1 = [int(v * 2) for v in d["bbox"]]
                    if d["remove"]:
                        red[y0:y1, x0:x1] = 255
                        red_boxes.append((x0, y0, x1, y1))
                    else:
                        blue[y0:y1, x0:x1] = 255
            else:
                cleaned, overlay_rb = clean_raster_image(before, sensitivity)
                cleaned_pdf_doc = fitz.open()
                p = cleaned_pdf_doc.new_page(width=before.shape[1], height=before.shape[0])
                ok, buf = cv2.imencode('.png', cleaned)
                if not ok:
                    st.error("WARNING: PNG encoding failed")
                    st.stop()
                p.insert_image(p.rect, stream=buf.tobytes())
                cleaned_pdf = cleaned_pdf_doc.tobytes()
                debug_data = [{"page": 0, "mode": "raster", "note": "fallback"}]
                red = cv2.inRange(overlay_rb, (0, 0, 200), (20, 20, 255))
                blue = cv2.inRange(overlay_rb, (200, 0, 0), (255, 30, 30))
                red_boxes = []

            masks = build_segmentation_masks(before, hatch_mask=red)
            green = masks.landscape_mask
            overlay = colorize_overlay(before, red, blue, green)
            transparent = mask_to_transparent_png(cleaned, cv2.bitwise_not(red))
            svg_path = export_debug_svg(before.shape[1], before.shape[0], red_boxes, Path("pw-plan-cleaner-01/outputs") / f"{Path(uploaded.name).stem}_debug.svg")

            outputs = save_outputs(
                Path("pw-plan-cleaner-01/outputs"),
                Path(uploaded.name).stem,
                cleaned_pdf,
                cleaned,
                {"vector_geometry": gmeta, "cleaner": debug_data},
                transparent_png=transparent,
                svg_path=svg_path,
                landscape_mask=masks.landscape_mask,
            )

            st.session_state.state = {
                "cleaned": cleaned,
                "overlay": overlay,
                "outputs": outputs,
                "debug": debug_data,
                "mode": "vector" if do_vector else "raster",
                "affected_count": int(np.count_nonzero(red > 0)),
            }

if st.session_state.state:
    data = st.session_state.state

    st.markdown("---")
    b1, b2, b3 = st.columns([1.2, 1.2, 1.6])
    with b1:
        st.markdown("#### Before / After")
        st.image(cv2.cvtColor(data["cleaned"], cv2.COLOR_BGR2RGB), use_container_width=True)
    with b2:
        st.markdown("#### Segmentation Preview")
        if show_debug:
            st.image(cv2.cvtColor(data["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
    with b3:
        st.markdown("#### OUTPUT / AFFECTED AREAS")
        st.markdown("<div class='red-marker'><b>AREA A</b> — FLOOR FINISH HATCH WILL BE REMOVED</div>", unsafe_allow_html=True)
        st.markdown("<div class='red-marker'><b>AREA B</b> — LANDSCAPE PROPOSAL CHANGE TARGET</div>", unsafe_allow_html=True)
        st.markdown("<div class='red-marker'><b>AREA C</b> — AI LANDSCAPE ENHANCEMENT TARGET</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-label'>AFFECTED PIXELS: {data['affected_count']}</div>", unsafe_allow_html=True)

        o = data["outputs"]
        st.download_button("EXPORT CLEANED PDF", data=Path(o["pdf"]).read_bytes(), file_name=Path(o["pdf"]).name)
        st.download_button("EXPORT PNG (TRANSPARENT)", data=Path(o["transparent_png"]).read_bytes(), file_name=Path(o["transparent_png"]).name)
        st.download_button("EXPORT SVG (VECTOR)", data=Path(o["svg"]).read_text(), file_name=Path(o["svg"]).name)
        st.download_button("EXPORT AI MASK", data=Path(o["landscape_mask"]).read_bytes(), file_name=Path(o["landscape_mask"]).name)
        st.download_button("EXPORT JSON", data=Path(o["debug"]).read_text(), file_name=Path(o["debug"]).name)

    if st.button("UNDO / RESET"):
        st.session_state.state = {}
        st.rerun()
