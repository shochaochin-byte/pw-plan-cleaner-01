from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import streamlit as st

from cleaner.export import save_outputs
from cleaner.geometry_parser import parse_pdf_geometry
from cleaner.halftone_duotone import duotone_flat, halftone_duotone
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root {
  --pupa-blue:#3542FA;
  --pupa-red:#E8382A;
  --ink:#111111;
  --paper:#F5F0E8;
  --panel:#EDE8DC;
  --border:#BBBBA8;
  --concrete:#9A9A8E;
}

/* ── Base ── */
.stApp { background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans',sans-serif; }
.stApp * { box-sizing:border-box; }
[data-testid="stAppViewContainer"] { background:var(--paper); }
[data-testid="stHeader"] { background:var(--paper); border-bottom:1px solid var(--border); }
section[data-testid="stSidebar"] { display:none; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ── Top header bar ── */
.pw-header {
  display:flex; align-items:flex-start; justify-content:space-between;
  padding:.75rem 1.25rem .5rem;
  border-bottom:2px solid var(--ink);
  background:var(--paper);
}
.pw-header-left h1 {
  font-family:'IBM Plex Sans',sans-serif;
  font-size:1.6rem; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase; margin:0; line-height:1;
}
.pw-header-left .pw-sub {
  font-size:.62rem; letter-spacing:.12em; color:#444; margin-top:.15rem;
  font-family:'IBM Plex Mono',monospace;
}
.pw-header-meta {
  display:flex; gap:2rem; align-items:center; font-family:'IBM Plex Mono',monospace;
  font-size:.62rem; letter-spacing:.1em; color:#555; margin-top:.3rem;
}
.pw-header-meta span b { color:var(--ink); }
.pw-status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--pupa-red); margin-right:.3rem; }
.pw-brand {
  text-align:right; line-height:1.1;
  font-family:'IBM Plex Sans',sans-serif; font-weight:700; font-size:1.1rem;
  letter-spacing:.1em; text-transform:uppercase;
}
.pw-brand-dot {
  display:inline-block; width:28px; height:28px; border-radius:50%;
  background:var(--pupa-red); margin-left:.4rem; vertical-align:middle;
}
.pw-brand-sub { font-size:.55rem; letter-spacing:.12em; color:#555; font-weight:400; margin-top:.1rem; }

/* ── Section label ── */
.sec-label {
  font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.14em;
  text-transform:uppercase; color:#444; border-bottom:1px solid var(--border);
  padding-bottom:.2rem; margin-bottom:.5rem; margin-top:.75rem;
}
.sec-label:first-child { margin-top:0; }

/* ── Upload drop zone ── */
.upload-zone {
  border:1.5px dashed var(--border); background:rgba(255,255,255,.5);
  padding:1.2rem .6rem; text-align:center; font-family:'IBM Plex Mono',monospace;
  font-size:.68rem; letter-spacing:.08em; color:#666; line-height:1.6;
}
.upload-zone .upload-icon { font-size:1.6rem; margin-bottom:.2rem; display:block; }
[data-testid="stFileUploader"] {
  border:1.5px dashed var(--border) !important; background:rgba(255,255,255,.5) !important;
  padding:.5rem !important; border-radius:0 !important;
}
[data-testid="stFileUploader"] label { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.08em; }
[data-testid="stFileDropzoneInstructions"] { font-size:.68rem !important; }

/* ── Layer rows ── */
.layer-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:.3rem 0; border-bottom:1px solid rgba(0,0,0,.07); font-size:.72rem;
  letter-spacing:.04em; font-family:'IBM Plex Mono',monospace;
}
.layer-row.active { color:var(--pupa-red); font-weight:600; }
.layer-row.blue   { color:var(--pupa-blue); }
.layer-icon { font-size:.85rem; }
.layer-swatch {
  width:20px; height:14px; border:1px solid var(--border);
  display:inline-block; margin-right:.4rem; flex-shrink:0;
}
.swatch-hatch { background: repeating-linear-gradient(45deg,#E8382A,#E8382A 2px,transparent 2px,transparent 6px); }
.swatch-arch  { background:#111; }
.swatch-land  { background: repeating-linear-gradient(0deg,#3542FA,#3542FA 2px,transparent 2px,transparent 5px); }
.swatch-seg   { background: repeating-linear-gradient(90deg,#888,#888 1px,transparent 1px,transparent 4px); }

/* ── Tool rows ── */
.tool-row {
  display:flex; align-items:center; gap:.45rem;
  padding:.28rem .4rem; font-size:.72rem; letter-spacing:.06em;
  font-family:'IBM Plex Mono',monospace; cursor:pointer; border-radius:0;
}
.tool-row.active { background:rgba(232,56,42,.13); color:var(--pupa-red); font-weight:600; }
.tool-icon { font-size:.85rem; }

/* ── Right panel sliders ── */
.slider-label {
  display:flex; justify-content:space-between; align-items:center;
  font-family:'IBM Plex Mono',monospace; font-size:.65rem; letter-spacing:.1em;
  text-transform:uppercase; color:#333; margin-bottom:-.3rem; margin-top:.55rem;
}
.slider-val { font-weight:600; color:var(--ink); }

/* ── Slider: target only thumb and track, never text containers ── */
[data-testid="stSlider"] [role="slider"] {
  background:var(--pupa-red) !important; border:2px solid var(--pupa-red) !important;
  width:14px !important; height:14px !important; border-radius:50% !important;
  box-shadow:none !important;
}
/* filled track portion */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
  background:rgba(232,56,42,.22) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
  background:var(--pupa-red) !important;
}

/* ── Ensure all widget labels stay readable ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stToggle"] label,
[data-testid="stToggle"] p,
[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stColorPicker"] label,
[data-testid="stColorPicker"] p,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
.stMarkdown p, .stMarkdown span,
[data-testid="stText"] {
  color: var(--ink) !important;
  font-family: 'IBM Plex Mono', monospace;
}

/* ── Toggle ── */
[data-testid="stToggle"] label { font-size:.7rem; letter-spacing:.08em; }
[data-testid="stToggle"] [role="switch"][aria-checked="true"] { background:var(--pupa-red) !important; }

/* ── Export buttons ── */
.export-row {
  display:flex; align-items:center; justify-content:space-between;
  border:1px solid var(--border); padding:.35rem .5rem; margin:.2rem 0;
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.06em;
  background:rgba(255,255,255,.5);
}
.export-tag { color:var(--pupa-red); font-weight:700; font-size:.68rem; }
[data-testid="stDownloadButton"] > button {
  width:100%; border:1px solid var(--border) !important; background:rgba(255,255,255,.5) !important;
  border-radius:0 !important; color:var(--ink) !important;
  font-family:'IBM Plex Mono',monospace !important; font-size:.68rem !important;
  letter-spacing:.06em !important; text-transform:uppercase !important;
  padding:.35rem .5rem !important; text-align:left !important;
  display:flex; justify-content:space-between;
}
[data-testid="stDownloadButton"] > button:hover { background:rgba(232,56,42,.08) !important; border-color:var(--pupa-red) !important; }

/* ── Run button ── */
.stButton > button {
  border:1.5px solid var(--ink) !important; background:var(--ink) !important;
  color:#fff !important; border-radius:0 !important;
  font-family:'IBM Plex Mono',monospace !important; font-size:.72rem !important;
  letter-spacing:.1em !important; text-transform:uppercase !important;
  padding:.5rem 1.2rem !important; width:100%;
}
.stButton > button:hover { background:var(--pupa-red) !important; border-color:var(--pupa-red) !important; }

/* ── Image containers ── */
[data-testid="stImage"] { border:1px solid var(--border); }

/* ── Red marker ── */
.red-marker {
  border-left:3px solid var(--pupa-red); padding:.35rem .6rem; margin:.3rem 0;
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.04em;
  background:rgba(232,56,42,.04);
}
.red-marker b { color:var(--pupa-red); }

/* ── Small labels ── */
.small-label { font-family:'IBM Plex Mono',monospace; font-size:.65rem; letter-spacing:.08em; color:#555; }

/* ── Bottom footer ── */
.pw-footer {
  border-top:2px solid var(--ink); background:var(--ink); color:#fff;
  display:flex; align-items:center; justify-content:space-between;
  padding:.55rem 1.25rem; font-family:'IBM Plex Mono',monospace;
  font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; margin-top:1rem;
}

/* ── Legend items ── */
.legend-item {
  display:flex; align-items:center; gap:.4rem;
  font-family:'IBM Plex Mono',monospace; font-size:.65rem; letter-spacing:.06em; margin:.2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Top header ──
st.markdown(
    """
<div class="pw-header">
  <div class="pw-header-left">
    <h1>PW-PLAN-CLEANER-01</h1>
    <div class="pw-sub">AI-ASSISTED ARCHITECTURAL PLAN CLEANING SYSTEM</div>
    <div class="pw-header-meta">
      <span><b>PROJECT:</b> PUPA WORKS – LANDSCAPE PROPOSAL</span>
      <span><b>FILE:</b> LAYOUT PLAN LEVEL 1.pdf</span>
      <span><span class="pw-status-dot"></span><b>STATUS:</b> HATCH DETECTION</span>
    </div>
  </div>
  <div class="pw-brand">
    P U P A<br>W O R K S <span class="pw-brand-dot"></span>
    <div class="pw-brand-sub">LANDSCAPE ARCHITECTURE<br>+ DESIGN</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if "state" not in st.session_state:
    st.session_state.state = {}

left, center, right = st.columns([1.15, 3.6, 1.25])

with left:
    st.markdown("<div class='sec-label'>↑ UPLOAD</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("DROP PDF HERE or click to upload", type=["pdf"], label_visibility="collapsed")

    st.markdown("""
<div class='sec-label'>LAYERS / OVERLAYS</div>
<div class='layer-row'>
  <span><span class='layer-swatch' style='background:#ddd'></span>ORIGINAL PDF</span>
  <span class='layer-icon'>👁</span>
</div>
<div class='layer-row active'>
  <span><span class='layer-swatch swatch-hatch'></span>HATCH DETECTION</span>
  <span class='layer-icon' style='color:var(--pupa-red)'>👁</span>
</div>
<div class='layer-row'>
  <span><span class='layer-swatch swatch-arch'></span>PRESERVED ARCHITECTURE</span>
  <span class='layer-icon'>👁</span>
</div>
<div class='layer-row blue'>
  <span><span class='layer-swatch swatch-land'></span>LANDSCAPE ZONES</span>
  <span class='layer-icon'>👁</span>
</div>
<div class='layer-row'>
  <span><span class='layer-swatch swatch-seg'></span>SEGMENTATION ZONES</span>
  <span class='layer-icon'>👁</span>
</div>
<div class='layer-row blue'>
  <span><span class='layer-swatch' style='background:#bbb'></span>DIMENSIONS / TEXT</span>
  <span class='layer-icon'>👁</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='sec-label' style='margin-top:.9rem'>TOOLS</div>
<div class='tool-row active'><span class='tool-icon'>⊞</span> HATCH DETECTION</div>
<div class='tool-row'><span class='tool-icon'>⌁</span> VECTOR CLEANER</div>
<div class='tool-row'><span class='tool-icon'>◈</span> SEGMENTATION</div>
<div class='tool-row'><span class='tool-icon'>◻</span> MASKING</div>
<div class='tool-row'><span class='tool-icon'>👁</span> PREVIEW</div>
<div class='tool-row'><span class='tool-icon'>↗</span> EXPORT</div>
""", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Auto", "Vector", "Raster"], horizontal=True, label_visibility="collapsed")

with right:
    st.markdown("<div class='sec-label'>HATCH DETECTION</div>", unsafe_allow_html=True)

    sensitivity = st.slider("SENSITIVITY", 0.0, 1.0, 0.68, 0.01, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>SENSITIVITY</span><span class='slider-val'>68%</span></div>", unsafe_allow_html=True)

    min_line = st.slider("MIN LINE LENGTH", 0, 64, 16, 1, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>MIN LINE LENGTH</span><span class='slider-val'>16 px</span></div>", unsafe_allow_html=True)

    density = st.slider("DENSITY THRESHOLD", 0.0, 1.0, 0.42, 0.01, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>DENSITY THRESHOLD</span><span class='slider-val'>42%</span></div>", unsafe_allow_html=True)

    angle_c = st.slider("ANGLE CONSISTENCY", 0.0, 1.0, 0.63, 0.01, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>ANGLE CONSISTENCY</span><span class='slider-val'>63%</span></div>", unsafe_allow_html=True)

    show_debug = st.toggle("DETECTION PREVIEW", value=True)

    st.markdown("""
<div class='sec-label' style='margin-top:.75rem'>LEGEND</div>
<div class='legend-item'><span class='layer-swatch swatch-hatch'></span> REMOVABLE HATCH</div>
<div class='legend-item'><span class='layer-swatch swatch-arch'></span> PRESERVED ARCHITECTURE</div>
<div class='legend-item'><span class='layer-swatch swatch-land'></span> LANDSCAPE ZONES</div>
<div class='legend-item'><span class='layer-swatch swatch-seg'></span> SEGMENTATION ZONES</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:.9rem'>HALFTONE DUOTONE PRINT</div>", unsafe_allow_html=True)
    enable_halftone = st.toggle("ENABLE HALFTONE", value=False)
    halftone_mode = st.radio("Render", ["Halftone (AM dots)", "Flat bitmap"], horizontal=False, label_visibility="collapsed")
    ink_dark  = st.color_picker("INK DARK",  "#1B1F5E", label_visibility="visible")
    ink_light = st.color_picker("INK LIGHT", "#E8382A", label_visibility="visible")
    ht_cell   = st.slider("CELL SIZE (px)", 6, 24, 10, 1, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>CELL SIZE</span><span class='slider-val'>{} px</span></div>".format(ht_cell), unsafe_allow_html=True)

if uploaded:
    pdf_bytes = uploaded.read()
    before = render_pdf_page(pdf_bytes, 0, 2.0)
    gmeta = parse_pdf_geometry(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_drawings = len(doc[0].get_drawings())
    do_vector = (mode == "Vector") or (mode == "Auto" and first_drawings > 20)

    with center:
        st.markdown("<div class='sec-label'>PLAN WORKSPACE</div>", unsafe_allow_html=True)
        if enable_halftone:
            with st.spinner("Rendering halftone duotone…"):
                hatch_for_ht = None
                if "state" in st.session_state and st.session_state.state:
                    # reuse previous run's red mask stored via overlay diff
                    pass
                if halftone_mode.startswith("Flat"):
                    ht_preview = duotone_flat(before, color_dark=ink_dark, color_light=ink_light)
                else:
                    ht_preview = halftone_duotone(before, cell=ht_cell, color_dark=ink_dark, color_light=ink_light)
            st.image(cv2.cvtColor(ht_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), use_container_width=True)
        if st.button("▶  RUN PREPROCESSING"):
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

            if halftone_mode.startswith("Flat"):
                ht_result = duotone_flat(before, hatch_mask=red, color_dark=ink_dark, color_light=ink_light)
            else:
                ht_result = halftone_duotone(before, hatch_mask=red, cell=ht_cell, color_dark=ink_dark, color_light=ink_light)

            st.session_state.state = {
                "before": before,
                "cleaned": cleaned,
                "overlay": overlay,
                "halftone": ht_result,
                "outputs": outputs,
                "debug": debug_data,
                "mode": "vector" if do_vector else "raster",
                "affected_count": int(np.count_nonzero(red > 0)),
            }

if st.session_state.state:
    data = st.session_state.state

    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:.75rem 0'>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns([1.15, 1.15, 1.15, 1.55])

    with b1:
        st.markdown("<div class='sec-label'>BEFORE / AFTER</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(data["before"], cv2.COLOR_BGR2RGB), caption="ORIGINAL PDF", use_container_width=True)
        st.image(cv2.cvtColor(data["cleaned"], cv2.COLOR_BGR2RGB), caption="CLEANED PLAN (HATCH REMOVED)", use_container_width=True)

    with b2:
        st.markdown("<div class='sec-label'>SEGMENTATION PREVIEW</div>", unsafe_allow_html=True)
        if show_debug:
            st.image(cv2.cvtColor(data["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("""
<div class='legend-item'><span class='layer-swatch swatch-arch'></span> INDOOR</div>
<div class='legend-item'><span class='layer-swatch swatch-land'></span> OUTDOOR</div>
<div class='legend-item'><span class='layer-swatch swatch-seg'></span> CIRCULATION</div>
<div class='legend-item'><span class='layer-swatch swatch-hatch'></span> LANDSCAPE</div>
<div class='legend-item'><span class='layer-swatch' style='background:#a8c4e0'></span> WATER</div>
""", unsafe_allow_html=True)

    with b3:
        st.markdown("<div class='sec-label'>HALFTONE DUOTONE PRINT</div>", unsafe_allow_html=True)
        if "halftone" in data:
            ht_rgb = cv2.cvtColor(data["halftone"], cv2.COLOR_BGR2RGB)
            st.image(ht_rgb, use_container_width=True)
            ok, ht_buf = cv2.imencode(".png", data["halftone"])
            if ok:
                st.download_button(
                    "EXPORT HALFTONE PNG  ↗ PNG",
                    data=ht_buf.tobytes(),
                    file_name="halftone_duotone.png",
                    mime="image/png",
                )
        elif show_debug:
            st.image(cv2.cvtColor(data["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)

    with b4:
        st.markdown("<div class='sec-label'>OUTPUT / EFFECTED AREAS</div>", unsafe_allow_html=True)
        st.markdown("<div class='red-marker'><b>AREA A</b><br>FLOOR FINISH HATCH<br>WILL BE REMOVED</div>", unsafe_allow_html=True)
        st.markdown("<div class='red-marker'><b>AREA B</b><br>SIGNIFICANT CHANGE<br>PROPOSED</div>", unsafe_allow_html=True)
        st.markdown("<div class='red-marker'><b>AREA C</b><br>AI LANDSCAPE<br>ENHANCEMENT TARGET</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-label' style='margin:.5rem 0'>AFFECTED PIXELS: {data['affected_count']}</div>", unsafe_allow_html=True)

        o = data["outputs"]
        st.markdown("<div class='sec-label' style='margin-top:.75rem'>EXPORT OPTIONS</div>", unsafe_allow_html=True)
        st.download_button("EXPORT CLEANED PDF  ↗ PDF", data=Path(o["pdf"]).read_bytes(), file_name=Path(o["pdf"]).name)
        st.download_button("EXPORT PNG (TRANSPARENT)  ↗ PNG", data=Path(o["transparent_png"]).read_bytes(), file_name=Path(o["transparent_png"]).name)
        st.download_button("EXPORT SVG (VECTOR)  ↗ SVG", data=Path(o["svg"]).read_text(), file_name=Path(o["svg"]).name)
        st.download_button("EXPORT AI MASK (LANDSCAPE)  ↗ MASK", data=Path(o["landscape_mask"]).read_bytes(), file_name=Path(o["landscape_mask"]).name)
        st.download_button("EXPORT METADATA (JSON)  ↗ JSON", data=Path(o["debug"]).read_text(), file_name=Path(o["debug"]).name)

        if st.button("↺  UNDO / RESET"):
            st.session_state.state = {}
            st.rerun()

# ── Footer ──
st.markdown("""
<div class='pw-footer'>
  <span>PW-PLAN-CLEANER-01</span>
  <span>PRESERVATION FIRST. AI READY. LANDSCAPE POSSIBLE.</span>
  <span>PUPA WORKS © 2025</span>
</div>
""", unsafe_allow_html=True)
