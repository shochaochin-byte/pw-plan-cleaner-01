from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import streamlit as st

from cleaner.ai_proposal import BACKENDS, ProposalResult
from cleaner.export import save_outputs, save_landscape_bundle
from cleaner.geometry_parser import parse_pdf_geometry
from cleaner.halftone_duotone import duotone_flat, halftone_duotone
from cleaner.layer_reader import read_pdf_layers, render_with_visibility
from cleaner.mask_package import build_mask_package, composite_masks_preview
from cleaner.masking import colorize_overlay, mask_to_transparent_png
from cleaner.preview import render_pdf_page
from cleaner.raster_cleaner import clean_raster_image
from cleaner.svg_export import export_debug_svg
from cleaner.vector_cleaner import clean_vector_pdf_bytes
from cleaner.zone_detector import auto_detect_zones, assign_tiers, canvas_json_to_mask, flood_fill_zone, merge_zone_masks


@st.cache_data(show_spinner=False)
def _cached_render(pdf_bytes: bytes, page_index: int = 0, zoom: float = 1.5) -> np.ndarray:
    return render_pdf_page(pdf_bytes, page_index, zoom)


@st.cache_data(show_spinner=False)
def _cached_geometry(pdf_bytes: bytes):
    return parse_pdf_geometry(pdf_bytes)


@st.cache_data(show_spinner=False)
def _cached_drawings_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return len(doc[0].get_drawings())


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_mask_package(img_bytes: bytes, hatch_bytes: bytes):
    """Cache mask building — keyed by image + hatch content."""
    from cleaner.mask_package import build_mask_package
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    hatch = np.frombuffer(hatch_bytes, np.uint8).reshape(img.shape[:2]) if hatch_bytes else None
    return build_mask_package(img, hatch)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_composite(img_bytes: bytes, pkg_key: str, visible_key: str):
    """Cache the slow composite_masks_preview — keyed by image + mask states."""
    # pkg_key and visible_key are str hashes; actual data comes from session state
    # This cache is invalidated when masks or visibility change
    return st.session_state.get("_cached_composite_result")


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_zone_detect(img_bytes: bytes, arch_bytes: bytes):
    from cleaner.zone_detector import auto_detect_zones
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    arch = np.frombuffer(arch_bytes, np.uint8).reshape(img.shape[:2])
    return auto_detect_zones(img, arch)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_tiers(land_bytes: bytes, arch_bytes: bytes, shape: tuple):
    from cleaner.zone_detector import assign_tiers
    land = np.frombuffer(land_bytes, np.uint8).reshape(shape)
    arch = np.frombuffer(arch_bytes, np.uint8).reshape(shape)
    return assign_tiers(land, arch)


def _img_to_bytes(img: np.ndarray) -> bytes:
    """Fast ndarray → bytes key for cache."""
    return img.tobytes()


def _composite_cached(base: np.ndarray, pkg, visible: dict) -> np.ndarray:
    """Compute composite only when inputs change; store result in session state."""
    import hashlib
    vis_key = str(sorted(visible.items()))
    # hash of architecture_locked is a cheap proxy for "masks changed"
    mask_hash = hashlib.md5(pkg.architecture_locked.tobytes()).hexdigest()[:8]
    img_hash  = hashlib.md5(base.tobytes()).hexdigest()[:8]
    cache_key = f"{img_hash}_{mask_hash}_{vis_key}"

    if st.session_state.get("_composite_key") == cache_key:
        return st.session_state["_composite_result"]

    from cleaner.mask_package import composite_masks_preview
    result = composite_masks_preview(base, pkg, visible=visible)
    st.session_state["_composite_key"] = cache_key
    st.session_state["_composite_result"] = result
    return result

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
if "layer_vis" not in st.session_state:
    st.session_state.layer_vis = {}      # OCG: {xref: bool}
if "mask_vis" not in st.session_state:
    st.session_state.mask_vis = {        # canonical mask overlays
        "architecture_locked": False,
        "landscape_editable": True,
        "hatch_removed": True,
        "intervention_red": True,
        "analysis_blue": False,
    }
if "zone_mask" not in st.session_state:
    st.session_state.zone_mask = None   # accumulated manual + auto zone mask
if "proposal" not in st.session_state:
    st.session_state.proposal = None    # ProposalResult | None

left, center, right = st.columns([1.15, 3.6, 1.25])

with left:
    st.markdown("<div class='sec-label'>↑ UPLOAD</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("DROP PDF HERE or click to upload", type=["pdf"], label_visibility="collapsed")

    # ── OCG PDF layers (functional) ──────────────────────────────────────────
    st.markdown("<div class='sec-label'>LAYERS / OVERLAYS</div>", unsafe_allow_html=True)

    if uploaded and st.session_state.state.get("ocg_layers"):
        ocg = st.session_state.state["ocg_layers"]
        if ocg.has_layers:
            st.markdown("<div class='small-label'>PDF LAYERS DETECTED</div>", unsafe_allow_html=True)
            for layer in ocg.layers:
                key = f"ocg_{layer.xref}"
                default = st.session_state.layer_vis.get(layer.xref, layer.on)
                val = st.toggle(layer.name, value=default, key=key, disabled=layer.locked)
                if val != st.session_state.layer_vis.get(layer.xref):
                    st.session_state.layer_vis[layer.xref] = val
                    st.rerun()
        else:
            st.markdown("<div class='small-label'>NO OCG LAYERS — USING HATCH DETECTION</div>", unsafe_allow_html=True)
    else:
        # Placeholder decoration when no PDF loaded
        st.markdown("""
<div class='layer-row'><span><span class='layer-swatch' style='background:#ddd'></span>ORIGINAL PDF</span></div>
<div class='layer-row active'><span><span class='layer-swatch swatch-hatch'></span>HATCH DETECTION</span></div>
<div class='layer-row'><span><span class='layer-swatch swatch-arch'></span>ARCHITECTURE</span></div>
<div class='layer-row blue'><span><span class='layer-swatch swatch-land'></span>LANDSCAPE ZONES</span></div>
""", unsafe_allow_html=True)

    # ── Canonical mask toggles ───────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.6rem'>MASK OVERLAYS</div>", unsafe_allow_html=True)
    mask_labels = {
        "architecture_locked": "Architecture (locked)",
        "landscape_editable":  "Landscape editable",
        "hatch_removed":       "Hatch removed",
        "intervention_red":    "AI intervention (red)",
        "analysis_blue":       "Analysis (blue)",
    }
    for key, label in mask_labels.items():
        st.session_state.mask_vis[key] = st.toggle(
            label, value=st.session_state.mask_vis[key], key=f"mvis_{key}"
        )

    # ── Zone input mode ──────────────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.6rem'>ZONE INPUT MODE</div>", unsafe_allow_html=True)
    zone_mode = st.radio(
        "Zone mode",
        ["Auto-detect", "Lasso / Polygon", "Flood fill"],
        horizontal=False, label_visibility="collapsed",
    )
    if st.button("CLEAR ZONES"):
        st.session_state.zone_mask = None
        st.rerun()

    # ── Processing mode ──────────────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.6rem'>PROCESSING</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Auto", "Vector", "Raster"], horizontal=True, label_visibility="collapsed")

with right:
    st.markdown("<div class='sec-label'>HATCH DETECTION</div>", unsafe_allow_html=True)

    sensitivity = st.slider("SENSITIVITY", 0.0, 1.0, 0.68, 0.01, label_visibility="collapsed")
    st.markdown(f"<div class='slider-label'><span>SENSITIVITY</span><span class='slider-val'>{int(sensitivity*100)}%</span></div>", unsafe_allow_html=True)

    min_line = st.slider("MIN LINE LENGTH", 0, 64, 16, 1, label_visibility="collapsed")
    st.markdown(f"<div class='slider-label'><span>MIN LINE LENGTH</span><span class='slider-val'>{min_line} px</span></div>", unsafe_allow_html=True)

    density = st.slider("DENSITY THRESHOLD", 0.0, 1.0, 0.42, 0.01, label_visibility="collapsed")
    st.markdown(f"<div class='slider-label'><span>DENSITY THRESHOLD</span><span class='slider-val'>{int(density*100)}%</span></div>", unsafe_allow_html=True)

    angle_c = st.slider("ANGLE CONSISTENCY", 0.0, 1.0, 0.63, 0.01, label_visibility="collapsed")
    st.markdown(f"<div class='slider-label'><span>ANGLE CONSISTENCY</span><span class='slider-val'>{int(angle_c*100)}%</span></div>", unsafe_allow_html=True)

    show_debug = st.toggle("DETECTION PREVIEW", value=True)

    st.markdown("""
<div class='sec-label' style='margin-top:.75rem'>LEGEND</div>
<div class='legend-item'><span class='layer-swatch swatch-hatch'></span> REMOVABLE HATCH</div>
<div class='legend-item'><span class='layer-swatch swatch-arch'></span> PRESERVED ARCHITECTURE</div>
<div class='legend-item'><span class='layer-swatch swatch-land'></span> LANDSCAPE ZONES</div>
<div class='legend-item'><span class='layer-swatch swatch-seg'></span> SEGMENTATION ZONES</div>
""", unsafe_allow_html=True)

    # ── API KEY ──────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.9rem'>ANTHROPIC API KEY</div>", unsafe_allow_html=True)
    _raw_key = st.text_input(
        "API key (plain or hex-encoded)", value="",
        type="password", label_visibility="collapsed",
        placeholder="sk-ant-… or hex-encoded key",
    )

    def _resolve_key(raw: str) -> str | None:
        raw = raw.strip()
        if not raw:
            import os
            return os.environ.get("ANTHROPIC_API_KEY")
        # Hex-encoded? All hex chars and even length → decode
        if len(raw) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in raw):
            try:
                decoded = bytes.fromhex(raw).decode("utf-8")
                if decoded.startswith("sk-"):
                    return decoded
            except Exception:
                pass
        return raw  # plain key

    _api_key = _resolve_key(_raw_key)
    _key_ok = bool(_api_key and _api_key.startswith("sk-"))
    if _raw_key and not _key_ok:
        st.markdown("<div class='small-label' style='color:var(--pupa-red)'>⚠ key not recognised</div>", unsafe_allow_html=True)
    elif _key_ok:
        st.markdown("<div class='small-label' style='color:#2a7a2a'>✓ key ready</div>", unsafe_allow_html=True)

    # ── AI LANDSCAPE PROPOSAL ────────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.9rem'>AI LANDSCAPE PROPOSAL</div>", unsafe_allow_html=True)
    ai_backend_name = st.selectbox(
        "Backend", list(BACKENDS.keys()), label_visibility="collapsed"
    )
    ai_style = st.selectbox(
        "Style", ["Naturalistic", "Tropical", "Formal", "Minimalist"], label_visibility="collapsed"
    )
    ai_tier = st.selectbox(
        "Tier", ["all", "foreground", "midground", "background"], label_visibility="collapsed"
    )
    st.markdown("<div class='slider-label'><span>STYLE</span><span class='slider-val'>" + ai_style + "</span></div>", unsafe_allow_html=True)
    ai_prompt = st.text_area(
        "Prompt", value="Place naturalistic landscape elements in the outdoor zones.",
        height=70, label_visibility="collapsed",
        placeholder="Describe the landscape intent…"
    )
    run_ai = st.button("▶  GENERATE LANDSCAPE")

    # ── HALFTONE DUOTONE PRINT ───────────────────────────────────────────────
    st.markdown("<div class='sec-label' style='margin-top:.9rem'>HALFTONE DUOTONE PRINT</div>", unsafe_allow_html=True)
    enable_halftone = st.toggle("ENABLE HALFTONE", value=False)
    halftone_mode = st.radio("Render", ["Halftone (AM dots)", "Flat bitmap"], horizontal=False, label_visibility="collapsed")
    ink_dark  = st.color_picker("INK DARK",  "#1B1F5E", label_visibility="visible")
    ink_light = st.color_picker("INK LIGHT", "#E8382A", label_visibility="visible")
    ht_cell   = st.slider("CELL SIZE (px)", 6, 24, 10, 1, label_visibility="collapsed")
    st.markdown("<div class='slider-label'><span>CELL SIZE</span><span class='slider-val'>{} px</span></div>".format(ht_cell), unsafe_allow_html=True)

if uploaded:
    pdf_bytes = uploaded.read()
    # ── Read OCG layers once on upload ──────────────────────────────────────
    if not st.session_state.state.get("ocg_layers"):
        ocg = read_pdf_layers(pdf_bytes)
        if ocg.has_layers and not st.session_state.layer_vis:
            st.session_state.layer_vis = ocg.visibility
        st.session_state.state["ocg_layers"] = ocg

    ocg = st.session_state.state.get("ocg_layers")

    # Re-render with current OCG visibility if PDF has layers
    if ocg and ocg.has_layers and st.session_state.layer_vis:
        before = render_with_visibility(pdf_bytes, st.session_state.layer_vis, 0, 1.5)
    else:
        before = _cached_render(pdf_bytes, 0, 1.5)

    gmeta = _cached_geometry(pdf_bytes)
    first_drawings = _cached_drawings_count(pdf_bytes)
    do_vector = (mode == "Vector") or (mode == "Auto" and first_drawings > 20)

    # ── Handle AI proposal (run outside column context) ─────────────────────
    if run_ai and st.session_state.state.get("pkg"):
        pkg = st.session_state.state["pkg"]
        from cleaner.ai_proposal import ClaudeVisionBackend, ProceduralBackend
        if ai_backend_name == "Claude Vision":
            backend = ClaudeVisionBackend(api_key=_api_key)
        else:
            backend = BACKENDS[ai_backend_name]
        with st.spinner(f"Generating landscape via {backend.name}…"):
            result = backend.propose(before, pkg, ai_prompt, ai_style, ai_tier)
        st.session_state.proposal = result
        if result.error:
            st.warning(f"AI backend: {result.error}")

    with center:
        st.markdown("<div class='sec-label'>PLAN WORKSPACE</div>", unsafe_allow_html=True)

        # Composite active mask overlays onto plan for preview
        display_img = before.copy()
        if st.session_state.state.get("pkg"):
            pkg = st.session_state.state["pkg"]
            display_img = _composite_cached(display_img, pkg, st.session_state.mask_vis)
        if st.session_state.proposal:
            prop = st.session_state.proposal
            if prop.overlay_png is not None and prop.overlay_png.shape[2] == 4:
                alpha = prop.overlay_png[:, :, 3:4] / 255.0
                rgb_overlay = prop.overlay_png[:, :, :3]
                display_img = (display_img * (1 - alpha * 0.6) + rgb_overlay * (alpha * 0.6)).astype(np.uint8)

        if enable_halftone:
            with st.spinner("Rendering halftone duotone…"):
                if halftone_mode.startswith("Flat"):
                    ht_preview = duotone_flat(before, color_dark=ink_dark, color_light=ink_light)
                else:
                    ht_preview = halftone_duotone(before, cell=ht_cell, color_dark=ink_dark, color_light=ink_light)
            st.image(cv2.cvtColor(ht_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        elif zone_mode == "Lasso / Polygon":
            # Show drawable canvas over plan image
            try:
                from streamlit_drawable_canvas import st_canvas
                from PIL import Image as PILImage
                pil_bg = PILImage.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                canvas_result = st_canvas(
                    fill_color="rgba(53, 66, 250, 0.20)",
                    stroke_width=2,
                    stroke_color="#3542FA",
                    background_image=pil_bg,
                    update_streamlit=True,
                    height=display_img.shape[0],
                    width=display_img.shape[1],
                    drawing_mode="polygon",
                    key="zone_canvas",
                )
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    drawn = canvas_json_to_mask(canvas_result.json_data, before.shape[:2])
                    if drawn.any():
                        prev = st.session_state.zone_mask
                        if prev is not None and prev.shape == drawn.shape:
                            st.session_state.zone_mask = merge_zone_masks(prev, drawn)
                        else:
                            st.session_state.zone_mask = drawn
            except ImportError:
                st.warning("streamlit-drawable-canvas not installed")
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("▶  RUN PREPROCESSING"):
            prog = st.progress(0, text="Initialising…")
            if do_vector:
                prog.progress(15, text="Detecting hatch vectors…")
                cleaned_pdf, debug_data, decisions_by_page = clean_vector_pdf_bytes(pdf_bytes, sensitivity)
                prog.progress(50, text="Rasterising cleaned PDF…")
                cleaned = render_pdf_page(cleaned_pdf, 0, 1.5)
                decision_page = decisions_by_page[0] if decisions_by_page else []
                red = np.zeros(before.shape[:2], dtype=np.uint8)
                blue = np.zeros(before.shape[:2], dtype=np.uint8)
                red_boxes = []
                for d in decision_page:
                    x0, y0, x1, y1 = [int(v * 1.5) for v in d["bbox"]]
                    if d["remove"]:
                        red[y0:y1, x0:x1] = 255
                        red_boxes.append((x0, y0, x1, y1))
                    else:
                        blue[y0:y1, x0:x1] = 255
            else:
                prog.progress(15, text="Running raster cleaner…")
                cleaned, overlay_rb = clean_raster_image(before, sensitivity)
                prog.progress(50, text="Encoding cleaned image…")
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

            prog.progress(65, text="Building canonical mask package…")
            pkg = build_mask_package(before, hatch_mask=red)

            # ── zone detection ──────────────────────────────────────────────
            if zone_mode == "Auto-detect" or st.session_state.zone_mask is None:
                auto_zone, _ = auto_detect_zones(before, pkg.architecture_locked)
                manual_zone = st.session_state.zone_mask
                if manual_zone is not None and manual_zone.shape == auto_zone.shape:
                    zone_final = merge_zone_masks(auto_zone, manual_zone)
                else:
                    zone_final = auto_zone
            else:
                zone_final = st.session_state.zone_mask

            # Merge detected zone into pkg
            import numpy as _np  # already imported as np
            merged_land = cv2.bitwise_or(pkg.landscape_editable, zone_final)
            merged_land = cv2.bitwise_and(merged_land, cv2.bitwise_not(pkg.architecture_locked))
            pkg.landscape_editable[:] = merged_land

            # Assign tiers
            fg, mg, bg = assign_tiers(pkg.landscape_editable, pkg.architecture_locked)
            pkg.tier_foreground = fg
            pkg.tier_midground  = mg
            pkg.tier_background = bg

            overlay = _composite_cached(before, pkg, st.session_state.mask_vis)
            transparent = mask_to_transparent_png(cleaned, cv2.bitwise_not(red))

            prog.progress(80, text="Exporting outputs…")
            svg_path = export_debug_svg(before.shape[1], before.shape[0], red_boxes, Path("pw-plan-cleaner-01/outputs") / f"{Path(uploaded.name).stem}_debug.svg")

            outputs = save_outputs(
                Path("pw-plan-cleaner-01/outputs"),
                Path(uploaded.name).stem,
                cleaned_pdf,
                cleaned,
                {"vector_geometry": gmeta, "cleaner": debug_data},
                transparent_png=transparent,
                svg_path=svg_path,
                landscape_mask=pkg.landscape_editable,
            )

            prog.progress(92, text="Rendering halftone…")
            if halftone_mode.startswith("Flat"):
                ht_result = duotone_flat(before, hatch_mask=red, color_dark=ink_dark, color_light=ink_light)
            else:
                ht_result = halftone_duotone(before, hatch_mask=red, cell=ht_cell, color_dark=ink_dark, color_light=ink_light)

            prog.progress(100, text="Done.")

            ok_pdf  = Path(outputs["pdf"]).read_bytes()
            ok_tpng = Path(outputs["transparent_png"]).read_bytes()
            ok_svg  = Path(outputs["svg"]).read_text()
            ok_lmask = Path(outputs["landscape_mask"]).read_bytes()
            ok_debug = Path(outputs["debug"]).read_text()

            # Preserve OCG info across state reset
            ocg_saved = st.session_state.state.get("ocg_layers")
            st.session_state.state = {
                "ocg_layers": ocg_saved,
                "before": before,
                "cleaned": cleaned,
                "overlay": overlay,
                "halftone": ht_result,
                "pkg": pkg,
                "outputs": outputs,
                "out_pdf": ok_pdf,
                "out_tpng": ok_tpng,
                "out_svg": ok_svg,
                "out_lmask": ok_lmask,
                "out_debug": ok_debug,
                "debug": debug_data,
                "mode": "vector" if do_vector else "raster",
                "affected_count": int(np.count_nonzero(red > 0)),
            }
            st.session_state.proposal = None

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
        proposal = st.session_state.proposal
        if proposal is not None and proposal.overlay_png is not None:
            st.markdown("<div class='sec-label'>AI LANDSCAPE PROPOSAL</div>", unsafe_allow_html=True)
            # Composite proposal over cleaned plan
            base = data["cleaned"].copy()
            if proposal.overlay_png.shape[2] == 4:
                alpha = proposal.overlay_png[:, :, 3:4] / 255.0
                rgb_ov = proposal.overlay_png[:, :, :3]
                comp = (base * (1 - alpha * 0.7) + rgb_ov * (alpha * 0.7)).astype(np.uint8)
            else:
                comp = proposal.overlay_png[:, :, :3]
            st.image(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(
                f"<div class='small-label'>Backend: {proposal.backend_used} | "
                f"Elements: {len(proposal.primitives)} | "
                f"Confidence: {int(proposal.confidence * 100)}%</div>",
                unsafe_allow_html=True,
            )
            ok_comp, comp_buf = cv2.imencode(".png", comp)
            if ok_comp:
                st.download_button(
                    "EXPORT PROPOSAL PNG  ↗ PNG",
                    data=comp_buf.tobytes(),
                    file_name="landscape_proposal.png",
                    mime="image/png",
                )
        elif "halftone" in data:
            st.markdown("<div class='sec-label'>HALFTONE DUOTONE PRINT</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(data["halftone"], cv2.COLOR_BGR2RGB), use_container_width=True)
            ok_ht, ht_buf = cv2.imencode(".png", data["halftone"])
            if ok_ht:
                st.download_button("EXPORT HALFTONE PNG  ↗ PNG",
                                   data=ht_buf.tobytes(), file_name="halftone_duotone.png")
        elif show_debug:
            st.markdown("<div class='sec-label'>ZONE OVERLAY</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(data["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)

    with b4:
        st.markdown("<div class='sec-label'>OUTPUT / EFFECTED AREAS</div>", unsafe_allow_html=True)
        n_prim = len(st.session_state.proposal.primitives) if st.session_state.proposal else 0
        pkg = data.get("pkg")
        if pkg is not None:
            zone_px = int(np.count_nonzero(pkg.landscape_editable > 0))
            st.markdown(f"<div class='red-marker'><b>LANDSCAPE ZONE</b><br>{zone_px:,} px editable</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='red-marker'><b>HATCH REMOVED</b><br>{data['affected_count']:,} px</div>", unsafe_allow_html=True)
        if n_prim:
            st.markdown(f"<div class='red-marker'><b>AI ELEMENTS PLACED</b><br>{n_prim} primitives</div>", unsafe_allow_html=True)

        o = data["outputs"]
        st.markdown("<div class='sec-label' style='margin-top:.75rem'>EXPORT OPTIONS</div>", unsafe_allow_html=True)
        if "out_pdf" in data:
            st.download_button("EXPORT CLEANED PDF  ↗ PDF", data=data["out_pdf"],
                               file_name=Path(o["pdf"]).name)
        if "out_tpng" in data:
            st.download_button("EXPORT PNG (TRANSPARENT)  ↗ PNG", data=data["out_tpng"],
                               file_name=Path(o["transparent_png"]).name)
        if "out_svg" in data:
            st.download_button("EXPORT SVG (VECTOR)  ↗ SVG", data=data["out_svg"],
                               file_name=Path(o["svg"]).name)
        if "out_lmask" in data:
            st.download_button("EXPORT LANDSCAPE MASK  ↗ MASK", data=data["out_lmask"],
                               file_name=Path(o["landscape_mask"]).name)
        if "out_debug" in data:
            st.download_button("EXPORT METADATA  ↗ JSON", data=data["out_debug"],
                               file_name=Path(o["debug"]).name)

        # Full bundle export (Grasshopper JSON + masks)
        if st.button("▶  EXPORT FULL BUNDLE"):
            prop = st.session_state.proposal
            bundle = save_landscape_bundle(
                base_dir=Path("pw-plan-cleaner-01/outputs"),
                stem=Path(uploaded.name).stem,
                cleaned_pdf=data.get("out_pdf"),
                original_bgr=data["before"],
                cleaned_bgr=data["cleaned"],
                proposal_bgr=None,
                masks=pkg.as_dict() if pkg else {},
                svg_elements=prop.svg_paths if prop else [],
                primitives=prop.primitives if prop else [],
                debug_data={"cleaner": data.get("debug", [])},
            )
            st.success(f"Bundle saved: {len(bundle)} files")
            if "grasshopper_json" in bundle:
                gh_txt = bundle["grasshopper_json"].read_text()
                st.download_button("EXPORT GRASSHOPPER JSON  ↗ JSON",
                                   data=gh_txt, file_name="grasshopper_exchange.json")

        if st.button("↺  UNDO / RESET"):
            st.session_state.state = {"ocg_layers": data.get("ocg_layers")}
            st.session_state.proposal = None
            st.session_state.zone_mask = None
            st.rerun()

# ── Footer ──
st.markdown("""
<div class='pw-footer'>
  <span>PW-PLAN-CLEANER-01</span>
  <span>PRESERVATION FIRST. AI READY. LANDSCAPE POSSIBLE.</span>
  <span>PUPA WORKS © 2025</span>
</div>
""", unsafe_allow_html=True)
