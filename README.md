# PW-PLAN-CLEANER-01

Local Streamlit tool to clean architectural PDF plan hatching while preserving key layout geometry.

## Features
- PDF upload and before/after preview
- Vector hatch detection using PyMuPDF `page.get_drawings()` heuristics
- Conservative safety rules (avoid touching text/title-block-adjacent geometry)
- Overlay semantics:
  - red = hatch candidates to remove
  - blue = preserved linework
- Sensitivity slider
- Export cleaned PDF, cleaned PNG, and debug JSON
- Raster fallback mode using OpenCV for flattened PDFs
- Undo/reset and never modifies original files

## Run
```bash
cd pw-plan-cleaner-01
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
Default mode is conservative. For safety, manually review output before production use.
