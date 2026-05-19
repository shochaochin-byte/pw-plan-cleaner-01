# PW-PLAN-CLEANER-01

Preservation-first architectural PDF cleaning and preprocessing system for **Pupa Works**.

## Primary Objective
Remove floor finish hatching and noisy floor textures while preserving walls, geometry, dimensions, text, title block, openings, circulation, and architectural intent.

## Core Stack
- Python
- PyMuPDF
- pdfplumber
- OpenCV
- Streamlit

## Project Structure
```
pw-plan-cleaner-01/
├── app.py
├── requirements.txt
├── README.md
├── cleaner/
│   ├── vector_cleaner.py
│   ├── raster_cleaner.py
│   ├── segmentation.py
│   ├── geometry_parser.py
│   ├── svg_export.py
│   ├── masking.py
│   ├── export.py
│   └── preview.py
├── outputs/
├── uploads/
├── debug/
└── models/
```

## Features
- Vector PDF analysis (`page.get_drawings`) and geometry metadata extraction
- Dimension/text-preservation-first filtering
- Hatch removal engine (thin, dense, repeated vectors)
- Raster fallback for flattened PDFs
- AI-prep segmentation masks:
  - architecture mask
  - landscape mask
  - hatch mask
  - circulation mask
- Visual overlays:
  - red = removable hatch
  - blue = preserved architecture
  - green = landscape-capable zones
- Export:
  - cleaned PDF
  - cleaned PNG
  - transparent PNG
  - SVG debug output
  - landscape mask PNG
  - debug JSON metadata

## Run
```bash
cd pw-plan-cleaner-01
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Safety Rules
- Never modifies original file.
- Always writes new outputs.
- Default mode is conservative.
- Avoids removal near text/title-zone and likely structural linework.
