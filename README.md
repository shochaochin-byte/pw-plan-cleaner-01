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

## Optional CAD Bridge (URL payload)
The app now supports an optional CAD bridge module (`cleaner/mcp_bridge.py`) that can ingest structured geometry from URL query params and merge it into internal guidance masks.

### Feature flag
- Enable with query param: `cad_bridge=1`
- Provide payload in `cad_payload` as either:
  - raw JSON string, or
  - URL-safe base64 encoded JSON

If the bridge flag is off, payload is missing, or payload is invalid, preprocessing continues normally without failure.

### Expected payload schema
```json
{
  "units": "mm",
  "layers": [
    { "id": "L1", "name": "LANDSCAPE_ZONE" },
    { "id": "A1", "name": "ARCH_LOCK" }
  ],
  "polylines": [
    {
      "layer": "L1",
      "closed": true,
      "points": [[120.0, 300.0], [640.0, 320.0], [620.0, 760.0], [130.0, 740.0]]
    }
  ],
  "curves": [
    {
      "layer": "A1",
      "control_points": [[300.0, 200.0], [350.0, 260.0], [400.0, 240.0], [450.0, 300.0]]
    }
  ]
}
```

### Example URL
```text
http://localhost:8501/?cad_bridge=1&cad_payload=%7B%22units%22%3A%22mm%22%2C%22layers%22%3A%5B%7B%22id%22%3A%22L1%22%2C%22name%22%3A%22LANDSCAPE_ZONE%22%7D%5D%2C%22polylines%22%3A%5B%7B%22layer%22%3A%22L1%22%2C%22closed%22%3Atrue%2C%22points%22%3A%5B%5B100%2C100%5D%2C%5B600%2C100%5D%2C%5B600%2C500%5D%2C%5B100%2C500%5D%5D%7D%5D%2C%22curves%22%3A%5B%5D%7D
```

### Export artifact
When a valid bridge payload is applied and full bundle export runs, `outputs/<stem>_export/cad_bridge.json` is emitted for Rhino/Grasshopper downstream consumption.
