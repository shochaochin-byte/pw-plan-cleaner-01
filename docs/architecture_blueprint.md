# Unified 3-Track Architecture Blueprint

## Track 1 — Vector Cleaning (PyMuPDF + Shapely)
- Module: `cleaner/track1_vector_engine.py`
- Entry point: `clean_cad_pdf(pdf_path, output_png_path, ...)`
- Flow:
  1. Read vector paths from PDF (`page.get_drawings()`).
  2. Remove explicit hatch layers when metadata exists (`NI-HATCH`/`HATCH`).
  3. For flattened drawings, apply local line-density filtering with `STRtree` neighbor queries.
  4. Emit cleaned raster preview for UI/segmentation.

## Track 2 — Local Semantic Segmentation (SAM2)
- Module: `cleaner/track2_local_zoning.py`
- Class: `LocalZoningEngine`
- Method: `segment_zone(image_path, click_x, click_y)`
- Flow:
  1. Accept single-click point prompt.
  2. Run local `ultralytics.SAM` zero-shot prediction on CPU/GPU.
  3. Return binary mask, contour geometry, and area metrics.

## Track 3 — CAD Bridge (Streamlit + Grasshopper)
- Streamlit receiver:
  - App hook: `init_api_endpoints()` in `app.py`
  - Utility: `cleaner/track3_cad_bridge.py`
  - Endpoint style: `?api_action=process_geometry`
  - Parses incoming CAD JSON and normalizes line segments.
- Grasshopper sender:
  - Script: `cad_bridge/grasshopper_client.py`
  - Serializes Rhino curves (`PointAtStart`, `PointAtEnd`) to JSON.
  - Posts payload to local Streamlit endpoint.

## End-to-end data path
1. Rhino/Grasshopper streams curve payload JSON to Streamlit.
2. Streamlit parses payload and can feed vectors into cleaning workflow.
3. Track 1 sanitizes vectors and outputs rasterized canvas.
4. Track 2 applies local click-driven zoning segmentation.
5. Results (masks/zones) are returned to UI and export pipeline.
