import json
import urllib.request


def send_to_plan_cleaner(curves, target_url="http://localhost:8501/?api_action=process_geometry"):
    serialized_curves = []
    for crv in curves:
        if crv and crv.IsValid:
            start_pt = crv.PointAtStart
            end_pt = crv.PointAtEnd
            serialized_curves.append(
                {
                    "layer": "Grasshopper",
                    "start": {"x": start_pt.X, "y": start_pt.Y, "z": start_pt.Z},
                    "end": {"x": end_pt.X, "y": end_pt.Y, "z": end_pt.Z},
                }
            )

    payload = {"curves": serialized_curves}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(target_url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")
