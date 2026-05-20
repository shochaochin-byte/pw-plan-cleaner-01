from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import socket
from urllib.parse import urlparse


@dataclass(frozen=True)
class CapabilityCheck:
    name: str
    status: str  # pass | warn | fail
    ok: bool
    detail: str


@dataclass(frozen=True)
class CapabilityReport:
    model_files: dict[str, bool]
    runtime: dict[str, bool]
    services: dict[str, bool]
    features: dict[str, bool]
    diagnostics: list[CapabilityCheck] = field(default_factory=list)


def _check_file(path: Path, label: str) -> CapabilityCheck:
    exists = path.exists()
    return CapabilityCheck(
        name=label,
        status="pass" if exists else "fail",
        ok=exists,
        detail=f"{path} {'found' if exists else 'missing'}",
    )


def _check_service(url: str, label: str, timeout_s: float = 0.4) -> CapabilityCheck:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok = False
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            ok = True
    except OSError:
        ok = False
    return CapabilityCheck(
        name=label,
        status="pass" if ok else "warn",
        ok=ok,
        detail=f"{url} {'reachable' if ok else 'unreachable'}",
    )


def validate_startup(project_root: Path | None = None) -> CapabilityReport:
    root = (project_root or Path.cwd()).resolve()

    sam_candidates = [
        root / "models" / "sam2.1_b.pt",
        root / "checkpoints" / "sam2.1_b.pt",
        root / "sam2.1_b.pt",
        Path(os.environ.get("SAM_MODEL_PATH", "")).expanduser() if os.environ.get("SAM_MODEL_PATH") else None,
    ]
    sam_candidates = [p for p in sam_candidates if p is not None]

    model_checks = [_check_file(p, f"SAM model: {p}") for p in sam_candidates]
    sam_model_ok = any(c.ok for c in model_checks)

    cpu_ok = True
    cuda_ok = False
    try:
        import torch  # type: ignore

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = bool(getattr(__import__("cv2"), "cuda", None))

    runtime_checks = [
        CapabilityCheck("CPU runtime", "pass" if cpu_ok else "fail", cpu_ok, "CPU available" if cpu_ok else "CPU unavailable"),
        CapabilityCheck("CUDA runtime", "pass" if cuda_ok else "warn", cuda_ok, "CUDA available" if cuda_ok else "CUDA unavailable"),
    ]

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    diffusers_url = os.environ.get("DIFFUSERS_BASE_URL", "http://127.0.0.1:7860")
    service_checks = [
        _check_service(ollama_url, "Ollama"),
        _check_service(diffusers_url, "Diffusers"),
    ]

    agentic_ok = any(c.ok for c in service_checks)
    ai_render_ok = all(c.ok for c in service_checks)
    sam_zoning_ok = sam_model_ok and cpu_ok

    features = {
        "sam_assisted_zoning": sam_zoning_ok,
        "agentic_layer_disambiguation": agentic_ok,
        "ai_render_modes": ai_render_ok,
    }

    return CapabilityReport(
        model_files={"sam2.1_b.pt": sam_model_ok},
        runtime={"cpu": cpu_ok, "cuda": cuda_ok},
        services={"ollama": service_checks[0].ok, "diffusers": service_checks[1].ok},
        features=features,
        diagnostics=[*model_checks, *runtime_checks, *service_checks],
    )
