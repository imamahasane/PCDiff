from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def read_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in [".yml", ".yaml"]:
        return yaml.safe_load(path.read_text())
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config type: {path.suffix}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def git_commit_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        return "unknown"


def run_provenance(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:

    import datetime

    info: Dict[str, Any] = {
        "git_commit": git_commit_hash(),
        "python_version": sys.version,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    if extra:
        info.update(extra)
    return info
