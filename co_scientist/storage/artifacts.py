"""JSON-on-disk helpers.

Artifacts live under `<data_dir>/artifacts/<session_id>/<kind>/<id>.json`.
The DB stores the path *relative* to `data_dir` for portability.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ..config import Config


def _rel(cfg: Config, p: Path) -> str:
    return str(p.relative_to(cfg.data_dir))


def session_root(cfg: Config, session_id: str) -> Path:
    return cfg.data_dir / "artifacts" / session_id


def _write(p: Path, payload: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    tmp.replace(p)  # atomic on POSIX


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


async def write_json(cfg: Config, session_id: str, kind: str, id_: str, payload: Any) -> str:
    """Persist a JSON artifact; return its relative path."""
    p = session_root(cfg, session_id) / kind / f"{id_}.json"
    await asyncio.to_thread(_write, p, payload)
    return _rel(cfg, p)


async def read_json(cfg: Config, rel_path: str) -> Any:
    p = cfg.data_dir / rel_path
    return await asyncio.to_thread(_read, p)


async def write_text(cfg: Config, session_id: str, kind: str, id_: str, suffix: str, body: str) -> str:
    p = session_root(cfg, session_id) / kind / f"{id_}{suffix}"

    def _do() -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(body)
        tmp.replace(p)

    await asyncio.to_thread(_do)
    return _rel(cfg, p)
