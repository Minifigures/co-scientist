"""Web fetch: pull a URL, extract clean text, cache on disk by URL hash.

- HTML → trafilatura.extract
- PDF (Content-Type or .pdf suffix) → pypdf text extraction
- Cap: 5 MB by default; 5 redirects max
- Cache: data/artifacts/<session>/papers/<sha1(url)>.json (survives resume)
"""

from __future__ import annotations

import asyncio
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

from ..config import Config
from ..ids import url_hash
from .base import ToolCtx, ToolResult


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch a URL and return its main text content (HTML → cleaned text; PDF → extracted text). "
        "Returns {url, title?, text, content_type, status, bytes}. "
        "Use after web_search / arxiv_search / pubmed_search to read the actual content."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
            "max_chars": {
                "type": "integer",
                "minimum": 200,
                "maximum": 200_000,
                "description": "Truncate text to this many characters (default 30000).",
            },
        },
        "required": ["url"],
    }

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        url = args.get("url", "").strip()
        max_chars = int(args.get("max_chars") or 30_000)
        if not url.startswith(("http://", "https://")):
            return ToolResult(is_error=True, error_message="URL must start with http(s)")

        cached = await self._read_cache(ctx, url)
        if cached is not None:
            cached = self._truncate(cached, max_chars)
            return ToolResult(
                content=cached,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(json.dumps(cached)),
            )

        try:
            async with httpx.AsyncClient(
                timeout=self._cfg.web_fetch.timeout_seconds,
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": self._cfg.web_fetch.user_agent},
            ) as client:
                r = await client.get(url)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"fetch failed: {e}")

        if r.status_code >= 400:
            return ToolResult(
                is_error=True,
                error_message=f"HTTP {r.status_code}",
                content={"url": url, "status": r.status_code},
            )
        if len(r.content) > self._cfg.web_fetch.max_bytes:
            return ToolResult(
                is_error=True,
                error_message=f"response too large ({len(r.content)} bytes)",
            )

        ct = (r.headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ct or url.lower().endswith(".pdf")
        try:
            if is_pdf:
                text = await asyncio.to_thread(_extract_pdf, r.content)
                title: str | None = None
            else:
                text, title = await asyncio.to_thread(_extract_html, r.text, url)
        except Exception as e:
            return ToolResult(
                is_error=True, error_message=f"extraction failed: {e}"
            )

        payload: dict[str, Any] = {
            "url": str(r.url),
            "title": title,
            "text": text,
            "content_type": ct,
            "status": r.status_code,
            "bytes": len(r.content),
        }
        await self._write_cache(ctx, url, payload)
        payload = self._truncate(payload, max_chars)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(json.dumps(payload)),
        )

    # ----------------------------- cache --------------------------------- #

    def _cache_path(self, ctx: ToolCtx, url: str) -> Path | None:
        if ctx.session_id is None:
            return None
        return (
            self._cfg.session_artifact_dir(ctx.session_id)
            / "papers"
            / f"{url_hash(url)}.json"
        )

    async def _read_cache(self, ctx: ToolCtx, url: str) -> dict[str, Any] | None:
        p = self._cache_path(ctx, url)
        if p is None or not p.exists():
            return None

        def _do() -> dict[str, Any]:
            return json.loads(p.read_text())

        return await asyncio.to_thread(_do)

    async def _write_cache(
        self, ctx: ToolCtx, url: str, payload: dict[str, Any]
    ) -> None:
        p = self._cache_path(ctx, url)
        if p is None:
            return

        def _do() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, default=str, ensure_ascii=False))
            tmp.replace(p)

        await asyncio.to_thread(_do)

    @staticmethod
    def _truncate(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
        text = payload.get("text") or ""
        if len(text) > max_chars:
            payload = {**payload, "text": text[:max_chars], "truncated": True}
        return payload


# --------------------------------------------------------------------------- #
# Extractors (sync; run via to_thread)


def _extract_html(html: str, url: str) -> tuple[str, str | None]:
    import trafilatura

    extracted = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True
    )
    title = None
    md = trafilatura.metadata.extract_metadata(html)
    if md:
        title = md.title
    return extracted or "", title


def _extract_pdf(data: bytes) -> str:
    import pypdf

    reader = pypdf.PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)
