#!/usr/bin/env python3
"""Minimal web UI server for visualizing recorded histogram snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a minimal Plotly UI for a histogram snapshot file."
    )
    parser.add_argument(
        "hist_file",
        type=Path,
        help="Path to the histogram JSON file produced by streaming-histogram.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument(
        "--ui-root",
        type=Path,
        default=Path("ui"),
        help="Directory containing static UI assets (default: ./ui).",
    )
    return parser.parse_args()


def _format_bound(raw: Any) -> tuple[str, float | None]:
    """Convert serialized bucket bounds into printable labels and numeric values."""
    if not isinstance(raw, dict):
        return ("?", None)

    if "finite" in raw:
        value = float(raw["finite"])
        return (f"{value:g}", value)
    if "neg-infinity" in raw:
        return ("-inf", None)
    if "pos-infinity" in raw:
        return ("inf", None)
    if "nan" in raw:
        return ("nan", None)
    return ("?", None)


def _normalize_bucket(entry: dict[str, Any]) -> dict[str, Any]:
    start_label, start_value = _format_bound(entry.get("start"))
    end_label, end_value = _format_bound(entry.get("end"))
    count = int(entry.get("count", 0))
    key = f"{start_label}|{end_label}"
    return {
        "key": key,
        "start": start_value,
        "end": end_value,
        "start_label": start_label,
        "end_label": end_label,
        "label": f"[{start_label}, {end_label})",
        "count": count,
    }


def _normalize_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    bins = [_normalize_bucket(bucket) for bucket in entry.get("bins", [])]
    return {
        "index": int(entry.get("index", 0)),
        "label": entry.get("label"),
        "bins": bins,
        "total": sum(bucket["count"] for bucket in bins),
    }


def _load_snapshots(hist_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    recorder = payload.get("recorder", {})
    snapshots = recorder.get("snapshots", [])
    normalized = [_normalize_snapshot(entry) for entry in snapshots]
    normalized.sort(key=lambda snap: snap["index"])
    return normalized


class HistogramRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, hist_path: Path, **kwargs: Any) -> None:
        self._hist_path = hist_path
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - keep default style compact
        sys.stderr.write(f"[server] {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        parsed = urlparse(self.path)
        if parsed.path == "/data":
            self._handle_data_request()
            return

        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _handle_data_request(self) -> None:
        try:
            payload = {"snapshots": _load_snapshots(self._hist_path)}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
        except FileNotFoundError:
            body = json.dumps({"error": "histogram file not found"}).encode("utf-8")
            self.send_response(404)
        except json.JSONDecodeError as exc:
            body = json.dumps({"error": f"failed to parse histogram JSON: {exc}"}).encode("utf-8")
            self.send_response(500)
        except Exception as exc:  # pragma: no cover - defensive
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    args = _parse_args()
    hist_path = args.hist_file.expanduser().resolve()
    if not hist_path.is_file():
        sys.exit(f"Histogram file not found: {hist_path}")

    ui_root = args.ui_root.expanduser().resolve()
    if not ui_root.is_dir():
        sys.exit(f"UI directory not found: {ui_root}")

    handler = partial(
        HistogramRequestHandler,
        directory=str(ui_root),
        hist_path=hist_path,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {hist_path} at http://{args.host}:{args.port} (UI root: {ui_root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
