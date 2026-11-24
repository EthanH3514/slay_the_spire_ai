"""Optional native metrics audit filter. Not imported by the live intake.

One ctypes call per bounded block; only selected original byte spans become
Python records. Native parse limitations use the exact reference decoder.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path

from .metrics_cards import cohort
from .metrics_stream import file_sha
from .native_tools import _no_windows_error_dialogs

ROOT = Path(__file__).resolve().parents[2]
REASONS = ("wrong_character", "not_exact_a20", "unsupported_or_unknown_flag:is_daily",
    "unsupported_or_unknown_flag:is_endless", "unsupported_or_unknown_flag:is_trial",
    "unsupported_or_unknown_flag:is_beta", "unsupported_or_unknown_flag:chose_seed",
    "special_seed", "modded_or_unknown_flag", "unknown_outcome")


class _Selection(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("index", "offset", "length", "category")]


class _Report(ctypes.Structure):
    _fields_ = [("records", ctypes.c_uint64), ("selected_count", ctypes.c_uint64),
                ("excluded", ctypes.c_uint64 * len(REASONS)), ("selected", ctypes.POINTER(_Selection))]


def build_directory():
    return Path(os.environ.get("SPIRE_METRICS_FILTER_DIR", ROOT / "runtime/metrics-cpp-filter-v4")).resolve()


@lru_cache(maxsize=2)
def _load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    library = (directory / manifest["library"]).resolve()
    if library.parent != directory or manifest["abi"] != 1:
        raise ValueError("invalid metrics filter manifest/ABI")
    if (file_sha(library) != manifest["library_sha256"] or
            file_sha(ROOT / "native/metrics_filter.cpp") != manifest["source_sha256"] or
            file_sha(ROOT / "native/metrics_filter_deps.json") != manifest["dependency_lock_sha256"] or
            file_sha(ROOT / "src/spire_ai/metrics_cards.py") != manifest["cohort_source_sha256"]):
        raise ValueError("stale or modified metrics filter build; rebuild into a new directory")
    with _no_windows_error_dialogs():
        dll = ctypes.CDLL(str(library))
    dll.spire_metrics_abi_version.argtypes = []
    dll.spire_metrics_abi_version.restype = ctypes.c_uint32
    if dll.spire_metrics_abi_version() != 1:
        raise ValueError("metrics filter ABI mismatch")
    dll.spire_metrics_filter.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(_Report))]
    dll.spire_metrics_filter.restype = ctypes.c_int
    dll.spire_metrics_free.argtypes = [ctypes.POINTER(_Report)]
    dll.spire_metrics_free.restype = None
    return dll


def build_identity():
    directory = build_directory()
    manifest = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    return {"manifest_sha256": file_sha(directory / "build.json"),
            "library_sha256": manifest["library_sha256"], "simdjson_version": manifest["simdjson_version"]}


def parse_cpp_block(payload):
    # Imported lazily to keep native build/dependencies optional for the app.
    from .metrics_decode_probe import MAX_BLOCK_BYTES, ParsedBlock, _stdlib, reject_constant
    if not isinstance(payload, bytes) or len(payload) > MAX_BLOCK_BYTES:
        raise ValueError("expected bounded immutable bytes")
    dll = _load(str(build_directory()))
    result = ctypes.POINTER(_Report)()
    status = dll.spire_metrics_filter(payload, len(payload), ctypes.byref(result))
    if status == 1:
        fallback = _stdlib(payload)
        fallback.compatibility_fallback = True
        return fallback
    if status == 3:
        raise MemoryError("native metrics filter exhausted memory")
    if status != 0 or not result:
        raise RuntimeError(f"native metrics filter failed with status {status}")
    try:
        report = result.contents
        if report.records > len(payload) // 2 or report.selected_count > report.records:
            raise RuntimeError("invalid native record counts")
        excluded = {reason: int(report.excluded[i]) for i, reason in enumerate(REASONS) if report.excluded[i]}
        if sum(excluded.values()) + report.selected_count != report.records:
            raise RuntimeError("inconsistent native counts")
        if report.selected_count and not report.selected:
            raise RuntimeError("missing native selection buffer")
        retained, previous_index, previous_end = [], -1, 0
        for i in range(report.selected_count):
            item = report.selected[i]
            if (item.index <= previous_index or item.index >= report.records or item.offset < previous_end or
                    item.length == 0 or item.offset + item.length > len(payload) or item.category not in (1, 2)):
                raise RuntimeError("invalid native record span/order")
            raw = payload[item.offset:item.offset + item.length]
            wrapped = json.loads(raw, parse_constant=reject_constant)
            category = "win_candidate" if item.category == 1 else "loss_retained_only"
            if not isinstance(wrapped, dict) or not isinstance(wrapped.get("event"), dict):
                raise RuntimeError("native wrapper mismatch")
            if cohort(wrapped["event"])[0] != category:
                raise RuntimeError("native cohort mismatch")
            retained.append((int(item.index), wrapped))
            previous_index, previous_end = item.index, item.offset + item.length
        return ParsedBlock(int(report.records), excluded, retained)
    finally:
        dll.spire_metrics_free(result)
