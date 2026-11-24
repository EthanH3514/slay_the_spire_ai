"""Isolated metrics decoder experiments; NOT the live intake implementation.

Native parsers operate on bounded, independently validated JSON-array blocks.
This module does not change the running intake's pinned parser/contract. Raw
events, including seeds and outcomes, stay audit data, never model features.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import io
import json
import multiprocessing

from .metrics_cards import cohort
from .metrics_stream import records

MODES = ("original", "cursor", "orjson", "simd-full", "simd-select", "cpp-select")
FIELDS = frozenset(("character_chosen", "ascension_level", "is_daily", "is_endless",
                    "is_trial", "is_beta", "chose_seed", "special_seed", "is_modded", "victory"))
MAX_BLOCK_BYTES = 8 * 1024 * 1024
_SIMD_PARSER = None


def reject_constant(value):
    raise ValueError("nonfinite JSON: " + value)


def cursor_records(stream, chunk_size=65536, max_record_chars=2_000_000, *, raw=False):
    """Same array contract as records(), compacting only at input refill.

raw=True is only for preparing byte-preserving benchmark blocks, after each
record has ALSO passed the original stdlib JSON decoder and object/size checks.
"""
    decoder = json.JSONDecoder(parse_constant=reject_constant)
    buffer, pos = "", 0

    def refill():
        nonlocal buffer, pos
        more = stream.read(chunk_size)
        buffer, pos = buffer[pos:] + more, 0
        return bool(more)

    def ready():
        nonlocal pos
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer):
                return
            if not refill():
                raise ValueError("truncated JSON array")

    ready()
    if buffer[pos] != "[":
        raise ValueError("expected JSON array")
    pos += 1
    first = True
    while True:
        ready()
        if buffer[pos] == "]":
            if buffer[pos + 1:].strip():
                raise ValueError("extra data after array")
            while tail := stream.read(chunk_size):
                if tail.strip():
                    raise ValueError("extra data after array")
            return
        if not first:
            if buffer[pos] != ",":
                raise ValueError("missing array separator")
            pos += 1
            ready()
        while True:
            start = pos
            try:
                value, end = decoder.raw_decode(buffer, pos)
                break
            except json.JSONDecodeError:
                if len(buffer) - pos > max_record_chars:
                    raise ValueError("record exceeds intake bound") from None
                if not refill():
                    raise ValueError("invalid or truncated record") from None
        if end - start > max_record_chars or not isinstance(value, dict):
            raise ValueError("expected bounded record object")
        pos, first = end, False
        yield buffer[start:end] if raw else value


@dataclass
class ParsedBlock:
    records: int
    excluded: dict
    retained: list
    compatibility_fallback: bool = False


def _filter(rows):
    excluded, retained, count = Counter(), [], 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("event"), dict):
            raise ValueError("expected event wrapper")
        count += 1
        category, reason = cohort(row["event"])
        if category == "exclude":
            excluded[reason] += 1
        else:
            retained.append((index, row))
    return ParsedBlock(count, dict(excluded), retained)


def _stdlib(payload, mode="cursor"):
    with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8", newline="") as stream:
        return _filter((records if mode == "original" else cursor_records)(stream))


class _CompatibilityFallback(Exception):
    pass


def _select(doc, simdjson):
    """Validate the entire native DOM, materialize only selected Python events.

Duplicate cohort/wrapper keys fall back to stdlib's last-key-wins semantics.
The native DOM validates even fields that are not selected (not regex filtering).
"""
    if not isinstance(doc, simdjson.Array):
        raise ValueError("expected JSON array")
    excluded, retained, count = Counter(), [], 0
    for index, wrapped in enumerate(doc):
        if not isinstance(wrapped, simdjson.Object):
            raise ValueError("expected event wrapper")
        if list(wrapped.keys()).count("event") != 1:
            raise _CompatibilityFallback()
        event = wrapped.get("event")
        if not isinstance(event, simdjson.Object):
            raise ValueError("expected event wrapper")
        keys = [k for k in event.keys() if k in FIELDS]
        if len(keys) != len(set(keys)):
            raise _CompatibilityFallback()
        projection = {k: event[k] for k in keys}
        # Truthiness of container-valued special_seed must match Python. All
        # other fields use exact type/identity checks in the existing cohort.
        if "special_seed" in projection and isinstance(projection["special_seed"],
                                                        (simdjson.Array, simdjson.Object)):
            projection["special_seed"] = bool(len(projection["special_seed"]))
        category, reason = cohort(projection)
        count += 1
        if category == "exclude":
            excluded[reason] += 1
        else:
            # Retained data still uses stdlib semantics, including nested
            # duplicate keys. mini is native minification, not regex extraction.
            row = json.loads(wrapped.mini, parse_constant=reject_constant)
            if cohort(row["event"])[0] != category:
                raise _CompatibilityFallback()
            retained.append((index, row))
    return ParsedBlock(count, dict(excluded), retained)


def parse_block(payload, mode):
    """Experimental bounded-block parser with ordered retained record offsets.

Blocks used by the performance harness have already passed the reference
per-record bound; native modes are NOT drop-in arbitrary-member stream readers.
"""
    global _SIMD_PARSER
    if mode not in MODES or len(payload) > MAX_BLOCK_BYTES:
        raise ValueError("unknown parser or block exceeds experiment bound")
    if mode in ("original", "cursor"):
        return _stdlib(payload, mode)
    if mode == "cpp-select":
        from .metrics_cpp_filter import parse_cpp_block
        return parse_cpp_block(payload)
    try:
        import simdjson
        if _SIMD_PARSER is None:
            _SIMD_PARSER = simdjson.Parser()
        if mode == "orjson":
            import orjson
            # orjson silently converts some out-of-range integers to float.
            # Native validation rejects those before materialization; no Python
            # full-byte regex prepass (measured slower than the old decoder).
            # This extra parse IS included in the benchmark cost/dependencies.
            _SIMD_PARSER.parse(payload)
            rows = orjson.loads(payload)
            if not isinstance(rows, list):
                raise ValueError("expected JSON array")
            return _filter(rows)
        if mode == "simd-full":
            rows = _SIMD_PARSER.parse(payload, recursive=True)
            if not isinstance(rows, list):
                raise ValueError("expected JSON array")
            return _filter(rows)
        return _select(_SIMD_PARSER.parse(payload), simdjson)
    except (ValueError, RuntimeError, OverflowError, _CompatibilityFallback):
        # Native strictness differences do not redefine the existing schema.
        # Invalid JSON still fails the reference decoder, even in excluded runs.
        # Drop a parser potentially referenced by an exception traceback.
        _SIMD_PARSER = None
        result = _stdlib(payload)
        result.compatibility_fallback = True
        return result


def ordered_blocks(blocks, mode, workers=1, *, max_pending=None):
    """Bounded spawn workers, one ordered parent consumer / checkpoint frontier.

The producer may be ONE decoder or a file. No worker opens the source archive
or writes SQLite. Results cannot commit out of source order. Startup/IPC/shutdown
are included when callers time consumption of this generator.
"""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for metadata, payload in blocks:
            yield metadata, parse_block(payload, mode)
        return
    limit = max_pending if max_pending is not None else 2 * workers
    if limit < 1:
        raise ValueError("pending bound must be positive")
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        try:
            for metadata, payload in blocks:
                # Drain BEFORE submitting beyond the bound, including results
                # completed out of order while the oldest block is still busy.
                if len(pending) >= limit:
                    item, future = pending.popleft()
                    yield item, future.result()
                pending.append((metadata, pool.submit(parse_block, payload, mode)))
            while pending:
                item, future = pending.popleft()
                yield item, future.result()
        finally:
            for _, future in pending:
                future.cancel()


def accept_block(store, parsed, member_index, member_name):
    """Centralized dedup/shards, preserving the live intake's counting semantics."""
    excluded = sum(parsed.excluded.values())
    if excluded + len(parsed.retained) != parsed.records:
        raise ValueError("inconsistent worker result")
    store.stats["counts"]["records"] += excluded
    store.stats["counts"]["exclude"] += excluded
    store.stats["excluded"].update(parsed.excluded)
    for record_index, wrapped in parsed.retained:
        store.accept(wrapped, member_index, member_name, record_index)
