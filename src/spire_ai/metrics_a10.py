"""Versioned A10--A20 intake; original A20 artifacts/modules stay unchanged.

The context adapter changes only the cohort function used by the old durable
store/reader. Native results are always rechecked by this Python specification.
False outcomes remain audit-only to detect win/loss UUID conflicts.
"""
from collections import Counter
from contextlib import contextmanager
import ctypes
from functools import lru_cache
import io
import json
from pathlib import Path

from .metrics_cards import cohort as a20_cohort
from .metrics_cpp_filter import _Report
from .metrics_decode_probe import ParsedBlock, cursor_records, reject_constant
from .metrics_stream import file_sha
from .native_tools import _no_windows_error_dialogs

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'metrics-a10-a20-intake-v1'
BUILD = ROOT/'runtime/metrics-a10-filter-v1'
MAX_BLOCK = 32 * 1024 * 1024
REASONS = ('wrong_character', 'outside_a10_a20',
    'unsupported_or_unknown_flag:is_daily', 'unsupported_or_unknown_flag:is_endless',
    'unsupported_or_unknown_flag:is_trial', 'unsupported_or_unknown_flag:is_beta',
    'unsupported_or_unknown_flag:chose_seed', 'special_seed', 'modded_or_unknown_flag', 'unknown_outcome')


def cohort(run):
    if run.get('character_chosen') != 'IRONCLAD':
        return 'exclude', 'wrong_character'
    level = run.get('ascension_level')
    if type(level) is not int or not 10 <= level <= 20:
        return 'exclude', 'outside_a10_a20'
    # Reuse precisely the old flag/outcome specification, not a changed run
    # history. Actual level remains untouched in storage, identity and features.
    return a20_cohort(dict(run, ascension_level=20))


@contextmanager
def intake_bindings():
    """Single coordinator context, never installed by a normal policy import."""
    from . import metrics_stream, metrics_dataset
    old = (metrics_stream.cohort, metrics_dataset.cohort)
    metrics_stream.cohort = metrics_dataset.cohort = cohort
    try:
        yield
    finally:
        metrics_stream.cohort, metrics_dataset.cohort = old


def reference(payload):
    excluded, retained, count = Counter(), [], 0
    with io.TextIOWrapper(io.BytesIO(payload), encoding='utf-8', newline='') as stream:
        for index, wrapped in enumerate(cursor_records(stream)):
            if not isinstance(wrapped.get('event'), dict):
                raise ValueError('expected event wrapper')
            category, reason = cohort(wrapped['event'])
            count += 1
            if category == 'exclude':
                excluded[reason] += 1
            else:
                retained.append((index, wrapped))
    return ParsedBlock(count, dict(excluded), retained)


@lru_cache(maxsize=2)
def load_native(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'build.json').read_text(encoding='utf-8'))
    if manifest['abi'] != 2 or manifest['cohort'] != VERSION or manifest['max_block_bytes'] != MAX_BLOCK:
        raise ValueError('A10 native contract mismatch')
    for name, expected in manifest['inputs'].items():
        if file_sha(ROOT/name) != expected:
            raise ValueError('A10 native source drift: '+name)
    path = (directory/manifest['library']).resolve()
    if path.parent != directory.resolve() or file_sha(path) != manifest['library_sha256']:
        raise ValueError('A10 library identity mismatch')
    with _no_windows_error_dialogs():
        dll = ctypes.CDLL(str(path))
    dll.spire_metrics_abi_version.restype = ctypes.c_uint32
    if dll.spire_metrics_abi_version() != 2:
        raise ValueError('A10 library ABI mismatch')
    dll.spire_metrics_filter.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(_Report))]
    dll.spire_metrics_filter.restype = ctypes.c_int
    dll.spire_metrics_free.argtypes = [ctypes.POINTER(_Report)]
    dll.spire_metrics_free.restype = None
    return dll


def parse(payload, directory=BUILD):
    if not isinstance(payload, bytes) or len(payload) > MAX_BLOCK:
        raise ValueError('bounded bytes required')
    dll = load_native(str(directory))
    result = ctypes.POINTER(_Report)()
    status = dll.spire_metrics_filter(payload, len(payload), ctypes.byref(result))
    if status == 1:
        answer = reference(payload)
        answer.compatibility_fallback = True
        return answer
    if status != 0 or not result:
        raise RuntimeError('A10 native filter failure: '+str(status))
    try:
        report = result.contents
        excluded = {key:int(report.excluded[i]) for i,key in enumerate(REASONS) if report.excluded[i]}
        if (report.records > len(payload)//2 or report.selected_count > report.records or
                sum(excluded.values())+report.selected_count != report.records or
                (report.selected_count and not report.selected)):
            raise ValueError('invalid native counts')
        retained, last_index, last_end = [], -1, 0
        for i in range(report.selected_count):
            item = report.selected[i]
            if (not last_index < item.index < report.records or item.offset < last_end or
                    not item.length or item.offset+item.length > len(payload) or item.category not in (1, 2)):
                raise ValueError('invalid native selection span')
            wrapped = json.loads(payload[item.offset:item.offset+item.length], parse_constant=reject_constant)
            category = 'win_candidate' if item.category == 1 else 'loss_retained_only'
            if not isinstance(wrapped.get('event'), dict) or cohort(wrapped['event'])[0] != category:
                raise ValueError('native/reference cohort mismatch')
            retained.append((int(item.index), wrapped))
            last_index, last_end = item.index, item.offset+item.length
        return ParsedBlock(int(report.records), excluded, retained)
    finally:
        dll.spire_metrics_free(result)
