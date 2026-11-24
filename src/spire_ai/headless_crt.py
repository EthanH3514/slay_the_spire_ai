"""Opt-in crash reporting for unattended Windows native workers.

SetErrorMode controls OS dialogs, while _set_error_mode controls CRT assertions.
Neither disables an assertion nor changes successful simulator execution.
"""
from __future__ import annotations

import ctypes
import sys

OUT_TO_STDERR = 1
REPORT_ERRMODE = 3
WRITE_ABORT_MSG = 1
CALL_REPORTFAULT = 2
OS_ERROR_FLAGS = 0x0001 | 0x0002 | 0x8000


def configure() -> dict:
    if sys.platform != 'win32':
        return {'platform': sys.platform, 'configured': False}
    kernel = ctypes.windll.kernel32
    kernel.GetErrorMode.restype = ctypes.c_uint
    kernel.SetErrorMode.argtypes = [ctypes.c_uint]
    kernel.SetErrorMode.restype = ctypes.c_uint
    previous = kernel.GetErrorMode()
    kernel.SetErrorMode(previous | OS_ERROR_FLAGS)
    result = {}
    for name in ('msvcrt', 'ucrtbase'):
        crt = ctypes.CDLL(name)
        mode = crt._set_error_mode
        mode.argtypes = [ctypes.c_int]
        mode.restype = ctypes.c_int
        if mode(OUT_TO_STDERR) == -1 or mode(REPORT_ERRMODE) != OUT_TO_STDERR:
            raise RuntimeError(f'{name}: cannot direct native assertions to stderr')
        # The legacy OS msvcrt.dll has no _set_abort_behavior export. Its stderr
        # assertion path is covered by the child-process regression as well.
        abort = getattr(crt, '_set_abort_behavior', None)
        if abort is not None:
            abort.argtypes = [ctypes.c_uint, ctypes.c_uint]
            abort.restype = ctypes.c_uint
            abort(WRITE_ABORT_MSG, WRITE_ABORT_MSG | CALL_REPORTFAULT)
        result[name] = mode(REPORT_ERRMODE)
    return {'platform': sys.platform, 'configured': True, 'crt_modes': result,
            'os_mode': kernel.GetErrorMode(), 'assertions_enabled': True}
