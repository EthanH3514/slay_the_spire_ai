"""Local MinGW runtime deployment and bounded native development tools."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
from typing import Mapping, Sequence


MINGW_RUNTIME_DLLS = (
    "libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll",
)
_WINDOWS = sys.platform == "win32"
_ERROR_MODE_LOCK = Lock()
_LOADER_ERRORS = {
    0xC0000135: "DLL not found or could not be loaded",
    0xC000007B: "invalid executable/DLL format or architecture",
    0xC0000139: "DLL entry point not found",
    0xC0000142: "DLL initialization failed",
}


class NativeToolError(RuntimeError):
    def __init__(
        self, category: str, message: str, *, returncode: int | None = None,
        stdout: str = "", stderr: str = "",
    ) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def compiler_from_cache(build_dir: Path) -> Path:
    """Use the compiler that built the native objects, not whichever is on PATH."""
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                compiler = Path(line.split("=", 1)[1]).resolve()
                if compiler.is_file():
                    return compiler
                raise NativeToolError("missing_compiler", f"cached compiler missing: {compiler}")
    raise NativeToolError(
        "missing_compiler", f"no CMake C++ compiler in {cache}; bootstrap the simulator first",
    )


def deploy_mingw_runtime(compiler: Path, output_dir: Path) -> tuple[str, ...]:
    """Copy matching runtime DLLs next to binaries; leave identical copies untouched."""
    compiler = compiler.resolve()
    if not compiler.is_file():
        raise NativeToolError("missing_compiler", f"compiler missing: {compiler}")
    sources = tuple(compiler.parent / name for name in MINGW_RUNTIME_DLLS)
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise NativeToolError(
            "missing_dependency", "compiler runtimes missing: " + ", ".join(missing),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sources:
        target = output_dir / source.name
        if target.is_file() and hashlib.sha256(target.read_bytes()).digest() == (
            hashlib.sha256(source.read_bytes()).digest()
        ):
            continue
        shutil.copy2(source, target)
        copied.append(source.name)
    return tuple(copied)


@contextmanager
def _no_windows_error_dialogs():
    if not _WINDOWS:
        yield
        return
    # Children inherit the parent's error mode. Restore it even on timeout/error.
    # Serialize our callers because SetErrorMode is process-wide.
    with _ERROR_MODE_LOCK:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetErrorMode.restype = ctypes.c_uint
        kernel.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel.SetErrorMode.restype = ctypes.c_uint
        previous = kernel.GetErrorMode()
        kernel.SetErrorMode(previous | 0x0001 | 0x0002 | 0x8000)
        try:
            yield
        finally:
            kernel.SetErrorMode(previous)


def _output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_native_tool(
    executable: Path, arguments: Sequence[str] = (), *, timeout: float = 15,
    require_runtime: bool = True, env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Fail before launch on absent local runtimes; kill/wait the child on timeout.

    This supervises one tool process, not a subprocess tree. Fixtures and the
    catalogue exporter do not spawn children. Compiler calls use a separate budget.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("native tool timeout must be finite and positive")
    executable = executable.resolve()
    if not executable.is_file():
        raise NativeToolError("missing_executable", f"executable missing: {executable}")
    if _WINDOWS and require_runtime:
        missing = [
            name for name in MINGW_RUNTIME_DLLS
            if not (executable.parent / name).is_file()
        ]
        if missing:
            raise NativeToolError(
                "missing_dependency",
                f"{executable.name} needs adjacent {', '.join(missing)}; "
                "run scripts/run_native_tests.py --deploy-only for this build directory",
            )
    try:
        with _no_windows_error_dialogs():
            result = subprocess.run(
                [str(executable), *arguments], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                env=None if env is None else dict(env),
                creationflags=subprocess.CREATE_NO_WINDOW if _WINDOWS else 0,
            )
    except subprocess.TimeoutExpired as error:
        raise NativeToolError(
            "timeout", f"{executable.name} exceeded {timeout:g}s; direct child killed and reaped",
            stdout=_output_text(error.stdout), stderr=_output_text(error.stderr),
        ) from error
    except OSError as error:
        raise NativeToolError("launch_error", f"{executable.name}: {error}") from error
    if result.returncode:
        code = result.returncode & 0xFFFFFFFF
        loader_reason = _LOADER_ERRORS.get(code) if _WINDOWS else None
        raise NativeToolError(
            "loader_error" if loader_reason else "process_failure",
            f"{executable.name} exited {result.returncode} (0x{code:08X})"
            + (f": {loader_reason}" if loader_reason else ""),
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        )
    return result
