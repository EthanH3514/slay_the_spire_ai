"""Build the pinned simulator and its public-state search adapter."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spire_ai.native_tools import deploy_mingw_runtime


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True,
    ).strip()


def checkout_mismatch(source, expected):
    for directory, revision in expected.items():
        checkout = source / directory
        if (not (checkout / ".git").exists()
                or git(checkout, "rev-parse", "HEAD") != revision
                or git(checkout, "status", "--porcelain", "--untracked-files=no",
                       "--ignore-submodules=all")):
            return checkout, revision
    return None


def simulator_source(requested, lock):
    expected = {".": lock["revision"]}
    expected.update({k: v["revision"] for k, v in lock["dependencies"].items()})
    submodule = (ROOT / "sts_lightspeed").resolve()
    if requested is None and checkout_mismatch(submodule, expected) is None:
        return submodule

    source = (requested or ROOT / "runtime/sts_lightspeed").resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "-c", "core.autocrlf=false", "clone", "--no-checkout", lock["repository"], source)
        run("git", "-C", source, "checkout", "--detach", lock["revision"])
        run("git", "-C", source, "submodule", "update", "--init", "--recursive")
    mismatch = checkout_mismatch(source, expected)
    if mismatch is not None:
        checkout, revision = mismatch
        raise SystemExit(f"Expected clean simulator checkout at {revision} in {checkout}")
    return source


def cmake_cache_args(build, source):
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key == "CMAKE_HOME_DIRECTORY:INTERNAL" and Path(value).resolve() != source:
                return ("--fresh",)
    return ()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Use this simulator checkout")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    lock = json.loads((ROOT / "src/spire_ai/simulator-lock.json").read_text())
    source = simulator_source(args.source, lock)
    compiler = shutil.which("g++")
    if compiler is None:
        raise SystemExit("Install MinGW-w64 GCC and put g++ on PATH.")
    build = ROOT / "runtime/native"
    run("cmake", *cmake_cache_args(build, source), "-S", source, "-B", build, "-G", "Ninja",
        f"-DCMAKE_CXX_COMPILER={Path(compiler).resolve().as_posix()}",
        f"-DPython_EXECUTABLE={Path(sys.executable).as_posix()}",
        "-DSTS_OPTIMIZATION=-O2", "-DSPIRE_AI_SESSION_SEARCH=ON",
        f"-DSPIRE_AI_SEARCH_SOURCE_ROOT={ROOT / 'native'}")
    run("cmake", "--build", build, "--target", "slaythespire", "-j", args.jobs)
    if sys.platform == "win32":
        deploy_mingw_runtime(Path(compiler), build)


if __name__ == "__main__":
    main()
