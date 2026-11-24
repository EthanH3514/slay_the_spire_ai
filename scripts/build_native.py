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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runtime/sts_lightspeed")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    lock = json.loads((ROOT / "src/spire_ai/simulator-lock.json").read_text())
    source = args.source.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "-c", "core.autocrlf=false", "clone", "--no-checkout", lock["repository"], source)
        run("git", "-C", source, "checkout", "--detach", lock["revision"])
        run("git", "-C", source, "submodule", "update", "--init", "--recursive")
    expected = {".": lock["revision"]}
    expected.update({k: v["revision"] for k, v in lock["dependencies"].items()})
    for directory, revision in expected.items():
        checkout = source / directory
        if git(checkout, "rev-parse", "HEAD") != revision or git(
            checkout, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all",
        ):
            raise SystemExit(f"Expected clean simulator checkout at {revision} in {checkout}")
    compiler = shutil.which("g++")
    if compiler is None:
        raise SystemExit("Install MinGW-w64 GCC and put g++ on PATH.")
    build = ROOT / "runtime/native"
    run("cmake", "-S", source, "-B", build, "-G", "Ninja",
        f"-DCMAKE_CXX_COMPILER={Path(compiler).resolve().as_posix()}",
        f"-DPython_EXECUTABLE={Path(sys.executable).as_posix()}",
        "-DSTS_OPTIMIZATION=-O2", "-DSPIRE_AI_SESSION_SEARCH=ON",
        f"-DSPIRE_AI_SEARCH_SOURCE_ROOT={ROOT / 'native'}")
    run("cmake", "--build", build, "--target", "slaythespire", "-j", args.jobs)
    if sys.platform == "win32":
        deploy_mingw_runtime(Path(compiler), build)


if __name__ == "__main__":
    main()
