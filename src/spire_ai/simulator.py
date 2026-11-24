from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


class SimulatorLoadError(RuntimeError):
    """Raised when the optional sts_lightspeed module cannot be loaded."""


@dataclass(frozen=True, slots=True)
class SimulatorSmokeResult:
    module_path: str
    seed: int
    seed_text: str
    ascension: int
    character: str
    observation_size: int
    current_hp: int
    maximum_hp: int
    gold: int
    act: int
    floor: int
    deck_size: int
    relic_count: int
    screen: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_sts_lightspeed(
    module_dir: Path,
    *,
    compiler_runtime_dir: Path | None = None,
) -> ModuleType:
    """Load the optional pybind11 simulator extension.

    MinGW-built Windows extensions depend on libgcc, libstdc++ and libwinpthread.
    Python 3.8+ no longer searches arbitrary PATH entries for extension dependencies,
    so the compiler runtime directory must be registered explicitly.
    """

    resolved_module_dir = module_dir.expanduser().resolve()
    if not resolved_module_dir.is_dir():
        raise SimulatorLoadError(
            f"simulator module directory does not exist: {resolved_module_dir}"
        )

    dll_handle = None
    if os.name == "nt":
        runtime_dir = compiler_runtime_dir or _infer_mingw_runtime_dir()
        if runtime_dir is None:
            raise SimulatorLoadError(
                "could not find the MinGW runtime directory; pass --compiler-runtime-dir"
            )
        resolved_runtime = runtime_dir.expanduser().resolve()
        if not resolved_runtime.is_dir():
            raise SimulatorLoadError(
                f"compiler runtime directory does not exist: {resolved_runtime}"
            )
        dll_handle = os.add_dll_directory(str(resolved_runtime))

    sys.path.insert(0, str(resolved_module_dir))
    try:
        return importlib.import_module("slaythespire")
    except (ImportError, OSError) as exc:
        raise SimulatorLoadError(f"failed to import sts_lightspeed: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(resolved_module_dir))
        except ValueError:
            pass
        # Keep the handle alive for the duration of import. Loaded dependent DLLs
        # remain mapped after import, so it is safe to close the search registration.
        if dll_handle is not None:
            dll_handle.close()


def smoke_test(
    module_dir: Path,
    *,
    seed: int = 123456789,
    ascension: int = 0,
    compiler_runtime_dir: Path | None = None,
) -> SimulatorSmokeResult:
    module = load_sts_lightspeed(
        module_dir,
        compiler_runtime_dir=compiler_runtime_dir,
    )
    try:
        game = module.GameContext(module.CharacterClass.IRONCLAD, seed, ascension)
        interface = module.getNNInterface()
        observation = interface.getObservation(game)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise SimulatorLoadError(f"sts_lightspeed API smoke test failed: {exc}") from exc

    return SimulatorSmokeResult(
        module_path=str(Path(module.__file__).resolve()),
        seed=seed,
        seed_text=str(module.get_seed_str(seed)),
        ascension=ascension,
        character="IRONCLAD",
        observation_size=len(observation),
        current_hp=int(game.cur_hp),
        maximum_hp=int(game.max_hp),
        gold=int(game.gold),
        act=int(game.act),
        floor=int(game.floor_num),
        deck_size=len(game.deck),
        relic_count=len(game.relics),
        screen=str(game.screen_state),
    )


def _infer_mingw_runtime_dir() -> Path | None:
    compiler = shutil.which("g++") or shutil.which("c++")
    return Path(compiler).resolve().parent if compiler else None
