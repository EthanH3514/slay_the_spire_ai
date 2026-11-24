from __future__ import annotations

import json
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Mapping

from .combat_sampling import (
    CARD_HISTORY_VERSION, CARD_SAMPLING_STATUS, ENCOUNTER_SAMPLING_STATUS,
    RESAMPLER_VERSION,
)
from .identifiers import canonical_identifier
from .encounter_recipes import ENCOUNTER_RECIPE_VERSION
from .mechanism_evidence import (
    DEFAULT_MECHANISM_EVIDENCE, MECHANISM_EVIDENCE_VERSION, EvidenceCapability,
)
from .policy_api import _freeze_public_value, _thaw_public_value


KINDS = {"card": "cards", "monster": "monsters", "move": "moves", "encounter": "encounters",
         "potion": "potions", "relic": "relics"}


class RuleCatalog:
    """Static metadata. Importing an entry grants no sampler/verification capability."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if document.get("schema_version") != 1:
            raise ValueError("unsupported rule catalogue schema")
        self.source = _freeze_public_value(document["source"])
        self._entries: dict[str, Mapping[str, Any]] = {}
        self._aliases: dict[tuple[str, str], str] = {}
        for kind, plural in KINDS.items():
            entries = {}
            # New item tables are optional for older fixture/catalogue documents;
            # the original four tables remain mandatory.
            records = (document["definitions"].get(plural, ()) if kind in {"potion", "relic"}
                       else document["definitions"][plural])
            for raw in records:
                key = canonical_identifier(raw["id"], "monster" if kind == "encounter" else kind)
                if key is None or key in entries:
                    raise ValueError(f"duplicate/invalid {kind} id: {key}")
                entries[key] = _freeze_public_value(raw)
                # Do not index display names: all four starter Strikes share a name.
                for alias in (raw["id"], raw.get("game_id", raw["id"])):
                    token = canonical_identifier(alias, kind)
                    previous = self._aliases.setdefault((kind, token), key)
                    if previous != key:
                        raise ValueError(f"ambiguous {kind} alias: {alias}")
            self._entries[kind] = MappingProxyType(entries)

    @classmethod
    def bundled(cls) -> RuleCatalog:
        resource = files("spire_ai").joinpath("rule-catalog-v1.json")
        return cls(json.loads(resource.read_text(encoding="utf-8")))

    def canonical_id(self, kind: str, value: object) -> str:
        if kind not in KINDS:
            raise ValueError(f"unknown catalogue kind: {kind}")
        token = canonical_identifier(value, "monster" if kind == "encounter" else kind)
        return self._aliases.get((kind, token), token or "<missing>")

    def lookup(self, kind: str, value: object) -> Mapping[str, Any] | None:
        return self._entries[kind].get(self.canonical_id(kind, value))

    def entries(self, kind: str) -> tuple[Mapping[str, Any], ...]:
        """Enumerate immutable imported records without granting behavior support."""
        if kind not in KINDS:
            raise ValueError(f"unknown catalogue kind: {kind}")
        return tuple(self._entries[kind].values())

    def status(self, kind: str, value: object) -> dict[str, Any]:
        key = self.canonical_id(kind, value)
        imported = key in self._entries[kind]
        evidence = (
            DEFAULT_MECHANISM_EVIDENCE.status(kind, key, imported=imported)
            if kind in {"card", "encounter"} else None
        )
        # Descriptive mirrors of runtime guards, NOT authority to permit a native root.
        card_supported = (
            kind == "card"
            and evidence["capabilities"][EvidenceCapability.SEARCH.value]["supported"]
        )
        encounter_supported = (
            kind == "encounter"
            and evidence["capabilities"][
                EvidenceCapability.COMBAT_RESAMPLING.value
            ]["supported"]
        )
        return {
            "imported": imported,
            "sampling": (
                CARD_SAMPLING_STATUS if card_supported else
                ENCOUNTER_SAMPLING_STATUS if encounter_supported else
                "not_audited"
            ),
            "resampler_version": RESAMPLER_VERSION,
            "card_history_version": CARD_HISTORY_VERSION,
            "verification": (
                MECHANISM_EVIDENCE_VERSION
                if evidence is not None and evidence["mechanisms"]
                else "no_executable_evidence"
            ),
            "evidence": evidence,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": self.source["revision"],
            "counts": {KINDS[kind]: len(rows) for kind, rows in self._entries.items()},
            "native_resampler_strategy": "declarative_recipes",
            "native_resampler_encounters": len(
                DEFAULT_MECHANISM_EVIDENCE.registered_ids(
                    "encounter", EvidenceCapability.COMBAT_RESAMPLING,
                )
            ),
            "encounter_recipe_version": ENCOUNTER_RECIPE_VERSION,
            "card_history_version": CARD_HISTORY_VERSION,
            "mechanism_evidence_version": MECHANISM_EVIDENCE_VERSION,
            "behavior_rules": "referenced_native_source_not_translated",
        }

    def provenance(self) -> dict[str, Any]:
        return _thaw_public_value(self.source)
