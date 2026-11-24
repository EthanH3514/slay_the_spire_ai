from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .combat_sampling import public_fingerprint
from .policy_api import PolicyInput, _reject_forbidden_policy_keys
from .rule_catalog import RuleCatalog


class RuleCoverage:
    """Public-only, bounded examples; observation is not full-mechanic verification."""

    def __init__(self, catalog: RuleCatalog, *, max_examples: int = 3) -> None:
        if not 0 <= max_examples <= 10:
            raise ValueError("max_examples must be between 0 and 10")
        self.catalog = catalog
        self.max_examples = max_examples
        self._observed: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._issues: dict[tuple[str, str], dict[str, Any]] = {}
        self._evidence: list[dict[str, Any]] = []
        self._intents: dict[tuple[str, str], dict[str, Any]] = {}
        self._local_moves: Counter[tuple[str, str]] = Counter()
        self.states = 0

    def observe(self, policy_input: PolicyInput, *, context: str) -> None:
        _validate_input(policy_input)
        self.states += 1
        observation = policy_input.observation
        run = observation.get("run", {})
        example = {"context": context, "public_fingerprint": public_fingerprint(policy_input)}
        if isinstance(run, Mapping):
            example.update({key: run[key] for key in ("floor", "act") if key in run})
        seen: set[tuple[str, str, str]] = set()
        seen_intents: set[tuple[str, str]] = set()

        def add(kind: str, value: object, variant: str = "") -> None:
            key = (kind, self.catalog.canonical_id(kind, value), variant)
            if key in seen:
                return
            seen.add(key)
            row = self._observed.setdefault(key, {
                "kind": kind, "id": key[1], "variant": variant, "observed_states": 0,
                "status": self.catalog.status(kind, value), "examples": [],
            })
            row["observed_states"] += 1
            if len(row["examples"]) < self.max_examples:
                row["examples"].append(example.copy())

        def cards(values: object) -> None:
            if isinstance(values, (tuple, list)):
                for card in values:
                    if isinstance(card, Mapping) and card.get("id") is not None:
                        upgrade = card.get("upgrades", card.get("upgraded", 0))
                        add("card", card["id"], f"upgrade={upgrade}")

        inventory = observation.get("inventory", {})
        if isinstance(inventory, Mapping):
            cards(inventory.get("deck", ()))
            for kind, plural in (("potion", "potions"), ("relic", "relics")):
                for item in inventory.get(plural, ()):
                    if isinstance(item, Mapping) and item.get("id") is not None:
                        if kind == "potion" and self.catalog.canonical_id(kind, item["id"]) == "empty_potion_slot":
                            continue
                        add(kind, item["id"])
        combat = observation.get("combat", {})
        if isinstance(combat, Mapping):
            for pile in (
                "hand", "draw_pile", "draw_pile_unordered", "draw_pile_known_bottom", "discard_pile", "exhaust_pile", "limbo",
            ):
                cards(combat.get(pile, ()))
            for monster in combat.get("monsters", ()):
                if isinstance(monster, Mapping):
                    if isinstance(monster.get("stasis_card"), Mapping):
                        cards((monster["stasis_card"],))
                    if monster.get("id") is not None:
                        add("monster", monster["id"])
                    move = monster.get("move_id")
                    if move is not None:
                        if str(move).lstrip("-").isdigit():
                            # CommunicationMod uses enemy-local bytes, not global move IDs.
                            self._local_moves[(
                                self.catalog.canonical_id("monster", monster.get("id")), str(move),
                            )] += 1
                        else:
                            add("move", move)
                    # Generic real-game intent is not a native specific move ID.
                    intent = monster.get("intent", monster.get("move_id"))
                    if intent is not None:
                        intent_key = (
                            self.catalog.canonical_id("monster", monster.get("id")), str(intent),
                        )
                        row = self._intents.setdefault(intent_key, {
                            "monster_id": intent_key[0], "public_intent": intent_key[1],
                            "observed_states": 0, "examples": [],
                        })
                        if intent_key not in seen_intents:
                            row["observed_states"] += 1
                            seen_intents.add(intent_key)
                        if len(row["examples"]) < self.max_examples:
                            intent_example = example.copy()
                            intent_example.update({
                                key: monster[key] for key in (
                                    "move_base_damage", "move_hits", "intent_base_damage",
                                    "intent_damage", "intent_hits",
                                ) if key in monster
                            })
                            row["examples"].append(intent_example)

    def record_issue(
        self, category: str, reason: str, policy_input: PolicyInput, *, context: str,
    ) -> None:
        _validate_input(policy_input)
        key = (category, reason)
        row = self._issues.setdefault(key, {
            "category": category, "reason": reason, "count": 0, "examples": [],
        })
        row["count"] += 1
        if len(row["examples"]) < self.max_examples:
            row["examples"].append({"context": context, "public_input": policy_input.as_dict()})

    def record_evidence(self, *, reference: str, scope: str, outcome: str) -> None:
        if outcome not in {"matched_prefix", "diverged", "incomplete"}:
            raise ValueError("evidence outcome must describe a scoped comparison")
        self._evidence.append({"reference": reference, "scope": scope, "outcome": outcome})

    def as_dict(self) -> dict[str, Any]:
        # JSON roundtrip detaches nested examples/status from recorder internals.
        return json.loads(json.dumps({
            "schema_version": 1, "catalog": self.catalog.summary(), "states": self.states,
            "catalog_provenance": self.catalog.provenance(),
            "observed": [self._observed[key] for key in sorted(self._observed)],
            "missing_catalog_ids": dict(Counter(
                row["kind"] for row in self._observed.values() if not row["status"]["imported"]
            )),
            "issues": [self._issues[key] for key in sorted(self._issues)],
            "evidence": self._evidence,
            "public_intents": [self._intents[key] for key in sorted(self._intents)],
            "backend_local_moves": [
                {"monster_id": key[0], "local_move_id": key[1], "occurrences": count,
                 "mapping_status": "not_mapped_to_native_global_move"}
                for key, count in sorted(self._local_moves.items())
            ],
            "verification_note": "evidence is scoped; observations never certify full mechanics",
        }))

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def _validate_input(policy_input: PolicyInput) -> None:
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("coverage accepts only sanitized PolicyInput")
    _reject_forbidden_policy_keys(policy_input.observation, reject_seeds=True)
