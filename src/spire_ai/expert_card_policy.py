"""Isolated partial-input card BC adapter. No whole-controller promotion.

Only sanitized public states enter projection/inference. Reward provenance is
tracked from public transitions, not inferred from a seed or simulator internals.
The old native adapter is left unchanged (including the control arm's inputs).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .actions import Action, ActionKind
from .expert_cards import VERSION as ROW_VERSION, card_key, encode_row, public_choice
from .policy_api import PolicyInput
from .policy_scope import assess_policy_scope
from .rule_coverage import _validate_input

VERSION = "partial-expert-card-inference-v1"
BOWL_DISPLAY = "+2 max hp; lose card choice"
REWARD_SCREENS = frozenset({"COMBAT_REWARD", "CARD_REWARD"})


class RewardOrigin:
    """One fresh instance per run; no policy decisions or hidden state."""
    def __init__(self):
        self.battle_floor = None
        self.reward_floor = None

    def observe(self, public):
        _validate_input(public)
        floor = public.observation.get("run", {}).get("floor")
        screen = public.observation.get("screen", {}).get("type")
        if isinstance(public.observation.get("combat"), Mapping):
            self.battle_floor, self.reward_floor = floor, None
            return False
        if screen in REWARD_SCREENS and floor == self.battle_floor and floor is not None:
            self.reward_floor = floor
        elif screen not in REWARD_SCREENS or floor != self.reward_floor:
            self.battle_floor = self.reward_floor = None
        return (screen == "CARD_REWARD" and self.reward_floor is not None
                and floor > 0 and public.observation.get("run", {}).get("act") in (1, 2, 3))


def _card(raw, catalog):
    if not isinstance(raw, Mapping):
        raise ValueError("missing structured card")
    entry = catalog.lookup("card", raw.get("id"))
    upgrades = raw.get("upgrades")
    if entry is None or type(upgrades) is not int or upgrades < 0:
        raise ValueError("unknown card identity/upgrade count")
    # Persistent misc/bottles/etc are intentionally masked, as in the dataset.
    return {"id": entry["id"], "upgrades": upgrades}


def _metric_id(card):
    return card["id"] + ("+" + str(card["upgrades"]) if card["upgrades"] else "")


@dataclass(frozen=True)
class CardProjection:
    row: dict
    encoded: object
    actions: tuple[Action, ...]
    meanings: tuple[tuple, ...]

    def decode(self, index):
        if type(index) is not int or not 0 <= index < len(self.actions):
            raise ValueError("invalid projected action row")
        return self.actions[index]


def project_reward(public, encoder):
    """Map native legal choices into the EXACT historical partial-input schema.

    Native Singing Bowl currently has a null card, not a typed effect. Decode
    ONLY its pinned exact public display plus owned relic; never guess any
    other null CHOOSE. ConsoleSimulator source is pinned by the evaluation plan.
    """
    _validate_input(public)
    observation = public.observation
    screen, run = observation.get("screen", {}), observation.get("run", {})
    if screen.get("type") != "CARD_REWARD" or "combat" in observation:
        raise ValueError("not an outside card reward")
    if run.get("character") != "IRONCLAD" or run.get("act") not in (1, 2, 3):
        raise ValueError("card pilot character/act scope")
    floor, ascension = run.get("floor"), run.get("ascension")
    if (type(floor) is not int or floor <= 0 or
            type(ascension) is not int or not 0 <= ascension <= 20):
        raise ValueError("missing public floor/ascension")
    scope = assess_policy_scope(observation, public.legal_actions)
    if scope.eligible_actions != public.legal_actions or not public.legal_actions:
        raise ValueError("unexpected card eligibility change")
    inventory = observation.get("inventory", {})
    if not isinstance(inventory.get("deck"), (list, tuple)) or not isinstance(
            inventory.get("relics"), (list, tuple)):
        raise ValueError("missing public inventory")
    deck = [_card(c, encoder.catalog) for c in inventory["deck"]]
    relics = []
    for relic in inventory["relics"]:
        entry = encoder.catalog.lookup("relic", relic.get("id"))
        if entry is None or entry["id"] in relics:
            raise ValueError("unknown/duplicate relic")
        relics.append(entry["id"])
    options = screen.get("card_reward_options")
    if not isinstance(options, (list, tuple)):
        raise ValueError("missing structured reward choices")
    legal = {a: i for i, a in enumerate(public.legal_actions)}
    if len(legal) != len(public.legal_actions):
        raise ValueError("duplicate legal action")
    descriptions = screen.get("choices", ())
    seen, cards = set(), {}
    skip = bowl = None
    for option in options:
        action = Action.from_dict(option["action"])
        if action not in legal or action in seen:
            raise ValueError("reward/legal action mismatch")
        seen.add(action)
        if option.get("card") is not None:
            if action.kind != ActionKind.CHOOSE:
                raise ValueError("card is not a CHOOSE action")
            item = _card(option["card"], encoder.catalog)
            key = card_key(item)
            if key in cards:
                raise ValueError("duplicate reward identity outside training contract")
            cards[key] = (item, action)
        elif action.kind in (ActionKind.PROCEED, ActionKind.CANCEL):
            if skip is not None:
                raise ValueError("ambiguous reward skip")
            skip = action
        elif (action.kind == ActionKind.CHOOSE and
              len(descriptions) == len(public.legal_actions) and
              descriptions[legal[action]] == BOWL_DISPLAY and "SINGING_BOWL" in relics):
            if bowl is not None:
                raise ValueError("duplicate bowl option")
            bowl = action
        else:
            raise ValueError("unknown non-card reward option")
    if seen != set(legal) or skip is None or not cards:
        raise ValueError("incomplete card reward semantics")
    if ("SINGING_BOWL" in relics) != (bowl is not None):
        raise ValueError("bowl availability differs from training contract")
    ordered = [cards[k] for k in sorted(cards)]
    # SKIP is a dummy legal target required by the shared row encoder, NEVER
    # a human label, prediction, training update or extra model input.
    example = public_choice(deck, relics, floor, ascension,
        {"picked": "SKIP", "not_picked": [_metric_id(c) for c, _ in ordered]}, encoder.catalog)
    if example["public"]["observation"]["run"]["act"] != run["act"]:
        raise ValueError("historical floor/act convention mismatch")
    row = {"version": ROW_VERSION, **example}
    actions = tuple(a for _, a in ordered) + (skip,) + ((bowl,) if bowl else ())
    meanings = tuple(("card", *card_key(c)) for c, _ in ordered) + (("skip",),)
    if bowl:
        meanings += (("singing_bowl",),)
    encoded = encode_row(row, encoder)
    if len(encoded.legal_actions) != len(actions):
        raise ValueError("projection action count mismatch")
    return CardProjection(row, encoded, actions, meanings)


class ExpertCardPolicy:
    def __init__(self, checkpoint, expected_sha, encoder, *, width=32):
        import torch
        from .learning_model import TensorBatch
        from .run_learning_model import MODEL_VERSION, make_model
        path = Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("card checkpoint SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("artifact_type") != "offline_card_bc_partial_only" or
                payload.get("production_ready") is not False or payload.get("critic_fitted") is not False or
                payload["epoch"] != 3 or payload["contract"]["config"]["epochs"] != 3 or
                payload["contract"]["schema_id"] != encoder.schema_id or
                payload["contract"]["model_version"] != MODEL_VERSION):
            raise ValueError("card checkpoint contract mismatch")
        self.model, self.encoder = make_model(encoder, width=width), encoder
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        with torch.inference_mode():
            actual, _ = self.model(TensorBatch(**payload["probe"]))
        if not torch.equal(actual, payload["probe_logits"]):
            raise ValueError("card checkpoint exact public probe mismatch")
        self.last_decision = None

    def choose(self, public):
        import torch
        from .run_learning_model import tensorize
        projected = project_reward(public, self.encoder)
        with torch.inference_mode():
            distribution, _ = self.model.distribution(tensorize([projected.encoded], self.encoder))
        probabilities = distribution.probs[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("non-finite card probabilities")
        index = int(probabilities.argmax())
        action = projected.decode(index)
        if action not in public.legal_actions:
            raise ValueError("projected action is not currently legal")
        self.last_decision = {"version": VERSION, "mode": "greedy_partial_card_bc",
            "input_sha256": projected.row["input_sha256"], "action": action.as_dict(),
            "meaning": projected.meanings[index], "probabilities": probabilities.tolist(),
            "candidates": projected.meanings, "fallback": False}
        return action
