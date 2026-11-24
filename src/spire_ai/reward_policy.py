"""A small, explainable Run specialist. Only CARD_REWARD is strategically ranked."""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Callable, Mapping

from .actions import Action, ActionKind
from .deck_profile import DECK_PROFILE_VERSION, profile_cards, profile_observation, profile_value
from .environment import InvalidActionError
from .policy_api import DecisionPolicy, PolicyInput
from .policy_scope import assess_policy_scope
from .rule_catalog import RuleCatalog


REWARD_POLICY_VERSION = "deck-reward-heuristic-v1"


@lru_cache(maxsize=1)
def _native_reward_names() -> dict[str, tuple[str, ...]]:
    names: dict[str, list[str]] = {}
    for entry in RuleCatalog.bundled().entries("card"):
        names.setdefault(entry["name"], []).append(entry["id"])
    return {name: tuple(ids) for name, ids in names.items()}


def native_reward_card(description: str) -> dict | None:
    """Decode ONLY the pinned ConsoleSimulator/Card printer grammar.

    Exact unique native names, never localized/fuzzy aliases. Ambiguous starter
    names and new printer formats remain unknown. See src/sim/PrintHelpers.cpp.
    """
    match = re.fullmatch(r"take (.+?)(?:=(\d+))?(?:\+(\d*))?", description)
    if not match:
        return None
    name, misc, upgrade_text = match.groups()
    ids = _native_reward_names().get(name, ())
    if len(ids) != 1:
        return None
    identifier = ids[0]
    if (misc is not None and identifier != "RITUAL_DAGGER"
            or upgrade_text not in {None, ""} and identifier != "SEARING_BLOW"):
        return None
    upgrades = 0 if upgrade_text is None else int(upgrade_text or "1")
    return {"id": identifier, "upgrades": upgrades,
            **({"misc": int(misc)} if misc is not None else {})}


def _reward_options(public: PolicyInput) -> tuple[tuple[Action, Mapping | None], ...]:
    screen = public.observation["screen"]
    native = screen.get("card_reward_options")
    if native is not None:
        if not isinstance(native, (list, tuple)):
            raise ValueError("invalid native reward options")
        mapped = {}
        for row in native:
            if not isinstance(row, Mapping) or not isinstance(row.get("action"), Mapping):
                raise ValueError("invalid native reward option")
            action = Action.from_dict(row["action"])
            if action in mapped:
                raise ValueError("duplicate native reward action")
            card = row.get("card")
            if action.kind not in {ActionKind.CHOOSE, ActionKind.PROCEED}:
                raise ValueError("unreviewed native reward action")
            if action.kind is ActionKind.PROCEED and card is not None:
                raise ValueError("native skip must not acquire a card")
            if action.kind is ActionKind.CHOOSE and not isinstance(card, Mapping):
                raise ValueError(f"unknown native reward candidate at choice {action.choice_index} (including bowl)")
            mapped[action] = card
        if set(mapped) != set(public.legal_actions):
            raise ValueError("native reward options do not match legal actions")
        return tuple((action, mapped[action]) for action in public.legal_actions)

    # CommunicationMod: rewardGroup order is shared by cards and choice_list.
    state = screen.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("cards"), (list, tuple)):
        raise ValueError("structured reward cards are missing")
    cards = state["cards"]
    options = []
    for action in public.legal_actions:
        if action.kind is ActionKind.CHOOSE:
            index = action.choice_index
            if type(index) is not int or not 0 <= index < len(cards):
                raise ValueError("unknown reward choice (including bowl)")
            if not isinstance(cards[index], Mapping):
                raise ValueError("invalid reward card")
            options.append((action, cards[index]))
        elif action.kind is ActionKind.CANCEL and state.get("skip_available") is True:
            options.append((action, None))
        else:
            raise ValueError("unreviewed reward action")
    return tuple(options)


class DeckRewardPolicy:
    """Public-only reward heuristic with explicit fallback and detached reports.

    May be installed as HierarchicalPolicy.run, or as a wrapper around a live
    random policy. Non-reward behavior remains the supplied fallback unchanged.
    """

    version = REWARD_POLICY_VERSION

    def __init__(self, fallback: DecisionPolicy | None = None, catalog: RuleCatalog | None = None,
                 decision_sink: Callable[[dict], None] | None = None):
        self.fallback = fallback
        self.catalog = catalog or RuleCatalog.bundled()
        self.last_decision: dict | None = None
        self.last_scope_decision = None
        self.decision_sink = decision_sink

    def choose(self, public: PolicyInput) -> Action:
        self.last_decision = None
        self.last_scope_decision = assess_policy_scope(public.observation, public.legal_actions)
        public = PolicyInput(public.observation, self.last_scope_decision.eligible_actions)
        if not public.legal_actions:
            raise InvalidActionError("reward policy received no legal actions")
        screen = public.observation.get("screen")
        reward = (isinstance(screen, Mapping) and str(screen.get("type", "")).upper() == "CARD_REWARD"
                  and not isinstance(public.observation.get("combat"), Mapping))
        if reward:
            try:
                action = self._rank(public)
            except ValueError as exc:
                action = self._fallback(public, reward=True)
                self.last_decision = {"version": self.version, "mode": "reward_fallback",
                                      "reason": str(exc), "action": action.as_dict()}
            if self.decision_sink:
                self.decision_sink(self.last_decision)
            return action
        action = self._fallback(public)
        self.last_decision = {"version": self.version, "mode": "non_reward_fallback",
                              "action": action.as_dict()}
        return action

    def _fallback(self, public: PolicyInput, *, reward: bool = False) -> Action:
        # On incomplete reward semantics prefer an actual available skip. Never
        # fabricate skip, nor reinterpret an unknown CHOOSE as a card or a bowl.
        screen = public.observation.get("screen", {})
        state = screen.get("state", {}) if isinstance(screen, Mapping) else {}
        for action in public.legal_actions if reward else ():
            native_skip = (action.kind is ActionKind.PROCEED
                           and "card_reward_options" in screen)
            live_skip = (action.kind is ActionKind.CANCEL and isinstance(state, Mapping)
                         and state.get("skip_available") is True)
            if native_skip or live_skip:
                return action
        if self.fallback is not None:
            action = self.fallback.choose(public)
        else:
            controls = {ActionKind.WAIT, ActionKind.REQUEST_STATE, ActionKind.CANCEL}
            action = next((a for a in public.legal_actions if a.kind not in controls), public.legal_actions[0])
        if action not in public.legal_actions:
            raise InvalidActionError("reward fallback returned illegal action")
        return action

    def _rank(self, public: PolicyInput) -> Action:
        options = _reward_options(public)
        before = profile_observation(public.observation, self.catalog)
        if any("unreviewed_card" in c.unknown for c in before.cards):
            raise ValueError("master deck contains unreviewed cards")
        player = public.observation.get("player", {})
        run = public.observation.get("run", {})
        if not isinstance(player, Mapping) or not isinstance(run, Mapping):
            raise ValueError("invalid public run/player context")
        hp, maximum = player.get("hp"), player.get("max_hp")
        valid_hp = type(hp) is int and type(maximum) is int and 0 <= hp <= maximum and maximum > 0
        fraction = hp / maximum if valid_hp else 1.0
        act = run.get("act", 1)
        valid_act = type(act) is int and 1 <= act <= 4 and "act" in run
        if not valid_act:
            act = 1
        before_value = profile_value(before, hp_fraction=fraction, act=act)
        candidates = []
        for action, card in options:
            if card is None:
                candidates.append({"action": action.as_dict(), "card": None,
                                   "score": 0.0, "components": {"skip": 0.0},
                                   "unknown": [], "after": None})
                continue
            after = profile_cards((*public.observation["inventory"]["deck"], card), self.catalog)
            description = profile_cards((card,), self.catalog).cards[0]
            if "unreviewed_card" in description.unknown:
                raise ValueError(f"unreviewed reward card: {description.identifier}")
            after_value = profile_value(after, hp_fraction=fraction, act=act)
            components = {key: round(value - before_value[key], 8) for key, value in after_value.items()}
            components["acquisition_dilution"] = -.12
            components["unvalued_mechanisms"] = -.15 * len(description.unknown)
            candidates.append({"action": action.as_dict(), "card": description.as_dict(),
                               "score": round(sum(components.values()), 8), "components": components,
                               "unknown": list(description.unknown), "after": after.as_dict()})
        if not candidates:
            raise ValueError("no reward candidates")
        # Exact ties favor skip; card ties remain at their original legal index.
        chosen = max(candidates, key=lambda c: (c["score"], c["card"] is None))
        action = Action.from_dict(chosen["action"])
        self.last_decision = {
            "version": self.version, "profile_version": DECK_PROFILE_VERSION,
            "mode": "reward_heuristic_partial" if before.unknown or any(c["unknown"] for c in candidates) else "reward_heuristic",
            "action": action.as_dict(), "before": before.as_dict(), "candidates": candidates,
            "context": {"hp_fraction": fraction, "hp_defaulted": not valid_hp,
                        "act": act, "act_defaulted": not valid_act},
            "limits": ["hand_set_utility_not_win_probability", "relic_and_potion_valuation_pending",
                       "no_co_draw_or_play_sequence_prediction", "public_threat_valuation_pending"],
        }
        return action
