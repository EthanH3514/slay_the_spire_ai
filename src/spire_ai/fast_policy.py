"""Opt-in, search-free full-run baseline with explicit partial/fallback reports.

This is an initial behavior policy, not an RL agent or an expert oracle. Native
run-choice decoding is separate from scoring; unknown choices are never silently
presented as understood. Only sanitized PolicyInput is accepted.
"""
from __future__ import annotations

from typing import Callable, Mapping

from .actions import Action, ActionKind
from .combat_search import StarterRulePolicy
from .deck_profile import profile_cards, profile_value
from .environment import InvalidActionError
from .policy_api import HierarchicalPolicy, PolicyInput
from .policy_scope import assess_policy_scope
from .reward_policy import DeckRewardPolicy
from .rule_catalog import RuleCatalog
from .rule_coverage import _validate_input
from .run_options import public_run_options
from .potion_contracts import CONTRACT_BY_KEY, POTION_CONTRACT_VERSION


FAST_POLICY_VERSION = "public-fast-baseline-v7-complete-potions"
POTION_HEURISTICS = frozenset(CONTRACT_BY_KEY)


def key(value: object) -> str:
    return "".join(c for c in str(value).lower() if c.isalnum())


def _fallback(public: PolicyInput) -> Action:
    # Exit an unknown optional screen when possible; otherwise preserve stable
    # legal ordering. Neither this fallback nor its reasons are expert labels.
    return next((a for a in public.legal_actions if a.kind in {
        ActionKind.PROCEED, ActionKind.CANCEL,
    }), public.legal_actions[0])


def _relics(public: PolicyInput) -> set[str]:
    return {key(r.get("id")) for r in public.observation.get("inventory", {}).get("relics", ())}


def _health(public: PolicyInput) -> tuple[int, int]:
    player = public.observation.get("player", {})
    return int(player.get("hp", 0)), max(1, int(player.get("max_hp", 80)))


def _potion_space(public: PolicyInput) -> bool:
    slots = public.observation.get("inventory", {}).get("potions", ())
    return any(key(p.get("id")) in {"emptypotionslot", "potionslot", "emptypotionid"}
               for p in slots)


class FastCombatPolicy:
    def __init__(self, catalog: RuleCatalog):
        self.rule = StarterRulePolicy(catalog=catalog)
        self.potion_override = True
        self.last_decision: dict | None = None

    def choose(self, public: PolicyInput) -> Action:
        self.last_decision = None
        base = self.rule.choose(public)
        combat = public.observation.get("combat", {})
        normal = str(combat.get("input_state", "")).upper() == "PLAYER_NORMAL"
        potions = public.observation.get("inventory", {}).get("potions", ())
        gaps = set()
        rule_gaps = getattr(self.rule, "last_backend", {}).get("scope_gaps", ())
        gaps.update("rule_scope:" + reason for reason in rule_gaps)
        from .public_piles import draw_order_sampling_reason
        if draw_order_sampling_reason(combat):
            gaps.add("uncertain_draw_order_conservative_autoplay_valuation")
        for p in potions:
            if key(p.get("id")) not in POTION_HEURISTICS | {"emptypotionslot", "potionslot", "emptypotionid"}:
                gaps.add("potion_use_valuation_pending:" + str(p.get("id")))
        if public.observation.get("inventory", {}).get("relics"):
            gaps.add("general_relic_interaction_valuation_pending")
        if not normal:
            gaps.add("combat_selection_rule_only")
        for card in combat.get("hand", ()):
            if self.rule.recipes.lookup(card.get("id")) is None:
                gaps.add("unreviewed_combat_card:" + str(card.get("id")))
        from .native_search import potion_choice
        action = potion_choice(public, base) if self.potion_override else base
        if action != base:
            gaps.add("potion_estimate_ignores_reactive_damage_and_future_value")
        self.last_decision = {
            "mode": "combat_potion_heuristic" if action != base else "combat_rule_scope_fallback" if rule_gaps else "combat_rule",
            "action": action.as_dict(), "gaps": sorted(gaps),
            "fallback": bool(rule_gaps), "search_calls": 0,
            "potion_contract_version": POTION_CONTRACT_VERSION,
        }
        return action


class FastRunPolicy:
    def __init__(self, catalog: RuleCatalog):
        self.catalog = catalog
        self.reward = DeckRewardPolicy(catalog=catalog)
        from .event_policy import PublicEventPolicy
        self.event = PublicEventPolicy(catalog)
        self.last_decision: dict | None = None

    def _remove_score(self, card: Mapping) -> float:
        entry = self.catalog.lookup("card", card.get("id")) or {}
        if entry.get("type") == "CURSE":
            return 8.0
        if entry.get("type") == "STATUS":
            return 7.0
        if entry.get("rarity") == "BASIC":
            return 2.0 if entry.get("type") == "ATTACK" else 1.5
        return -1.0

    def choose(self, public: PolicyInput) -> Action:
        screen = public.observation.get("screen", {})
        if screen.get("type") == "CARD_REWARD":
            action = self.reward.choose(public)
            report = self.reward.last_decision
            gaps = set(report.get("limits", ()))
            gaps.update(issue for row in report.get("candidates", ()) for issue in row.get("unknown", ()))
            if report.get("reason"):
                gaps.add(report["reason"])
            self.last_decision = {"mode": report["mode"], "action": action.as_dict(),
                                  "gaps": sorted(gaps), "fallback": report["mode"] == "reward_fallback"}
            return action
        try:
            options = public_run_options(public)
        except ValueError as error:
            action = _fallback(public)
            self.last_decision = {"mode": "run_fallback", "fallback": True,
                                  "action": action.as_dict(), "gaps": [str(error)]}
            return action
        if screen.get("type") == "EVENT":
            action = self.event.choose(public, options)
            self.last_decision = self.event.last_decision
            return action
        hp, maximum = _health(public)
        fraction = hp / maximum
        relics = _relics(public)
        deck = public.observation.get("inventory", {}).get("deck", ())
        removable = max((self._remove_score(c) for c in deck), default=-1.0)
        gaps = set()
        scores = []
        cached_profile = None

        def deck_value(cards):
            return sum(profile_value(profile_cards(cards, self.catalog), hp_fraction=fraction,
                                     act=int(public.observation.get("run", {}).get("act", 1))).values())

        for action, row in zip(public.legal_actions, options):
            kind = row["kind"]
            score = -1.0
            if kind == "unknown":
                gaps.add("unknown_run_choice_semantics")
            elif kind == "event_choice":
                gaps.add("structured_event_value_pending")
            elif kind == "leave":
                score = 0.0
            elif kind == "gold":
                score = 10.0
            elif kind == "card_reward":
                score = 9.0
            elif kind in {"relic", "boss_relic"}:
                score = 5.0
                gaps.add("relic_acquisition_valuation_pending")
            elif kind == "potion":
                score = 4.0 if _potion_space(public) else -2.0
                gaps.add("potion_reward_replacement_pending")
            elif kind == "open_chest":
                score = 5.0
            elif kind == "key":
                score = -2.0  # This track ends at the Act 3 boss, not the Heart.
            elif kind in {"rest", "heal"}:
                amount = int(row.get("amount", int(maximum * .3)))
                if kind == "rest" and "regalpillow" in relics:
                    amount += 15
                effective = 0 if "markofthebloom" in relics else min(maximum - hp, amount)
                score = effective / maximum * (35 if fraction < .55 else 10)
                gaps.add("rest_and_heal_relic_interactions_partial")
            elif kind == "max_hp":
                score = 2.0
            elif kind == "smith":
                score = 5.0
            elif kind == "dig":
                score = 4.0
            elif kind == "lift":
                score = 3.0
            elif kind == "recall":
                score = -3.0
            elif kind in {"remove", "buy_remove"}:
                score = removable - float(row.get("price", 0)) / 100
                gaps.add("remove_value_ignores_deck_engine_and_removal_triggers")
            elif kind == "buy_card":
                if cached_profile is None:
                    cached_profile = deck_value(deck)
                score = (deck_value((*deck, row["card"])) - cached_profile - .12) * 5 - row["price"] / 100
                gaps.add("shop_card_value_partial_no_future_budget_plan")
            elif kind == "buy_potion":
                if _potion_space(public) and key(row["item_id"]) in POTION_HEURISTICS:
                    score = 2.0 - row["price"] / 50
                gaps.add("shop_potion_valuation_partial")
            elif kind == "buy_relic":
                gaps.add("shop_relic_valuation_pending")
            elif kind == "select_card":
                task, card = row["task"], row["card"]
                if task in {"remove", "transform", "transform_upgrade", "offer"}:
                    score = self._remove_score(card)
                    gaps.add("card_disposal_task_valuation_partial")
                elif task == "upgrade":
                    upgraded = {**card, "upgrades": card.get("upgrades", 0) + 1}
                    score = deck_value((upgraded,)) - deck_value((card,))
                    gaps.add("upgrade_profile_delta_not_full_deck_value")
                else:
                    score = 0.0
                    gaps.add("card_selection_task_valuation_pending:" + task)
            else:
                gaps.add("unvalued_run_choice:" + str(kind))
            scores.append((score, action))
        action = max(scores, key=lambda row: row[0])[1]
        selected = options[public.legal_actions.index(action)]
        fallback = selected["kind"] in {"unknown", "event_choice"}
        if screen.get("type") == "EVENT":
            gaps.add("event_display_parse_is_not_effect_fidelity_evidence")
        self.last_decision = {"mode": "run_fallback" if fallback else "run_heuristic",
                              "fallback": fallback, "action": action.as_dict(), "gaps": sorted(gaps),
                              "candidates": [{"action": a.as_dict(), "kind": row["kind"], "score": score}
                                             for (score, a), row in zip(scores, options)]}
        return action


class FastRoutePolicy:
    def __init__(self, catalog: RuleCatalog):
        self.catalog = catalog
        self.last_decision: dict | None = None

    def choose(self, public: PolicyInput) -> Action:
        hp, maximum = _health(public)
        try:
            rows = public_run_options(public)
        except ValueError as error:
            action = _fallback(public)
            self.last_decision = {"mode": "route_fallback", "fallback": True,
                                  "action": action.as_dict(), "gaps": [str(error)]}
            return action
        deck = public.observation.get("inventory", {}).get("deck", ())
        profile = profile_cards(deck, self.catalog)
        damage_density = profile.features.get("damage", 0) / max(1, len(deck))
        gold = int(public.observation.get("player", {}).get("gold", 0))
        scores = {"boss": 10, "treasure": 6, "rest": 7 if hp / maximum < .55 else 2,
                  "shop": 4 if gold >= 150 else -1, "event": 2,
                  "monster": 3 if hp / maximum >= .65 else 0,
                  "elite": 4 if hp / maximum >= .8 and damage_density >= 5 else -4}
        ranked = [(scores.get(row.get("room"), -10), action) for row, action in zip(rows, public.legal_actions)]
        action = max(ranked, key=lambda row: row[0])[1]
        fallback = not any(row.get("kind") == "route" for row in rows)
        map_gap = ("full_map_not_yet_used_for_route_value" if
                   public.observation.get("screen", {}).get("decision_context", {}).get("map_link_status") == "linked"
                   else "full_map_edges_and_future_campfire_distance_missing")
        self.last_decision = {"mode": "route_fallback" if fallback else "route_heuristic",
                              "fallback": fallback, "action": action.as_dict(),
                              "gaps": [map_gap,
                                       "route_threat_and_deck_strength_are_heuristic"]}
        return action


class FastBaselinePolicy:
    """Audited routing wrapper. Source environment and hidden channels never enter."""
    version = FAST_POLICY_VERSION

    def __init__(self, *, catalog: RuleCatalog | None = None,
                 decision_sink: Callable[[dict], None] | None = None):
        catalog = catalog or RuleCatalog.bundled()
        self.combat = FastCombatPolicy(catalog)
        self.run = FastRunPolicy(catalog)
        self.route = FastRoutePolicy(catalog)
        self.last_decision: dict | None = None
        self.decision_sink = decision_sink

    def choose(self, public: PolicyInput) -> Action:
        _validate_input(public)
        self.last_decision = None
        scope = assess_policy_scope(public.observation, public.legal_actions)
        public = PolicyInput(public.observation, scope.eligible_actions)
        if not public.legal_actions:
            raise InvalidActionError("fast policy received no eligible actions")
        name = HierarchicalPolicy.specialist_for(public).value
        specialist = getattr(self, name)
        action = specialist.choose(public)
        if action not in public.legal_actions:
            raise InvalidActionError("fast policy returned an illegal action")
        self.last_decision = {**specialist.last_decision, "version": self.version,
                              "specialist": name, "search_calls": 0,
                              "scope_exclusions": [{"action": item.action.as_dict(), "reason": item.reason}
                                                   for item in scope.excluded_actions]}
        if self.decision_sink:
            self.decision_sink(self.last_decision)
        return action
