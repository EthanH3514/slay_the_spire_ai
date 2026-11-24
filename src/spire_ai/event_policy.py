"""Public, compositional event baseline; scores are utilities, not win odds.

Display contracts are not effect-order or simulator-fidelity certificates. No
unrevealed reward identity, event outcome, map realization or RNG enters scoring.
An unvalued clause makes the WHOLE offer unvalued, including its tempting reward.
"""
from __future__ import annotations

from .deck_profile import profile_cards, profile_value
from .event_options import EVENT_OPTIONS_VERSION
from .potion_contracts import EMPTY_SLOTS, potion_key

EVENT_POLICY_VERSION = "public-event-resource-heuristic-v1"
UNREMOVABLE = frozenset({"ASCENDERS_BANE", "NECRONOMICURSE", "CURSE_OF_THE_BELL"})


class PublicEventPolicy:
    def __init__(self, catalog):
        self.catalog = catalog
        self.last_decision = None

    def _removal(self, card):
        entry = self.catalog.lookup("card", card.get("id")) or {}
        if entry.get("id") in UNREMOVABLE:
            return None
        if entry.get("type") == "CURSE":
            return 8.
        if entry.get("type") == "STATUS":
            return 6.
        if entry.get("rarity") == "BASIC":
            return 2.
        return -2.

    def _upgrade(self, card, fraction, act):
        entry = self.catalog.lookup("card", card.get("id")) or {}
        if not entry or entry.get("type") in {"STATUS", "CURSE"} or card.get("upgrades", 0):
            return None
        def value(c):
            return sum(profile_value(profile_cards((c,), self.catalog), hp_fraction=fraction, act=act).values())
        # A partial card-profile delta; no claim that all upgraded cards can be
        # played together or that this measures an infinite engine's value.
        return max(0., (value({**card, "upgrades": 1}) - value(card)) * 5)

    def _score(self, public, row):
        contract = row.get("event_choice", {})
        gaps, components = set(), []
        if (contract.get("version") != EVENT_OPTIONS_VERSION or contract.get("status") != "parsed"
                or contract.get("unknown_clauses") or not contract.get("display_order")):
            return {"score": None, "components": [], "gaps": ["unknown_run_choice_semantics"]}
        player = public.observation.get("player", {})
        hp, maximum = int(player.get("hp", 0)), max(1, int(player.get("max_hp", 80)))
        gold = int(player.get("gold", 0))
        fraction = hp / maximum
        hp_weight = 1. + 2. * (1. - fraction)
        inventory = public.observation.get("inventory", {})
        relics = {potion_key(r.get("id")) for r in inventory.get("relics", ())}
        deck = [dict(c) for c in inventory.get("deck", ())]
        slots = sum(potion_key(p.get("id")) in EMPTY_SLOTS for p in inventory.get("potions", ()))
        act = int(public.observation.get("run", {}).get("act", 1))
        loss, healing, max_loss, score = 0, 0, 0, 0.
        unsupported = False
        for op in contract["display_order"]:
            kind, n = op["kind"], int(op.get("amount", op.get("count", 1)))
            value = None
            if kind == "leave":
                value = 0.
            elif kind in {"heal", "heal_full"}:
                amount = maximum if kind == "heal_full" else n
                useful = 0 if "markofthebloom" in relics else min(maximum - hp - healing, amount)
                healing += useful
                value = useful * hp_weight * .5
            elif kind in {"lose_hp", "damage"}:
                loss += n
                value = -n * hp_weight
                gaps.add("event_hp_loss_relic_callbacks_not_modeled")
            elif kind in {"gain_gold", "gain_gold_range"}:
                if kind == "gain_gold_range":
                    n = int(op["minimum"])
                    gaps.add("gold_range_valued_at_public_minimum_not_expected_value")
                value = 0. if "ectoplasm" in relics else n / 25.
                if "ectoplasm" not in relics:
                    gold += n
            elif kind in {"lose_gold", "lose_all_gold"}:
                amount = gold if kind == "lose_all_gold" else min(gold, n)
                gold -= amount
                value = -amount / 25.
            elif kind == "gain_max_hp":
                value = n * .75
                gaps.add("max_hp_value_is_partial_not_a_health_simulation")
            elif kind == "lose_max_hp":
                max_loss += n
                value = -n * 1.25 - max(0, hp - (maximum - max_loss)) * hp_weight
            elif kind in {"add_curse", "random_curse"}:
                # Deliberately do not assume Omamori's charge from identity alone.
                value = -10. * n
                gaps.add("curse_and_prevention_value_is_conservative_prior")
            elif kind in {"random_relic", "obtain_relic"}:
                value = n * {"common": 5., "rare": 7., "boss": 8.}.get(op.get("rarity"), 5.)
                gaps.add("relic_value_is_generic_prior_not_item_or_synergy_value")
            elif kind == "random_potion":
                useful = min(slots, n)
                slots -= useful
                value = 1.5 * useful
                gaps.add("random_potion_prior_no_replacement_value")
            elif kind == "card_offer":
                value = 2. * n
                gaps.add("unrevealed_card_offer_generic_prior")
            elif kind in {"select_remove", "remove_set", "lose_card"}:
                candidates = [(i, self._removal(c)) for i, c in enumerate(deck)]
                if kind == "remove_set":
                    selector = op.get("selector")
                    candidates = [(i, v) for i, v in candidates if (
                        selector == "curses" and (self.catalog.lookup("card", deck[i].get("id")) or {}).get("type") == "CURSE"
                        or selector == "strikes" and potion_key(deck[i].get("id")) == "strikered")]
                    n = len(candidates)
                elif kind == "lose_card":
                    candidates = [(i, v) for i, v in candidates if deck[i].get("id") == op.get("id")]
                candidates = [(i, v) for i, v in candidates if v is not None]
                chosen = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
                value = sum(v for _, v in chosen)
                for i, _ in sorted(chosen, reverse=True):
                    deck.pop(i)
                gaps.add("removal_value_ignores_engine_and_relic_callbacks")
            elif kind in {"select_upgrade", "random_upgrade", "upgrade_set"}:
                candidates = [(i, self._upgrade(c, fraction, act)) for i, c in enumerate(deck)]
                if kind == "upgrade_set":
                    if op.get("selector") == "strikes_and_defends":
                        candidates = [(i, v) for i, v in candidates if potion_key(deck[i].get("id")) in {"strikered", "defendred"}]
                    n = len(candidates)
                candidates = [(i, v) for i, v in candidates if v is not None]
                if kind == "random_upgrade":
                    value = (sum(v for _, v in candidates) * min(n, len(candidates)) / len(candidates)
                             if candidates else 0.)
                    gaps.add("random_upgrade_uniform_eligible_card_prior")
                    # If another card-mutating clause follows, an identity would
                    # have to be invented. Refuse such a compound offer below.
                else:
                    chosen = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
                    value = sum(v for _, v in chosen)
                    for i, _ in chosen:
                        deck[i]["upgrades"] = 1
                gaps.add("upgrade_value_uses_partial_card_profile")
            if value is None:
                unsupported = True
                gaps.add("unvalued_event_effect:" + kind)
            else:
                score += value
            components.append({"kind": kind, "utility": value})
        mutations = [op["kind"] for op in contract["display_order"] if op["kind"] in {
            "select_remove", "remove_set", "lose_card", "select_upgrade", "random_upgrade", "upgrade_set"}]
        if "random_upgrade" in mutations and len(mutations) > 1:
            unsupported = True
            gaps.add("compound_random_deck_mutation_unvalued")
        if loss >= hp or max_loss >= maximum:
            score -= 1000.
            gaps.add("potentially_lethal_cost_no_assumed_heal_before_damage")
        return {"score": None if unsupported else score, "components": components, "gaps": sorted(gaps)}

    def choose(self, public, options):
        candidates = [{"action": a.as_dict(), "kind": row["kind"], **self._score(public, row)}
                      for a, row in zip(public.legal_actions, options)]
        valued = [i for i, row in enumerate(candidates) if row["score"] is not None]
        index = max(valued, key=lambda i: candidates[i]["score"]) if valued else 0
        action = public.legal_actions[index]
        gaps = {gap for row in candidates for gap in row["gaps"]}
        gaps.update({"event_display_parse_is_not_effect_fidelity_evidence",
                     "event_utilities_are_untrained_resource_heuristics"})
        self.last_decision = {"mode": "event_heuristic" if valued else "run_fallback",
                              "fallback": not bool(valued), "action": action.as_dict(),
                              "event_policy_version": EVENT_POLICY_VERSION,
                              "gaps": sorted(gaps), "candidates": candidates}
        return action
