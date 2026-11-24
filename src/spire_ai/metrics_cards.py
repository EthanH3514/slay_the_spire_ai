"""Historical metrics intake for win-only, partial-state card imitation.

Outcome is an intake filter, never a per-action reward or a network feature.
This adapter does not certify human skill, no-SL play or historical rules parity.
"""
from __future__ import annotations

import copy

from .expert_cards import HistoryGap, NO_CARD, card, integer, reconstruct
from .rule_catalog import RuleCatalog

VERSION = "metrics-win-card-prefix-v2"
# This is a schema compatibility set, NOT a claim of current-game rules parity.
LEGACY_BUILDS = frozenset({"2020-07-30"})
EMPTY_INVENTORY_BONUSES = frozenset({
    "HUNDRED_GOLD", "TWO_FIFTY_GOLD", "TEN_PERCENT_HP_BONUS",
    "TWENTY_PERCENT_HP_BONUS", "THREE_SMALL_POTIONS",
})
CHOICE_BONUSES = frozenset({
    "THREE_CARDS", "THREE_RARE_CARDS", "RANDOM_COLORLESS", "RANDOM_COLORLESS_2",
})
INVENTORY_SAFE_COSTS = frozenset({"NONE", "TEN_PERCENT_HP_LOSS", "NO_GOLD", "PERCENT_DAMAGE"})


def cohort(run):
    """Separate wins and false outcomes before reconstructing any policy rows."""
    if run.get("character_chosen") != "IRONCLAD":
        return "exclude", "wrong_character"
    if type(run.get("ascension_level")) is not int or run["ascension_level"] != 20:
        return "exclude", "not_exact_a20"
    for flag in ("is_daily", "is_endless", "is_trial", "is_beta", "chose_seed"):
        if run.get(flag) is not False:
            return "exclude", "unsupported_or_unknown_flag:" + flag
    if run.get("special_seed"):
        return "exclude", "special_seed"
    if "is_modded" in run and run["is_modded"] is not False:
        return "exclude", "modded_or_unknown_flag"
    if type(run.get("victory")) is not bool:
        return "exclude", "unknown_outcome"
    # False may mean abandonment or incomplete reporting, not a combat death.
    if not run["victory"]:
        return "loss_retained_only", "not_a_behavior_label"
    return "win_candidate", "victory_true_not_expert_or_no_sl_certification"


def adapt_neow(run, *, catalog=None):
    """Recover only explicit/deterministic starting inventory, never final cards.

NeowReward.activate in the local vanilla bytecode supports the inventory-only
semantics below. HP, money and potions remain unknown at each later decision.
The original log is left unchanged. Missing random outcomes are refusals.
"""
    catalog = catalog or RuleCatalog.bundled()
    adapted = copy.deepcopy(run)
    if "neow_bonus_log" in run:
        # The existing importer validates the detailed schema, including nulls.
        return adapted, "explicit_neow_detail"
    if run.get("build_version") not in LEGACY_BUILDS:
        raise HistoryGap("unreviewed_legacy_build")
    cost, bonus = run.get("neow_cost"), run.get("neow_bonus")
    if cost not in INVENTORY_SAFE_COSTS:
        raise HistoryGap("unobserved_neow_cost_inventory:" + str(cost))
    if not isinstance(run.get("card_choices"), list):
        raise HistoryGap("missing_card_choices")
    zero_choices = [c for c in run["card_choices"] if integer(c["floor"]) == 0]
    if bonus in CHOICE_BONUSES:
        if len(zero_choices) != 1:
            raise HistoryGap("missing_or_ambiguous_neow_card_choice")
        choice = zero_choices[0]
        if not isinstance(choice.get("not_picked"), list):
            raise HistoryGap("invalid_neow_offer")
        picked = choice["picked"]
        if picked == "Singing Bowl":
            raise HistoryGap("unexpected_neow_bowl")
        offered = list(choice["not_picked"]) + ([] if picked in NO_CARD else [picked])
        parsed = [card(c, catalog) for c in offered]
        if len(parsed) != 3 or len({(c["id"], c["upgrades"]) for c in parsed}) != 3:
            raise HistoryGap("invalid_neow_offer")
        if any(c["upgrades"] for c in parsed):
            raise HistoryGap("unexpected_neow_upgrade")
        # Floor-zero acquisition is replayed ONCE by reconstruct. Do not also
        # place it in cardsObtained, and do not train its unverified action UI.
    elif zero_choices:
        raise HistoryGap("unexpected_neow_card_choice")
    elif bonus not in EMPTY_INVENTORY_BONUSES and bonus != "THREE_ENEMY_KILL":
        raise HistoryGap("unobserved_neow_bonus_inventory:" + str(bonus))
    # Other zero-floor mutation channels have no verified cross-table order.
    for table in ("event_choices", "campfire_choices", "relics_obtained"):
        if any(integer(row["floor"]) == 0 for row in run.get(table, [])):
            raise HistoryGap("unexpected_neow_mutation_table:" + table)
    for table in ("item_purchase_floors", "items_purged_floors"):
        if any(integer(floor) == 0 for floor in run.get(table, [])):
            raise HistoryGap("unexpected_neow_mutation_table:" + table)
    adapted["neow_bonus_log"] = {
        "cardsRemoved": [], "cardsTransformed": [], "cardsUpgraded": [],
        "cardsObtained": [],
        "relicsObtained": ["NeowsLament"] if bonus == "THREE_ENEMY_KILL" else [],
    }
    return adapted, "legacy_inventory_only:" + bonus


def reconstruct_win(run, *, catalog=None):
    category, reason = cohort(run)
    if category != "win_candidate":
        raise HistoryGap("not_win_bc_cohort:" + reason)
    adapted, neow_mode = adapt_neow(run, catalog=catalog)
    # Verified vanilla Falling/Beggar call AbstractEvent.logMetricCardRemoval
    # /AtCost, which stores AbstractCard.cardID (no upgrade suffix). This is
    # opt-in for the legacy format, leaving earlier expert artifacts unchanged.
    base_ids = ("Falling", "Beggar") if "neow_bonus_log" not in run else ()
    # Writhing Mass's Implant adds Parasite to the permanent deck, not the
    # card-choice log. Combat summaries identify the encounter, NOT whether it
    # used Implant. Taint every subsequent prefix, including that floor's
    # reward, even if the final deck has no Parasite. Never backfill the curse.
    inventory_barriers = []
    combats = run.get("damage_taken")
    if combats is None:
        if any(integer(choice["floor"]) >= 35 for choice in run["card_choices"]):
            inventory_barriers.append({"floor": 35, "reason": "missing_act3_combat_inventory_history"})
    elif not isinstance(combats, list):
        raise HistoryGap("invalid_combat_inventory_history")
    else:
        for combat in combats:
            if not isinstance(combat, dict) or not isinstance(combat.get("enemies"), str):
                raise HistoryGap("invalid_combat_inventory_history")
            floor = integer(combat["floor"])
            if combat["enemies"] == "Writhing Mass":
                inventory_barriers.append({"floor": floor, "reason": "unlogged_combat_deck_effect:WRITHING_MASS"})
    result = reconstruct(adapted, catalog=catalog, base_id_removal_events=base_ids,
                         inventory_barriers=inventory_barriers)
    return {**result, "intake_version": VERSION, "neow_mode": neow_mode}
