"""Typed *display contracts* from the pinned native event-choice grammar.

These describe offered costs/effects, never the secretly rolled outcome. Parsing
the whole displayed offer does not certify original-game or simulator fidelity.
Unknown clauses are retained and prevent a 'parsed' result; no fuzzy NLP scoring.
"""
from __future__ import annotations

from functools import lru_cache
import re

from .rule_catalog import RuleCatalog


EVENT_OPTIONS_VERSION = "public-event-display-v1"


def effect(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}


NEOW_BONUSES = {
    "Choose a card to obtain.": effect("card_offer", pool="class", count=1, rarity="unspecified"),
    "Obtain a random rare card.": effect("random_card", pool="class", count=1, rarity="rare"),
    "Remove a card.": effect("select_remove", count=1),
    "Upgrade a card.": effect("select_upgrade", count=1),
    "Transform a card.": effect("select_transform", count=1),
    "Choose a colorless card to obtain.": effect("card_offer", pool="colorless", count=1, rarity="unspecified"),
    "Obtain three potions.": effect("random_potion", count=3),
    "Obtain a random common relic.": effect("random_relic", rarity="common", count=1),
    "Max Hp +10%.": effect("gain_max_hp_fraction", numerator=10, denominator=100, rounding="unspecified"),
    "Obtain Neow's Lament.": effect("obtain_relic", id="NEOWS_LAMENT", count=1),
    "Obtain 100 gold.": effect("gain_gold", amount=100),
    "Choose a rare colorless card to obtain.": effect("card_offer", pool="colorless", count=1, rarity="rare"),
    "Remove two cards.": effect("select_remove", count=2),
    "Obtain a random rare relic.": effect("random_relic", rarity="rare", count=1),
    "Choose a rare card to obtain.": effect("card_offer", pool="class", count=1, rarity="rare"),
    "Obtain 250 gold.": effect("gain_gold", amount=250),
    "Transform two cards in your cards.": effect("select_transform", count=2),
    "Max Hp +20%.": effect("gain_max_hp_fraction", numerator=20, denominator=100, rounding="unspecified"),
    "Obtain a random boss relic.": effect("random_relic", rarity="boss", count=1),
}
NEOW_DRAWBACKS = {
    "": None,
    "Max Hp -10%.": effect("lose_max_hp_fraction", numerator=10, denominator=100, rounding="unspecified"),
    "Lose all gold.": effect("lose_all_gold"),
    "Obtain a curse.": effect("random_curse", count=1),
    "Take 30% Hp damage.": effect("damage_fraction", numerator=30, denominator=100,
                                   hp_basis="unspecified_in_native_label", rounding="unspecified"),
    "Lose your starter relic.": effect("lose_starter_relic"),
}


@lru_cache(maxsize=1)
def _named_items() -> dict:
    names = {}
    for kind in ("card", "relic", "potion"):
        for entry in RuleCatalog.bundled().entries(kind):
            for name in (entry.get("name"), entry.get("game_id"), entry["id"]):
                if name:
                    names.setdefault((kind, name), set()).add(entry["id"])
    return names


def _named(name: str, kind: str) -> str | None:
    aliases = {"J.A.X": "JAX", "Bites": "Bite", "Strikes": "Strike"}
    ids = _named_items().get((kind, aliases.get(name, name)), set())
    return next(iter(ids)) if len(ids) == 1 else None


def _clause(text: str) -> list[dict] | None:
    fixed = {
        "Nothing happens": [effect("leave")], "Heal to full HP": [effect("heal_full")],
        "Lose all Gold": [effect("lose_all_gold")], "Lose ALL of your Gold": [effect("lose_all_gold")],
        "Remove all Curses from your deck": [effect("remove_set", selector="curses")],
        "Remove all Strikes": [effect("remove_set", selector="strikes")],
        "Upgrade all Strikes and Defends": [effect("upgrade_set", selector="strikes_and_defends")],
        "Upgrade all Cards": [effect("upgrade_set", selector="all_upgradeable")],
        "You can no longer heal": [effect("disable_healing")],
        "Duplicate a card in your deck": [effect("select_duplicate", count=1)],
        "Immediately travel to the boss": [effect("travel_to_boss")],
        "Escape": [effect("leave")], "End the search and resume your journey": [effect("leave")],
        "Fight": [effect("begin_combat", encounter="unrevealed")],
        "Fight a Boss from Act 1": [effect("begin_combat", encounter="unrevealed_act1_boss")],
        "Anger the Mushrooms": [effect("begin_combat", encounter="unrevealed")],
        "Trigger a trap": [effect("continue_event", next_offer="unrevealed")],
        "Receive a Relic": [effect("random_relic", count=1, rarity="unspecified")],
        "Obtain a relic": [effect("random_relic", count=1, rarity="unspecified")],
        "Obtain a random Relic": [effect("random_relic", count=1, rarity="unspecified")],
        "Obtain a Rare Relic": [effect("random_relic", count=1, rarity="rare")],
        "Reward: Rare Relic": [effect("combat_reward_relic", count=1, rarity="rare")],
        "Get a Potion": [effect("random_potion", count=1)],
        "Get a Colorless Card": [effect("random_card", count=1, pool="colorless", rarity="unspecified")],
        "Choose 1 of 20 cards to add to your deck": [effect("card_offer", count=1, offer_size=20, pool="class")],
        "Store a card in your deck in return": [effect("select_store", count=1)],
        "Find Loot": [effect("search_loot", outcome="unrevealed")],
        "Spin the wheel and get a prize": [effect("random_event_outcome", distribution="unspecified")],
        "Obtain the Book": [effect("random_relic", pool="cursed_tome_books", count=1)],
    }
    if text in fixed:
        return fixed[text]
    for pattern, kind in (
        (r"Heal (\d+)\s*HP", "heal"), (r"Lose (\d+)\s*HP", "lose_hp"),
        (r"Take (\d+)\s*damage", "damage"), (r"Lose (\d+) Max H[Pp]", "lose_max_hp"),
        (r"Gain (\d+) Max HP", "gain_max_hp"), (r"Max HP \+(\d+)", "gain_max_hp"),
        (r"(?:Gain|Obtain|Receive) (\d+) Gold", "gain_gold"),
        (r"Lose (\d+) (?:Gold|gold)", "lose_gold"), (r"Lose (\d+)g", "lose_gold"),
    ):
        if match := re.fullmatch(pattern, text):
            return [effect(kind, amount=int(match[1]))]
    if match := re.fullmatch(r"Gain (\d+)-(\d+) Gold", text):
        if int(match[1]) <= int(match[2]):
            return [effect("gain_gold_range", minimum=int(match[1]), maximum=int(match[2]), distribution="unspecified")]
    if match := re.fullmatch(r"(Remove|Upgrade|Transform) a card(?: from your (?:deck|cards)| in your cards)?", text):
        return [effect("select_" + match[1].lower(), count=1)]
    if match := re.fullmatch(r"(Upgrade|Transform) (\d+) random cards", text):
        return [effect("random_" + match[1].lower(), count=int(match[2]))]
    if match := re.fullmatch(r"Choose and Transform (\d+) cards in your deck", text):
        return [effect("select_transform", count=int(match[1]))]
    if match := re.fullmatch(r"(?:Receive|Obtain|Become Cursed\s*[-:])\s*(?:(\d+) )?(.+)", text):
        count, name = int(match[1] or 1), match[2]
        for kind in ("card", "relic"):
            identifier = _named(name.removeprefix("a "), kind)
            if identifier:
                operation = "add_curse" if text.startswith("Become Cursed") else "obtain_" + kind
                return [effect(operation, id=identifier, count=count)]
    if match := re.fullmatch(r"Add (\d+) Colorless Card to your deck", text):
        return [effect("card_offer", count=int(match[1]), pool="colorless", rarity="unspecified")]
    if match := re.fullmatch(r"Lose (.+)", text):
        for kind in ("card", "relic", "potion"):
            if identifier := _named(match[1], kind):
                return [effect("lose_" + kind, id=identifier, count=1)]
    if match := re.fullmatch(r"(\d+)%:? (?:Find a Relic|Become Cursed - (.+))", text):
        if 0 <= int(match[1]) <= 100:
            outcome = effect("random_relic", count=1, rarity="unspecified")
            if match[2]:
                identifier = _named(match[2], "card")
                if not identifier:
                    return None
                outcome = effect("add_curse", id=identifier, count=1)
            return [effect("displayed_chance", numerator=int(match[1]), denominator=100,
                           outcome=outcome, other_outcomes="unspecified")]
    if match := re.fullmatch(r"Bet (\d+) Gold - (\d+)%: win (\d+) Gold", text):
        if 0 <= int(match[2]) <= 100:
            return [effect("lose_gold", amount=int(match[1])), effect("displayed_chance",
                numerator=int(match[2]), denominator=100, outcome=effect("gain_gold", amount=int(match[3])),
                other_outcomes="unspecified")]
    if match := re.fullmatch(r"(\d+)% that an Elite will return to fight you", text):
        if 0 <= int(match[1]) <= 100:
            return [effect("displayed_chance", numerator=int(match[1]), denominator=100,
                outcome=effect("begin_combat", encounter="unrevealed_elite"), other_outcomes="unspecified")]
    return None


def event_choice(event_name: str | None, label: str) -> dict:
    """No actual random identity, future screen, or native phase integer is read."""
    operations, unknown = [], []
    if not event_name:
        unknown.append("missing_event_identity")
    elif event_name == "NEOW":
        parts = label.split("\t")
        bonus = NEOW_BONUSES.get(parts[0].strip())
        drawback = parts[1].strip() if len(parts) == 2 else ""
        if bonus is None or len(parts) > 2 or drawback not in NEOW_DRAWBACKS:
            unknown.append(label)
        else:
            # Display order only; a separate fidelity audit must check execution
            # order/rounding and all relic callbacks before this becomes a model.
            operations.append(dict(bonus))
            if NEOW_DRAWBACKS[drawback] is not None:
                operations.append(dict(NEOW_DRAWBACKS[drawback]))
    elif match := re.fullmatch(r"\[([^\]]+)\]\s*(.*)", label):
        option, body = match[1], match[2]
        if not body and option == "Leave":
            operations.append(effect("leave"))
        elif not body and option in {"Fight", "Fight!"}:
            operations.append(effect("begin_combat", encounter="unrevealed"))
        elif not body and option == "Read" and event_name == "Cursed Tome":
            operations.append(effect("continue_event", next_offer="unrevealed"))
        elif event_name == "The Woman in Blue" and (buy := re.fullmatch(r"Buy (\d+) Potion", option)) and (
                cost := re.fullmatch(r"(\d+) Gold\.", body)):
            operations.extend([effect("lose_gold", amount=int(cost[1])), effect("random_potion", count=int(buy[1]))])
        else:
            # Split only known punctuation boundaries. Internal dots in J.A.X.
            # remain intact, and every unrecognized remainder is reported.
            body = body.replace(", Become Cursed", ". Become Cursed")
            clauses = re.split(r"\.(?:\s+|$)", body)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                parsed = _clause(clause)
                if parsed is None:
                    unknown.append(clause)
                else:
                    operations.extend(parsed)
            if not body:
                unknown.append("undocumented_option_effect")
    else:
        unknown.append(label)
    if (event_name == "We Meet Again!" and not unknown and operations
            and all(op["kind"] in {"lose_potion", "lose_card", "lose_gold"} for op in operations)):
        # The printer omits this non-random rule of the donation: the item
        # identity remains unknown. Source: GameContext::chooseEventOption.
        operations.append(effect("random_relic", rarity="unspecified", count=1,
                                 source="reviewed_native_donation_rule"))
    is_cost = lambda op: op["kind"].startswith("lose_") or op["kind"] in {
        "damage", "damage_fraction", "add_curse", "random_curse", "disable_healing"}
    return {"version": EVENT_OPTIONS_VERSION, "event_name": event_name,
            "status": "parsed" if operations and not unknown else "partial" if operations else "unknown",
            "fidelity": "display_contract_not_effect_certification",
            "display_order": operations, "costs": [op for op in operations if is_cost(op)],
            "effects": [op for op in operations if not is_cost(op)], "unknown_clauses": unknown}
