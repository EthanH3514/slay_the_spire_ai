"""Public resident-card contracts, independent of hidden backend state."""
from collections.abc import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES, ResidentRuleKind
from .card_recipes import SelfCopyMode
from .public_piles import public_draw_cards


def public_turn_play_count(combat: Mapping) -> int:
    if type(combat.get("turn_play_count_schema")) is not int or combat["turn_play_count_schema"] != 1:
        raise ValueError("Normality requires public turn_play_count_schema=1")
    count = combat.get("cards_played_this_turn")
    if type(count) is not int or count < 0:
        raise ValueError("Normality requires nonnegative public cards_played_this_turn")
    return count


def resident_hp_loss_packets(hand, played_index, recipes=DEFAULT_CARD_RECIPES) -> tuple[int, ...]:
    packets = []
    for index, card in enumerate(hand):
        if index == played_index or not isinstance(card, Mapping):
            continue
        recipe = recipes.lookup(card.get("id"))
        rule = recipe.resident_rule if recipe is not None else None
        if rule is not None and rule.kind is ResidentRuleKind.OTHER_PLAY_HP_LOSS:
            packets.append(rule.amount)
    return tuple(packets)


def hand_resident_curse_reason(combat: Mapping, *, require_order_certainty: bool = True) -> str | None:
    if require_order_certainty and combat.get("top_order_uncertain", False) is not False:
        return "public top order after random insertion is not supported"
    # Presence in another pile still matters: a rollout can draw/retrieve it.
    # Do not infer Normality's counter from manual actions or audit internals.
    for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        for card in public_draw_cards(combat) if pile == "draw_pile" else combat.get(pile, ()):
            if not isinstance(card, Mapping):
                continue
            recipe = DEFAULT_CARD_RECIPES.lookup(card.get("id"))
            if recipe and recipe.self_copy and recipe.self_copy.mode is SelfCopyMode.STAT_DRAW_TOP:
                if type(combat.get("top_knowledge_schema")) is not int or combat["top_knowledge_schema"] != 1:
                    return "Pride requires tracked public top knowledge schema1"
                if (not isinstance(combat.get("public_known_top_ids"), (list, tuple))
                        or (require_order_certainty and combat.get("top_order_uncertain") is not False)):
                    return "Pride requires explicit public top IDs and certainty"
            rule = recipe.resident_rule if recipe is not None else None
            if rule is not None and rule.kind is ResidentRuleKind.TURN_PLAY_LIMIT:
                try:
                    public_turn_play_count(combat)
                except ValueError as error:
                    return str(error)
    return None
