"""Public-only persistent autoplay admission. No latent queues, order or RNG."""
from collections import deque
from typing import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES, RandomHandGeneration
from .identifiers import canonical_identifier
from .public_piles import public_draw_cards

def mayhem_stacks(combat: Mapping) -> int:
    player = combat.get("player", {})
    powers = player.get("powers", ()) if isinstance(player, Mapping) else ()
    if not isinstance(powers, (list, tuple)):
        raise ValueError("Mayhem requires a structured public power list")
    matches = [p for p in powers if isinstance(p, Mapping)
               and canonical_identifier(p.get("id"), "power") == "mayhem"]
    if not matches:
        return 0
    if len(matches) != 1 or type(matches[0].get("amount")) is not int or not 1 <= matches[0]["amount"] <= 2**31 - 1:
        raise ValueError("Mayhem requires one positive public native-integer stack record")
    return matches[0]["amount"]


def future_autoplay_cards(combat: Mapping, *, include_exhaust: bool = True) -> tuple[Mapping, ...]:
    """All public copies, not the current top or an assumed next-turn ordering."""
    return (public_draw_cards(combat) + tuple(combat.get("hand", ()))
            + tuple(combat.get("discard_pile", ()))
            + (tuple(combat.get("exhaust_pile", ())) if include_exhaust else ()))


def deferred_autoplay_reason(combat: Mapping) -> str | None:
    from .mechanism_evidence import DEFAULT_MECHANISM_EVIDENCE, EvidenceCapability
    from .random_generation import generation_pool
    from .rule_catalog import RuleCatalog
    try:
        stacks = mayhem_stacks(combat)
    except ValueError as error:
        return str(error)
    try:
        cards = future_autoplay_cards(combat)
    except TypeError:
        return "Mayhem requires structured public candidate piles"
    copies = sum(isinstance(c, Mapping) and canonical_identifier(c.get("id"), "card") == "mayhem" for c in cards)
    if not stacks and not copies:
        return None
    # Do not silently drop malformed draw entries in public_draw_cards().
    if "draw_pile_known_top" in combat and not isinstance(combat["draw_pile_known_top"], Mapping):
        return "Mayhem requires a structured public known top"
    for name in ("hand", "draw_pile", "draw_pile_unordered", "draw_pile_known_bottom", "discard_pile", "exhaust_pile"):
        values = combat.get(name, ())
        if not isinstance(values, (list, tuple)) or any(not isinstance(c, Mapping) for c in values):
            return "Mayhem requires structured public candidate piles"
    catalog = RuleCatalog.bundled()
    pending = deque(cards)
    seen, contracts = set(), set()
    powers = combat.get("player", {}).get("powers", ())
    if any(isinstance(p, Mapping) and canonical_identifier(p.get("id"), "power") == "magnetism" for p in powers):
        contracts.add(RandomHandGeneration.COLORLESS_TURN_START)
        pending.extend(generation_pool(RandomHandGeneration.COLORLESS_TURN_START, catalog))
    while pending:
        card = pending.popleft()
        key = canonical_identifier(card.get("id"), "card") if isinstance(card, Mapping) else None
        if key in seen:
            continue
        seen.add(key)
        if not DEFAULT_MECHANISM_EVIDENCE.supports("card", key, EvidenceCapability.SEARCH, imported=True):
            return f"Mayhem future autoplay candidate lacks scoped evidence: {card.get('id') if isinstance(card, Mapping) else '<malformed>'!r}"
        recipe = DEFAULT_CARD_RECIPES.lookup(key)
        if any(step.delayed_damage is not None for step in recipe.effects):
            from .bombs import bomb_scope_reason
            prospective = dict(combat, hand=tuple(combat.get("hand", ())) + (card,))
            if reason := bomb_scope_reason(prospective):
                return f"Mayhem future autoplay candidate {card.get('id')!r}: {reason}"
        for step in recipe.effects:
            if step.generated_card_id:
                pending.append({"id": step.generated_card_id})
            pools = [step.random_hand_generation, step.random_draw_generation, step.discovery_generation]
            if step.turn_trigger:
                pools.extend(e.random_hand_generation for e in step.turn_trigger.effects)
            for contract in pools:
                if contract is not None and contract not in contracts:
                    contracts.add(contract)
                    pending.extend(generation_pool(contract, catalog))
    return None
