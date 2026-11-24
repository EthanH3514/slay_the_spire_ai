"""Public top-card candidates and conservative automatic-effect admission.

This module never accepts native RNG, private card queues, or backend features.
The native rule uses a known top, otherwise a multiset mean for ordinary unknown
order, or minimum candidate heuristic utility for uncertain conditional order.
It never uses the first entry of an unordered pile. Rollouts execute real queues.
"""
from typing import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES, EffectPrimitive
from .mayhem import deferred_autoplay_reason
from .public_piles import public_draw_cards, opening_priority_ids


def has_top_autoplay(card: Mapping) -> bool:
    recipe = DEFAULT_CARD_RECIPES.lookup(card.get("id"))
    return bool(recipe and any(s.primitive is EffectPrimitive.AUTOPLAY_TOP_CARD
                               for s in recipe.effects))


def top_card_candidates(combat: Mapping) -> tuple[Mapping, ...]:
    opening = set(opening_priority_ids(combat))
    top = combat.get("draw_pile_known_top")
    if isinstance(top, Mapping):
        return (top,)
    # PolicyInput retains ordered draw_pile only when Frozen Eye makes it public.
    ordered = combat.get("draw_pile")
    if ordered:
        return (ordered[-1],)
    unknown = tuple(c for c in combat.get("draw_pile_unordered", ()) if isinstance(c, Mapping))
    if unknown:
        if opening and not combat.get("opening_order_uncertain"):
            return tuple(c for c in unknown if c.get("unique_id") in opening)
        return unknown
    bottom = combat.get("draw_pile_known_bottom", ())
    if bottom:
        return (bottom[-1],)
    return tuple(c for c in combat.get("discard_pile", ()) if isinstance(c, Mapping))


def autoplay_support_reason(combat: Mapping, *, played: Mapping | None = None) -> str | None:
    """Do not let automatic effects bypass the ordinary legal-action evidence gate.

    Conservatively check both piles even with a known top: nested Havoc, draws and
    reshuffles can reach deeper cards before another PLAYER_NORMAL decision.
    Unreviewed statuses/curses remain gated too; the native lifecycle tests are not
    a blanket admission of all their passive/relic interactions.
    """
    from .mechanism_evidence import DEFAULT_MECHANISM_EVIDENCE, EvidenceCapability
    reason = deferred_autoplay_reason(combat)
    if reason is not None:
        return reason
    active = (played,) if played is not None else combat.get("hand", ())
    if not any(isinstance(c, Mapping) and has_top_autoplay(c) for c in active):
        return None
    cards = public_draw_cards(combat) + tuple(combat.get("discard_pile", ()))
    for card in cards:
        if not isinstance(card, Mapping) or not DEFAULT_MECHANISM_EVIDENCE.supports(
            "card", card.get("id"), EvidenceCapability.SEARCH, imported=True,
        ):
            identifier = card.get("id") if isinstance(card, Mapping) else "<malformed>"
            return f"top-card autoplay candidate lacks scoped evidence: {identifier!r}"
    return None
