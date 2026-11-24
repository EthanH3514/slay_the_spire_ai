"""Public eligibility for selected retrieval; actions move, never draw/copy."""
from typing import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES, EffectStep, SelectedCardSource, SelectedCardFilter
from .identifiers import canonical_identifier
from .public_piles import public_draw_cards
from .rule_catalog import RuleCatalog

RETRIEVAL_TASKS = frozenset({"EXHUME", "SECRET_TECHNIQUE", "SECRET_WEAPON"})


def retrieval_step(task: str) -> EffectStep:
    if task not in RETRIEVAL_TASKS:
        raise ValueError("unreviewed retrieval task")
    return DEFAULT_CARD_RECIPES.lookup(task).effects[0]


def eligible(card: Mapping, step: EffectStep, catalog: RuleCatalog) -> bool:
    if step.selected_card_filter is SelectedCardFilter.NON_EXHUME:
        return canonical_identifier(card.get("id"), "card") != "exhume"
    entry = catalog.lookup("card", card.get("id")) or {}
    card_type = str(card.get("type", entry.get("type", ""))).upper()
    return card_type == ("SKILL" if step.selected_card_filter is SelectedCardFilter.SKILL else "ATTACK")


def retrieval_candidates(step: EffectStep, combat: Mapping, catalog: RuleCatalog) -> tuple[Mapping, ...]:
    cards = (public_draw_cards(combat) if step.selected_card_source is SelectedCardSource.DRAW_PILE
             else combat.get("exhaust_pile", ()))
    return tuple(c for c in cards if isinstance(c, Mapping) and eligible(c, step, catalog))
