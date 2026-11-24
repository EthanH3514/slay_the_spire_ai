"""Public selected-copy eligibility; native execution owns identity creation."""
from typing import Mapping

from .card_recipes import SelectedCardCopy
from .rule_catalog import RuleCatalog


def copy_eligible(card: Mapping, catalog: RuleCatalog, spec: SelectedCardCopy = SelectedCardCopy()) -> bool:
    entry = catalog.lookup("card", card.get("id")) or {}
    return str(card.get("type", entry.get("type", ""))).upper() in spec.card_types


def pending_copy_count(combat: Mapping) -> int:
    value = combat.get("selection_copy_count")
    if type(value) is not int or value not in (1, 2):
        raise ValueError("selected copy requires public copy count (1 or 2)")
    return value
