"""Reviewed public decision rules, separate from legality and learned value.

Used by the opt-in event planner; frozen historical controllers stay unchanged.
"""
from collections.abc import Mapping

from .identifiers import canonical_identifier
from .run_options import public_run_options

VERSION = 'free-potion-and-ignore-note-v1'
LEAVE_ONLY_EVENTS = frozenset({'Note For Yourself'})


def _free_leave(row):
    contract = row.get('event_choice')
    if contract:
        order = contract.get('display_order', ())
        return (contract.get('status') == 'parsed' and not contract.get('unknown_clauses')
                and len(order) == 1 and order[0].get('kind') == 'leave')
    return row.get('kind') == 'leave'


def eligible_actions(public, actions, *, event_name=None):
    """Keep public action IDs/metadata intact; constrain planner choices only.

    Free rewards are already offered bottles, not events charging HP/gold.
    Keep every bottle choice; block only departure while slots/offers remain.
    """
    screen = public.observation.get('screen', {})
    if event_name in LEAVE_ONLY_EVENTS:
        if screen.get('type') != 'EVENT':
            raise ValueError('leave-only event requires its initial decision menu')
        rows = dict(zip(public.legal_actions, public_run_options(public), strict=True))
        leaves = tuple(a for a in actions if _free_leave(rows[a]))
        if len(leaves) != 1:
            raise ValueError('leave-only event has no unique verified free exit')
        return leaves
    if screen.get('type') != 'COMBAT_REWARD':
        return tuple(actions)
    inventory = public.observation.get('inventory', {})
    slots = inventory.get('potions', ())
    if not any(isinstance(p, Mapping) and isinstance(p.get('id'), str)
               and canonical_identifier(p['id'], 'potion') == 'empty_potion_slot' for p in slots):
        return tuple(actions)
    if any(canonical_identifier(r.get('id'), 'relic') == 'sozu' for r in inventory.get('relics', ())):
        return tuple(actions)
    rows = dict(zip(public.legal_actions, public_run_options(public), strict=True))
    free = []
    for a in actions:
        row = rows[a]
        price = row.get('price', 0)
        if (row['kind'] == 'potion' and isinstance(row.get('item_id'), str) and row['item_id']
                and type(price) is int and price == 0
                and not any(row.get(k) for k in ('cost', 'hp_cost', 'effects', 'event_choice'))):
            free.append(a)
    if not free:
        return tuple(actions)
    return tuple(a for a in actions if not _free_leave(rows[a]))
