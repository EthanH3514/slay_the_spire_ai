"""Public, immediate potion replacement plans and checked commitments.

No predicted drops, seeds or latent outcomes can authorize discarding a bottle.
A plan names both the discarded slot and a currently executable acquisition.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import copy

from .actions import Action, ActionKind
from .identifiers import canonical_identifier
from .policy_api import PolicyInput

VERSION = 'immediate-potion-replacement-v1'


def key(value):
    return canonical_identifier(value, 'potion')


@dataclass(frozen=True)
class Replacement:
    fingerprint: str
    discard: Action
    acquire: Action
    source: str
    item_id: str
    price: int = 0

    def as_dict(self):
        return dict(version=VERSION, fingerprint=self.fingerprint, discard=self.discard.as_dict(),
                    acquire=self.acquire.as_dict(), source=self.source, item_id=self.item_id, price=self.price)


def plans(public):
    inventory = public.observation.get('inventory', {})
    slots = inventory.get('potions')
    if not isinstance(slots, (list, tuple)) or not slots:
        raise ValueError('replacement requires public potion slots')
    if any(not isinstance(p, Mapping) or not isinstance(p.get('id'), str) or not p['id'] for p in slots):
        raise ValueError('replacement requires identified potion slots')
    if any(key(p['id']) == 'empty_potion_slot' for p in slots): return ()
    if any(canonical_identifier(r['id'], 'relic') == 'sozu' for r in inventory.get('relics', ())): return ()
    sources = []
    combat = public.observation.get('combat')
    if combat is not None:
        for a in public.legal_actions:
            if a.kind == ActionKind.USE_POTION and key(slots[a.potion_index]['id']) == 'entropic_brew':
                sources.append((a, 'refill', slots[a.potion_index]['id'], 0))
        sources.sort(key=lambda s:s[0].potion_index)
    else:
        screen = public.observation.get('screen', {})
        screen_type = screen.get('type')
        for o in screen.get('decision_options', ()):
            kind = o.get('kind')
            if not ((screen_type == 'SHOP' and kind == 'buy_potion') or
                    (screen_type == 'COMBAT_REWARD' and kind == 'potion')): continue
            action = Action.from_dict(o['action'])
            if action not in public.legal_actions: continue
            price = o.get('price', 0)
            if type(price) is not int or price < 0: raise ValueError('invalid public potion price')
            if price > public.observation.get('player', {}).get('gold', 0): continue
            if not isinstance(o.get('item_id'), str) or not o['item_id']: raise ValueError('unknown offered potion')
            sources.append((action, kind, o['item_id'], price))
    result = []
    for slot in range(len(slots)):
        for action, source, item, price in sources:
            if source == 'refill' and action.potion_index == slot: continue
            result.append(Replacement(public.fingerprint(), Action(ActionKind.DISCARD_POTION,potion_index=slot),
                                      action,source,item,price))
    return tuple(result)


def constrain(public):
    if not any(a.kind == ActionKind.DISCARD_POTION for a in public.legal_actions): return public
    allowed = {p.discard for p in plans(public)}
    choices = tuple(a for a in public.legal_actions if a.kind != ActionKind.DISCARD_POTION or a in allowed)
    if not choices: raise ValueError('replacement constraint removed all actions')
    return public.with_legal_actions(choices)


def combat_plan(public, discard):
    choices = [p for p in plans(public) if p.discard == discard and p.source == 'refill']
    if not choices: raise ValueError('discard has no immediate replacement')
    return choices[0]


@dataclass
class Commitment:
    plan: Replacement
    before: dict

    @classmethod
    def bind(cls, public, plan):
        if plan not in plans(public): raise ValueError('stale or unavailable replacement')
        return cls(plan, copy.deepcopy(public.as_dict()['observation']))

    def followup(self, public):
        """Only the declared discard may occur between selection and acquisition."""
        after = public.as_dict()['observation']
        expected = {k:v for k,v in self.before.items() if k != 'screen'}
        actual = {k:v for k,v in after.items() if k != 'screen'}
        expected = dict(expected, inventory=dict(expected['inventory']))
        pots = [dict(p) for p in expected['inventory']['potions']]
        pots[self.plan.discard.potion_index] = dict(id='EMPTY_POTION_SLOT')
        expected['inventory']['potions'] = pots
        if actual != expected or after.get('screen',{}).get('type') != self.before.get('screen',{}).get('type'):
            raise ValueError('replacement intervening state changed')
        if self.plan.source == 'refill':
            a = self.plan.acquire
            if a not in public.legal_actions or key(pots[a.potion_index]['id']) != key(self.plan.item_id):
                raise ValueError('replacement generator unavailable')
            return a
        # Rebind the actual current public offer; never reuse a potentially stale
        # choice index if the menu changed after discarding.
        matches = [o for o in after.get('screen',{}).get('decision_options',())
                   if o.get('kind') == self.plan.source and key(o.get('item_id')) == key(self.plan.item_id)
                   and o.get('price',0) == self.plan.price and Action.from_dict(o['action']) in public.legal_actions]
        exact = [o for o in matches if Action.from_dict(o['action']) == self.plan.acquire]
        if len(exact) == 1: return Action.from_dict(exact[0]['action'])
        if len(matches) != 1: raise ValueError('replacement offer missing or ambiguous')
        return Action.from_dict(matches[0]['action'])
