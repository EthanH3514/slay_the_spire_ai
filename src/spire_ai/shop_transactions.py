"""Opt-in public shop transactions and whole-visit consistency diagnostics.

This is an effects model, not a trained shop policy. A pending random event or
selection is a frontier, never a resolved afterstate or an invented reward.
Historical menus contain only affordable goods and are explicitly incomplete.
"""
from collections import Counter
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from .actions import Action
from .policy_api import PolicyInput, _freeze_public_value, _thaw_public_value, _reject_forbidden_policy_keys
from .run_afterstates import (PublicRunState, capture, local_transition, _catalog,
                             _digest, _int, _reject_raw_history, LocalEffectGap)
from .rule_coverage import _validate_input
from .run_learning_encoding import BOTTLES

VERSION = 'public-shop-transactions-v1'
SHELF_VERSION = 'public-shop-shelf-v1'
BUY_KINDS = ('buy_card', 'buy_relic', 'buy_potion')
SELECT_RELICS = {'BOTTLED_FLAME': 'bottle_attack', 'BOTTLED_LIGHTNING': 'bottle_skill',
                'BOTTLED_TORNADO': 'bottle_power', 'DOLLYS_MIRROR': 'duplicate_card'}
RANDOM_RELICS = {'WAR_PAINT': ('upgrade_skills', 2), 'WHETSTONE': ('upgrade_attacks', 2),
                'ORRERY': ('card_rewards', 5), 'CAULDRON': ('potion_rewards', 5)}
# Orrery's bytecode adds four rewards, then CombatRewardScreen.open adds the
# initial room card reward. Keep the reward count separate from its identities.
MAX_HP_RELICS = {'STRAWBERRY': 7, 'PEAR': 10, 'MANGO': 14, 'LEES_WAFFLE': 7}


def canonical(kind, identifier):
    if kind == 'potion' and identifier == 'EMPTY_POTION_SLOT':
        return identifier
    item = _catalog().lookup(kind, identifier)
    if item is None:
        raise LocalEffectGap('unknown_shop_'+kind+':'+str(identifier))
    return item['id']


def relic_ids(state):
    return {canonical('relic', r['id']) for r in state.run.public.observation['inventory']['relics']}


def normalize_offer(row, key):
    _reject_forbidden_policy_keys(row, reject_seeds=True)
    kind = row['kind']
    if kind not in BUY_KINDS:
        raise ValueError('not a shop good')
    result = dict(key=key, kind=kind, price=_int(row['price'], 'price'))
    if kind == 'buy_card':
        card = row['card']
        if set(card) != {'id','upgrades'}:raise ValueError('shop card identity fields')
        result['card'] = dict(id=canonical('card', card['id']), upgrades=_int(card['upgrades'], 'upgrades'))
    else:
        result['item_id'] = canonical('relic' if kind == 'buy_relic' else 'potion', row['item_id'])
    return _freeze_public_value(result)


def offer_identity(row):
    return (row['kind'], row['price'],
            (row['card']['id'], row['card']['upgrades']) if row['kind'] == 'buy_card' else row['item_id'])


@dataclass(frozen=True, slots=True)
class ShopState:
    run: PublicRunState
    offers: tuple
    removal_price: int | None
    complete_shelf: bool
    pending: tuple = ()
    closed: bool = False
    removal_services_used: int = 0

    def fingerprint(self):
        return _digest(dict(version=VERSION, run=self.run.fingerprint(),
            offers=_thaw_public_value(self.offers), removal_price=self.removal_price,
            complete_shelf=self.complete_shelf, pending=_thaw_public_value(self.pending), closed=self.closed,
            removal_services_used=self.removal_services_used))


@dataclass(frozen=True, slots=True)
class ShopChoice:
    key: str
    kind: str
    target_index: int | None = None
    # Only a root action observed in the actual current menu can be executed.
    action: Action | None = None


@dataclass(frozen=True, slots=True)
class ShopTransition:
    before: ShopState
    after: ShopState
    choice: ShopChoice
    paid: int
    effects: object


def capture_shop(public, *, run_public=None, shelf=None):
    _validate_input(public)
    if public.observation.get('screen', {}).get('type') != 'SHOP':
        raise ValueError('visible shop required')
    run = capture(public, relic_order_known=True, run_public=run_public)
    for name in ('hp','max_hp','gold'):
        if run.public.observation['player'].get(name) is None:
            raise LocalEffectGap('unknown_shop_'+name)
    options = public.observation['screen']['decision_options']
    actions = tuple(Action.from_dict(o['action']) for o in options)
    if len(set(actions)) != len(actions) or set(actions) != set(public.legal_actions):
        raise ValueError('shop option/action mismatch')
    removals = [o for o in options if o['kind'] == 'buy_remove']
    if len(removals) > 1:
        raise ValueError('ambiguous removal fee')
    if shelf is None:
        # Legal actions reveal neither all stock nor sold-out/merely unaffordable.
        offers = tuple(normalize_offer(o, 'visible:'+str(i)) for i,o in enumerate(options) if o['kind'] in BUY_KINDS)
        return ShopState(run, offers, _int(removals[0]['price'],'price') if removals else None, False)
    _reject_raw_history(shelf)
    _reject_forbidden_policy_keys(shelf, reject_seeds=True)
    # Reject unknown/hidden wrapper keys rather than silently ignoring them.
    if set(shelf) != {'version','available','complete','act','floor','items','removal_price'}:
        raise ValueError('public shelf fields')
    if (shelf['version'] != SHELF_VERSION or shelf['available'] is not True or shelf['complete'] is not True or
            any(shelf[k] != public.observation['run'][k] for k in ('act','floor'))):
        raise ValueError('public shelf identity')
    offers = []
    for row in shelf['items']:
        expected={'kind','slot','price','card' if row.get('kind')=='buy_card' else 'item_id'}
        if set(row)!=expected:raise ValueError('public shelf item fields')
        slot=_int(row['slot'],'slot',maximum=6 if row['kind']=='buy_card' else 2)
        offers.append(normalize_offer(row,row['kind']+':'+str(slot)))
    if len({o['key'] for o in offers}) != len(offers):raise ValueError('duplicate shelf slot')
    cost=shelf['removal_price']
    if cost is not None:cost=_int(cost,'removal_price')
    stock=Counter(offer_identity(o) for o in offers)
    visible=Counter(offer_identity(normalize_offer(o,'')) for o in options if o['kind'] in BUY_KINDS)
    if visible-stock or (removals and removals[0]['price']!=cost):
        raise ValueError('shelf disagrees with current legal menu')
    return ShopState(run,tuple(offers),cost,True)


def _derive(state, obs, *, info=None, **changes):
    obs['screen']={}
    p=capture(PolicyInput(obs,()),relic_order_known=True,
              run_public=state.run.run_public if info is None else info)
    return replace(state,run=replace(p,phase='local_effect_resolved'),**changes)


def _primitive(state,operation,**kwargs):
    # Shop protocol establishes availability; the shared resolver owns the
    # deck mutation, bottle joins, eggs and card acquisition/removal callbacks.
    t=local_transition(replace(state.run,phase='decision'),operation,**kwargs)
    return replace(state,run=t.after)


def _pending(state,kind,**detail):
    return replace(state,pending=state.pending+(_freeze_public_value(dict(kind=kind,**detail)),))


def _require_ready(state):
    if not isinstance(state,ShopState) or state.closed or state.pending:
        raise ValueError('shop transaction requires a settled open state')


def removable_indices(state):
    result=[]
    for i in range(len(state.run.public.observation['inventory']['deck'])):
        try:_primitive(state,'remove',target_index=i)
        except LocalEffectGap:raise
        except ValueError:continue
        result.append(i)
    return tuple(result)


def choices(state):
    _require_ready(state)
    money=state.run.public.observation['player']['gold'];ids=relic_ids(state)
    pots=state.run.public.observation['inventory'].get('potions')
    free=bool(pots is not None and any(canonical('potion',p['id'])=='EMPTY_POTION_SLOT' for p in pots))
    result=[ShopChoice('leave','leave')]
    for o in state.offers:
        if o['price']>money:continue
        if o['kind']=='buy_potion' and ('SOZU' in ids or not free):continue
        if o['kind']=='buy_relic' and o['item_id'] in {'PRISMATIC_SHARD','RUNIC_DOME'}:continue
        result.append(ShopChoice(o['key'],o['kind']))
    if state.removal_price is not None and state.removal_price<=money:
        result.extend(ShopChoice('remove','buy_remove',i) for i in removable_indices(state))
    return tuple(result)


def bind_root_action(state, choice):
    """Decode against THIS observation; future shelf keys are not action indices."""
    options=state.run.public.observation.get('screen',{}).get('decision_options')
    if options is None:raise ValueError('only observed root actions can be bound')
    if choice.kind in ('leave','buy_remove'):
        found=[o for o in options if o['kind']==choice.kind]
    else:
        offer=next(o for o in state.offers if o['key']==choice.key)
        same=[o for o in state.offers if offer_identity(o)==offer_identity(offer)]
        rank=same.index(offer)
        found=[o for o in options if o['kind'] in BUY_KINDS and
               offer_identity(normalize_offer(o,''))==offer_identity(offer)]
        found=found[rank:rank+1]
    if len(found)!=1:raise ValueError('shop action missing or ambiguous')
    return replace(choice,action=Action.from_dict(found[0]['action']))


def _pay(state,price):
    obs=state.run.public.as_dict()['observation']
    if obs['player']['gold']<price:raise ValueError('unaffordable shop transaction')
    obs['player']['gold']-=price
    for r in obs['inventory']['relics']:
        if canonical('relic',r['id'])=='MAW_BANK':r['counter']=0
    return _derive(state,obs)


def begin_removal(state):
    _require_ready(state)
    if state.removal_price is None or not removable_indices(state):raise ValueError('removal unavailable')
    paid=state.removal_price
    after=replace(_pay(state,paid),removal_price=None,removal_services_used=state.removal_services_used+1)
    return _pending(after,'select_remove',paid=paid)


def finish_removal(state,target_index):
    if len(state.pending)!=1 or state.pending[0]['kind']!='select_remove':
        raise ValueError('no paid removal awaiting selection')
    if target_index not in removable_indices(state):raise ValueError('invalid removal target')
    # Fee is already paid. Local primitive must not charge it again.
    return replace(_primitive(state,'remove',target_index=target_index),pending=())


def _relic(state,identifier):
    ids=relic_ids(state)
    if identifier in ids:raise ValueError('duplicate owned relic offered')
    entry=_catalog().lookup('relic',identifier)
    obs=state.run.public.as_dict()['observation'];player=obs['player']
    counter={'OMAMORI':2,'MAW_BANK':1,'MATRYOSHKA':2,'LIZARD_TAIL':1}.get(identifier,0)
    obs['inventory']['relics'].append(dict(id=identifier,counter=counter))
    if identifier in MAX_HP_RELICS:
        amount=MAX_HP_RELICS[identifier];player['max_hp']+=amount
        if 'MARK_OF_THE_BLOOM' not in ids:
            player['hp']=player['max_hp'] if identifier=='LEES_WAFFLE' else min(player['max_hp'],player['hp']+amount)
    if identifier=='POTION_BELT':
        if 'potions' not in obs['inventory']:raise LocalEffectGap('unknown_potion_capacity')
        obs['inventory']['potions'] += [dict(id='EMPTY_POTION_SLOT'),dict(id='EMPTY_POTION_SLOT')]
    state=_derive(state,obs)
    if identifier=='MEMBERSHIP_CARD':
        offers=tuple(_freeze_public_value(dict(o,price=math.floor(o['price']*.5+.5))) for o in state.offers)
        # Native shelf lacks the undiscounted purge base. With Courier, original
        # ShopScreen reprices from that base; never guess it by inverse rounding.
        cost=state.removal_price
        if cost is not None:
            if 'SMILING_MASK' in ids:cost=50
            elif 'THE_COURIER' in ids:state=_pending(state,'removal_price_reveal')
            else:cost=math.floor(cost*.5+.5)
        state=replace(state,offers=offers,removal_price=cost)
        if not state.complete_shelf:state=_pending(state,'full_shelf_reveal',reason='discount')
    elif identifier=='SMILING_MASK' and state.removal_price is not None:
        state=replace(state,removal_price=50)
    elif identifier in SELECT_RELICS:
        task=SELECT_RELICS[identifier]
        deck=state.run.public.observation['inventory']['deck']
        eligible=[i for i,c in enumerate(deck) if
            task=='duplicate_card' or _catalog().lookup('card',c['id'])['type']==task.removeprefix('bottle_').upper()]
        if eligible:state=_pending(state,task,candidates=eligible)
    elif identifier in RANDOM_RELICS:
        kind,count=RANDOM_RELICS[identifier]
        if kind.startswith('upgrade_'):
            target_type='ATTACK' if kind=='upgrade_attacks' else 'SKILL'
            eligible=[i for i,c in enumerate(state.run.public.observation['inventory']['deck'])
                if _catalog().lookup('card',c['id'])['type']==target_type and
                (not c['upgrades'] or c['id']=='SEARING_BLOW')]
            if len(eligible)<=count:
                for i in eligible:state=_primitive(state,'upgrade',target_index=i)
            else:state=_pending(state,kind,count=count,candidates=eligible)
        else:state=_pending(state,kind,count=count)
    elif identifier in {'MOLTEN_EGG','TOXIC_EGG','FROZEN_EGG'}:
        from .expert_cards import EGG_TYPES
        offers=[]
        for o in state.offers:
            if o['kind']=='buy_card' and not o['card']['upgrades'] and _catalog().lookup('card',o['card']['id'])['type']==EGG_TYPES[identifier]:
                o=_freeze_public_value(dict(o,card=dict(o['card'],upgrades=1)))
            offers.append(o)
        state=replace(state,offers=tuple(offers))
    elif entry['tier'] not in {'Common','Uncommon','Rare','Shop'} or identifier in {'OLD_COIN'}:
        # Such offers are outside a normal Ironclad shop acquisition context.
        state=_pending(state,'nonstandard_relic_acquisition',relic=identifier)
    return state


def transact(state,choice):
    _require_ready(state)
    if choice not in tuple(replace(c,action=choice.action) for c in choices(state)):
        raise ValueError('unavailable shop choice')
    if choice.action is not None and bind_root_action(state,choice).action != choice.action:
        raise ValueError('bound shop action does not match transaction')
    return _transact_available(state,choice)


def transact_choices(state):
    """Resolve every legal choice against one immutable root.

    Eligibility is computed once, not once per candidate. This owns the choice
    enumeration, so callers cannot supply an unchecked action or stale cache.
    The individual card mutation still validates its exact target and callbacks.
    """
    legal=choices(state)
    return tuple(_transact_available(state,c) for c in legal)


def _transact_available(state,choice):
    before=state
    paid=0
    if choice.kind=='leave':
        after=replace(_derive(state,state.run.public.as_dict()['observation']),closed=True)
    elif choice.kind=='buy_remove':
        # The root already proved fee/target availability. Payment only changes
        # gold and Maw Bank's counter, not card removal eligibility. Keep the
        # staged public begin/finish API for actual selection-window reconciliation.
        paid=state.removal_price
        after=replace(_pay(state,paid),removal_price=None,
                      removal_services_used=state.removal_services_used+1)
        after=_primitive(after,'remove',target_index=choice.target_index)
    else:
        offer=next(o for o in state.offers if o['key']==choice.key);paid=offer['price']
        after=replace(_pay(state,paid),offers=tuple(o for o in state.offers if o['key']!=choice.key))
        if choice.kind=='buy_card':after=_primitive(after,'take_card',card=offer['card'])
        elif choice.kind=='buy_potion':
            obs=after.run.public.as_dict()['observation']
            slot=next(i for i,p in enumerate(obs['inventory']['potions']) if canonical('potion',p['id'])=='EMPTY_POTION_SLOT')
            obs['inventory']['potions'][slot]=dict(id=offer['item_id']);after=_derive(after,obs)
        else:after=_relic(after,offer['item_id'])
        if 'THE_COURIER' in relic_ids(after):after=_pending(after,'restock',slot=choice.key,category=choice.kind)
        if not after.complete_shelf and after.run.public.observation['player']['gold']>before.run.public.observation['player']['gold']:
            after=_pending(after,'full_shelf_reveal',reason='gold_gain')
    b=before.run.public.observation['player'];a=after.run.public.observation['player']
    effects=MappingProxyType({k+'_delta':a[k]-b[k] for k in ('hp','max_hp','gold')})
    return ShopTransition(before,after,choice,paid,effects)


def persistent_identity(state,*,include_gold=True):
    """Comparison at shop EXIT, never claiming remaining shelf opportunities equal."""
    obs=state.run.public.as_dict()['observation'];inv=obs['inventory']
    cards=[]
    for i,c in enumerate(inv['deck']):
        detail=state.run.run_public['deck'][i] if state.run.run_public else {}
        cards.append((canonical('card',c['id']),c['upgrades'],detail.get('misc',c.get('misc')),
                      *(detail.get(k) for k in BOTTLES)))
    relics=[(canonical('relic',r['id']),r.get('counter')) for r in inv['relics']]
    player=dict(obs['player'])
    if not include_gold:player.pop('gold',None)
    return dict(player=player,deck=sorted(cards,key=repr),relics=relics,
        potions=[canonical('potion',p['id']) for p in inv.get('potions',[])],
        keys=obs.get('keys'),run=obs.get('run'),map=obs.get('map'))


def exit_dominance(before,after):
    """A narrow no-benefit certificate, not a general learned shop valuation."""
    if before.pending or after.pending:return dict(proven=False,reason='unresolved_effect')
    if after.removal_services_used<before.removal_services_used:
        return dict(proven=False,reason='future_removal_cost_improved')
    if persistent_identity(before,include_gold=False)!=persistent_identity(after,include_gold=False):
        return dict(proven=False,reason='persistent_state_changed')
    cost=before.run.public.observation['player']['gold']-after.run.public.observation['player']['gold']
    # Gold can have non-monotone value with these relics (Bloody Idol healing,
    # Ectoplasm collection, and Maw Bank spending). Do not prove dominance there.
    if relic_ids(before)&{'BLOODY_IDOL','ECTOPLASM','MAW_BANK'}:
        return dict(proven=False,reason='gold_interaction_requires_value')
    return dict(proven=cost>0,reason='same_exit_resources_less_gold' if cost>0 else 'no_strict_gold_loss',gold_lost=cost,
        extra_removal_services=after.removal_services_used-before.removal_services_used)


def enumerate_visits(root,*,max_transactions=2,max_nodes=128):
    """Bounded deterministic prefixes plus explicit stochastic/selection frontiers.

No model scores are invented. Every settled prefix supplies a leave alternative;
node and depth limits are returned, not hidden as a complete shopping optimum.
"""
    _require_ready(root);_int(max_transactions,'max_transactions',minimum=1,maximum=8)
    _int(max_nodes,'max_nodes',minimum=1,maximum=100000)
    queue=[(root,())];exits=[];frontiers=[];expanded=0;cutoff=False
    while queue:
        state,path=queue.pop(0)
        exits.append((transact(state,ShopChoice('leave','leave')).after,path))
        if len(path)>=max_transactions:continue
        for choice in choices(state):
            if choice.kind=='leave':continue
            if expanded>=max_nodes:cutoff=True;break
            after=transact(state,choice).after;expanded+=1;tail=path+(choice,)
            if after.pending:frontiers.append((after,tail))
            else:queue.append((after,tail))
        if cutoff:break
    return dict(exits=exits,frontiers=frontiers,expanded=expanded,node_limit_reached=cutoff,
        depth_limit=max_transactions,complete_shelf=root.complete_shelf,
        unresolved_prefixes=len(queue),optimality_claim=False)
