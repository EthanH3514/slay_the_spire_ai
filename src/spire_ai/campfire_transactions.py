"""Public ordinary-clear campfire alternatives BEFORE committing to SMITH.

Each candidate consumes this campfire once. Exact local rest/lift/deck changes
are resolved; Dream Catcher and Dig retain unknown reward windows. Entry-room
effects (e.g. Eternal Feather/Tea Set) have already happened at this snapshot.
No hypothetical free Skip, Recall, private seed or future reward identity.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
import struct

from .actions import Action
from .policy_api import PolicyInput, _freeze_public_value, _thaw_public_value
from .run_afterstates import (capture, local_transition, LocalEffectGap, _catalog,
                             _relics, _digest, _heal)
from .run_options import public_run_options

VERSION = 'public-ordinary-campfire-transactions-v1'
KINDS = ('rest','smith','remove','lift','dig','leave')
PENDING = ('card_reward','relic_reward')


@dataclass(frozen=True, slots=True)
class CampfireState:
    run: object
    closed: bool = False
    pending: tuple = ()

    def fingerprint(self):
        return _digest(dict(version=VERSION,run=self.run.fingerprint(),closed=self.closed,
                            pending=_thaw_public_value(self.pending)))


@dataclass(frozen=True, slots=True)
class CampfireChoice:
    kind: str
    target_index: int | None
    action: Action


@dataclass(frozen=True, slots=True)
class CampfireTransition:
    before: CampfireState
    after: CampfireState
    choice: CampfireChoice
    effects: object


def capture_campfire(public, *, run_public=None):
    run=capture(public,relic_order_known=True,run_public=run_public)
    if public.observation.get('screen',{}).get('type')!='REST':
        raise ValueError('observed campfire decision required')
    for name in ('hp','max_hp'):
        if run.public.observation.get('player',{}).get(name) is None:
            raise LocalEffectGap('unknown_campfire_'+name)
    state=CampfireState(run)
    _options(state)  # Verify complete semantic menu, no Recall/unknown option.
    return state


def _options(state):
    if not isinstance(state,CampfireState) or state.closed or state.pending:
        raise ValueError('unconsumed campfire required')
    options=public_run_options(state.run.public)
    actions=state.run.public.legal_actions
    if not options or len(actions)!=len(set(actions)) or len({o['kind'] for o in options})!=len(options):
        raise ValueError('complete distinct campfire options required')
    if any(o['kind'] not in KINDS for o in options):
        raise LocalEffectGap('unreviewed_or_recall_campfire_option')
    if any(o['kind']=='leave' for o in options) and len(options)!=1:
        raise ValueError('ordinary campfire Skip cannot compete with another activity')
    return options


def target_indices(state,kind):
    if kind not in ('smith','remove'):raise ValueError('targeted campfire activity required')
    operation='upgrade' if kind=='smith' else 'remove'
    result=[]
    for i in range(len(state.run.public.observation['inventory']['deck'])):
        try:local_transition(state.run,operation,target_index=i)
        except LocalEffectGap:raise
        except ValueError:continue  # Explicitly ineligible card, not a dropped unknown effect.
        result.append(i)
    return tuple(result)


def choices(state):
    options=_options(state);relics=_relics(state.run,_catalog());result=[]
    for option in options:
        kind=option['kind'];action=Action.from_dict(option['action'])
        if ((kind=='rest' and 'COFFEE_DRIPPER' in relics) or
                (kind=='smith' and 'FUSION_HAMMER' in relics)):
            raise ValueError('campfire menu contradicts blocking relic')
        required={'remove':'PEACE_PIPE','lift':'GIRYA','dig':'SHOVEL'}.get(kind)
        if required and required not in relics:raise ValueError('campfire activity missing required relic')
        if kind=='lift':
            rows=[r for r in state.run.public.observation['inventory']['relics']
                  if _catalog().lookup('relic',r['id'])['id']=='GIRYA']
            if len(rows)!=1 or type(rows[0].get('counter')) is not int or not 0<=rows[0]['counter']<3:
                raise LocalEffectGap('unknown_or_ineligible_girya_counter')
        if kind in ('smith','remove'):
            indices=target_indices(state,kind)
            if not indices:raise ValueError('empty targeted campfire option')
            result.extend(CampfireChoice(kind,i,action) for i in indices)
        else:result.append(CampfireChoice(kind,None,action))
    return tuple(result)


def _f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def transact(state,choice):
    if choice not in choices(state):raise ValueError('unobserved/forged campfire candidate')
    return _resolve(state,choice)


def transact_choices(state):
    return tuple(_resolve(state,choice) for choice in choices(state))


def _resolve(state,choice):
    effects=dict(hp_delta=0,max_hp_delta=0,gold_delta=0,healing_requested=0,max_hp_loss_requested=0)
    pending=[]
    if choice.kind in ('smith','remove'):
        t=local_transition(state.run,'upgrade' if choice.kind=='smith' else 'remove',target_index=choice.target_index)
        after=t.after;effects=dict(t.effects)
    else:
        obs=state.run.public.as_dict()['observation'];relics=_relics(state.run,_catalog())
        player=obs['player']
        if choice.kind=='rest':
            # Java int(float(maxHp)*0.3f), then Regal Pillow, then normal heal
            # cap/block. Never reapply effects from entering the room.
            heal=int(_f32(_f32(player['max_hp'])*_f32(.3)))+(15 if 'REGAL_PILLOW' in relics else 0)
            effects['healing_requested']=heal
            _heal(player,heal,relics)
            effects['hp_delta']=player['hp']-state.run.public.observation['player']['hp']
            if 'DREAM_CATCHER' in relics:pending.append(dict(kind='card_reward',count=1))
        elif choice.kind=='lift':
            for relic in obs['inventory']['relics']:
                if _catalog().lookup('relic',relic['id'])['id']=='GIRYA':relic['counter']+=1
        elif choice.kind=='dig':pending.append(dict(kind='relic_reward',count=1))
        elif choice.kind!='leave':raise ValueError('unknown campfire activity')
        obs['screen']={}
        after=replace(capture(PolicyInput(obs,()),relic_order_known=True,run_public=state.run.run_public),
                      phase='local_effect_resolved',opportunities_known=False)
    return CampfireTransition(state,CampfireState(after,True,tuple(_freeze_public_value(p) for p in pending)),
                              choice,MappingProxyType(effects))


def selection_action(root,public,choice):
    """Bind the original public deck INSTANCE to the native ordered selector.

Native UPGRADE/REMOVE iterate eligible deck indices in order. Validate every
option; do not use a same-name dictionary that confuses misc/bottle instances.
"""
    if choice not in choices(root) or choice.kind not in ('smith','remove'):
        raise ValueError('original targeted candidate required')
    if public.observation.get('screen',{}).get('type')!='CARD_SELECT':
        raise ValueError('campfire commitment did not open its target selector')
    options=public_run_options(public);indices=target_indices(root,choice.kind)
    if len(options)!=len(indices):raise ValueError('campfire target count mismatch')
    deck=root.run.public.observation['inventory']['deck'];task='upgrade' if choice.kind=='smith' else 'remove'
    for option,index in zip(options,indices,strict=True):
        shown=option.get('card',{});card=deck[index]
        if (option['kind']!='select_card' or option.get('task')!=task or
                _catalog().canonical_id('card',shown.get('id'))!=_catalog().canonical_id('card',card['id']) or
                shown.get('upgrades')!=card['upgrades'] or
                ('misc' in shown and shown['misc']!=card.get('misc',0))):
            raise ValueError('campfire target instance mapping mismatch')
    return Action.from_dict(options[indices.index(choice.target_index)]['action'])


def verify_immediate(transition,public,supplement):
    from .shop_transactions import persistent_identity
    actual=replace(transition.after,run=capture(public,relic_order_known=True,run_public=supplement))
    expected=persistent_identity(transition.after);observed=persistent_identity(actual)
    for name in ('player','deck','relics','potions','keys'):
        if expected[name]!=observed[name]:raise ValueError('campfire afterstate mismatch: '+name)
    screen=public.observation['screen']['type']
    if transition.after.pending:
        if screen not in ('COMBAT_REWARD','CARD_REWARD','REWARDS'):
            raise ValueError('campfire reward window was lost')
    elif screen!='MAP':raise ValueError('resolved campfire did not return to map')


def execute(environment,root,choice):
    """Execute only an explicitly selected candidate; no ranking/fallback here."""
    import json
    from .policy_api import policy_input_from_state
    if environment._current_public_input().fingerprint()!=root.run.public.fingerprint():
        raise ValueError('stale campfire root')
    transition=transact(root,choice)
    result=environment.step(choice.action)
    if choice.kind in ('smith','remove'):
        action=selection_action(root,policy_input_from_state(result.state),choice)
        result=environment.step(action)
    verify_immediate(transition,policy_input_from_state(result.state),
                     json.loads(environment.session.public_run_supplement()))
    return result,transition
