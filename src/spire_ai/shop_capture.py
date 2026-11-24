"""Public shop snapshots and explicit offline interventions; no outcome labels.

The packet contains only what the player sees at this decision. Episode seeds,
policy RNG, action history and future outcomes belong to separate audit files.
"""
import hashlib
import json
from dataclasses import replace

from .actions import Action
from .policy_api import PolicyInput, policy_input_from_state, _reject_forbidden_policy_keys
from .run_afterstates import capture, local_transition, _digest, _reject_raw_history
from .shop_transactions import (capture_shop, choices, canonical,
    removable_indices, persistent_identity)

VERSION = 'complete-public-shop-packet-v1'
FIELDS = {'version', 'public', 'run_supplement', 'shelf'}


def packet(public, supplement, shelf):
    value = dict(version=VERSION, public=public.as_dict(), run_supplement=supplement, shelf=shelf)
    load_packet(value)
    return value


def load_packet(value):
    if not isinstance(value, dict) or set(value) != FIELDS or value['version'] != VERSION:
        raise ValueError('versioned public shop packet fields required')
    _reject_forbidden_policy_keys(value, reject_seeds=True)
    _reject_raw_history(value)
    wire=value['public']
    if set(wire) != {'observation','legal_actions'}:
        raise ValueError('public decision fields required')
    public=PolicyInput(wire['observation'],tuple(Action.from_dict(a) for a in wire['legal_actions']))
    root=capture_shop(public,run_public=value['run_supplement'],shelf=value['shelf'])
    if not root.complete_shelf:
        raise ValueError('complete visible shelf required')
    return root


def collect_packet(environment, state):
    public=policy_input_from_state(state)
    return packet(public,json.loads(environment.session.public_run_supplement()),
                  json.loads(environment.session.public_shop_shelf()))


def choice_wire(choice):
    return dict(key=choice.key,kind=choice.kind,target_index=choice.target_index)


def resolve_choice(root, wire):
    if not isinstance(wire,dict) or set(wire) != {'key','kind','target_index'}:
        raise ValueError('shop intervention fields required')
    found=[c for c in choices(root) if choice_wire(c)==wire]
    if len(found)!=1:raise ValueError('intervention no longer available')
    return found[0]


def nominate(value):
    """A bounded COST/SEMANTICS pilot, never a claim of exhaustive ranking.

    Leave plus one affordable card plus one eligible paid-removal target.
    Within each category a fixed public hash orders candidates; no outcome,
    expert label, learned score or real RNG seed participates in nomination.
    Full shelf/candidate inventory remains saved even for untested choices.
    """
    root=load_packet(value);legal=choices(root);fingerprint=_digest(value)
    selected=[]
    for kind in ('leave','buy_card','buy_remove'):
        pool=[c for c in legal if c.kind==kind]
        if pool:
            selected.append(min(pool,key=lambda c:hashlib.sha256(
                (fingerprint+_digest(choice_wire(c))).encode()).hexdigest()))
    return [choice_wire(c) for c in selected]


def removal_action(root, public, target_index):
    options=public.observation.get('screen',{}).get('decision_options',())
    if public.observation.get('screen',{}).get('type')!='CARD_SELECT':
        raise ValueError('paid removal did not open a selection')
    allowed=removable_indices(root)
    if len(options)!=len(allowed):raise ValueError('removal option/instance count mismatch')
    deck=root.run.public.observation['inventory']['deck']
    for option,index in zip(options,allowed,strict=True):
        card=option.get('card',{})
        if (option.get('kind')!='select_card' or option.get('task')!='remove' or
            (canonical('card',card.get('id')),card.get('upgrades')) !=
            (canonical('card',deck[index]['id']),deck[index]['upgrades'])):
            raise ValueError('removal option/instance mapping mismatch')
    if target_index not in allowed:raise ValueError('invalid removal target')
    action=Action.from_dict(options[allowed.index(target_index)]['action'])
    if action not in public.legal_actions:raise ValueError('removal action is not legal')
    return action


def resources(root):
    identity=persistent_identity(root)
    return {k:identity[k] for k in ('player','deck','relics','potions','keys')}


def verify_immediate(predicted, actual_public, supplement):
    actual=replace(predicted,run=capture(actual_public,relic_order_known=True,run_public=supplement))
    for effect in predicted.pending:
        if effect['kind'] not in ('upgrade_attacks', 'upgrade_skills'):
            continue
        # The relic upgrades distinct deck slots in place. Only the observed
        # upgrade counts identify the targets; derive all other fields locally.
        before = predicted.run.public.observation['inventory']['deck']
        after = actual.run.public.observation['inventory']['deck']
        if len(before) != len(after):
            raise ValueError('shop random upgrade changed deck size')
        targets = [i for i, (a, b) in enumerate(zip(before, after))
                   if a['upgrades'] != b['upgrades']]
        if len(targets) != effect['count'] or not set(targets).issubset(effect['candidates']):
            raise ValueError('shop random upgrade target count or eligibility mismatch')
        for i in targets:
            run = local_transition(replace(predicted.run, phase='decision'), 'upgrade', target_index=i).after
            predicted = replace(predicted, run=run)
        if predicted.run.run_public != actual.run.run_public:
            raise ValueError('shop random upgrade card instance mismatch')
    if resources(predicted)!=resources(actual):
        raise ValueError('shop intervention resource/instance mismatch')


def comparison_summary(rows):
    """Descriptive single-latent-world results, deliberately not BC targets."""
    groups={}
    for row in rows:
        if row['status'] not in ('win','loss','censored'):
            raise ValueError('failed branch is not a gameplay outcome')
        groups.setdefault(row['root_job'],[]).append(row)
    pairs=[]
    for root,group in groups.items():
        controls=[r for r in group if r['intervention']['kind']=='leave' and not r.get('repeat')]
        if len(controls)!=1:raise ValueError('one leave control per root required')
        control=controls[0]
        for row in group:
            if row.get('repeat') or row is control:continue
            pairs.append(dict(root_job=root,choice=row['intervention'],control=control['status'],
                candidate=row['status'],floor_delta=row['final_floor']-control['final_floor'],
                terminal_pair=all(r['status'] in ('win','loss') for r in (control,row)),
                # HP on different routes/floors is not matched-battle preservation.
                same_terminal_floor=row['final_floor']==control['final_floor']))
    return dict(pairs=pairs,roots=len(groups),independent_game_worlds_per_root=1,
        training_ready=False,supervised_choice_rows=0,
        target_limitation='one fixed latent future; subset nominations; no reliable best-action label',
        no_sl_deployment=False,offline_audit_only=True)
