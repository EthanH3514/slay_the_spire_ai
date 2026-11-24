"""Public tile memory and opt-in, exact-entry SL planning for Match and Keep.

Unknown cards are learned only by legal flips on disposable entry clones. They
are not inserted into live policy observations or the neural evaluator inputs.
"""
from collections import defaultdict
import copy
import re

from .policy_api import PolicyInput, _reject_forbidden_policy_keys

VERSION = 'public-match-and-keep-v1'
REDUCTION = 'match-acquisition-order-with-mismatch-suffix-v1'


def attach(public, board):
    _reject_forbidden_policy_keys(board, reject_seeds=True)
    if not isinstance(board, dict) or board.get('version') != VERSION:
        raise ValueError('Match and Keep public version mismatch')
    if board.get('available') is not True:
        raise ValueError('active Match and Keep board required')
    if set(board) != {'version', 'available', 'attempts_remaining', 'tiles'}:
        raise ValueError('unexpected Match and Keep fields')
    attempts = board['attempts_remaining']; tiles = board['tiles']
    if type(attempts) is not int or not 1 <= attempts <= 5 or not isinstance(tiles, list) or len(tiles) != 12:
        raise ValueError('invalid Match and Keep attempts/tiles')
    obs = public.observation; screen = obs['screen']
    if (screen['type'] != 'EVENT' or 'combat' in obs or
            screen.get('event_origin', {}).get('event_name') != 'Match and Keep'):
        raise ValueError('Match and Keep board outside owned event')
    for i, t in enumerate(tiles):
        if (not isinstance(t, dict) or type(t.get('index')) is not int or t['index'] != i or
                type(t.get('removed')) is not bool or type(t.get('known')) is not bool or
                set(t) != {'index', 'removed', 'known'} | ({'card'} if t['known'] else set()) or
                (t['removed'] and not t['known'])):
            raise ValueError('invalid tile memory or unknown identity leak')
        if t['known']:
            c=t['card']
            if (not isinstance(c,dict) or set(c) != {'id','upgrades','misc'} or
                    not isinstance(c['id'],str) or not c['id'] or c['id']=='INVALID' or
                    type(c['upgrades']) is not int or c['upgrades']<0 or type(c['misc']) is not int):
                raise ValueError('invalid known tile card')
    out = copy.deepcopy(public.as_dict()['observation']); s=out['screen']
    labels=s.get('choices',()); options=s.get('decision_options',())
    if len(labels)!=len(public.legal_actions) or len(options)!=len(labels):
        raise ValueError('unaligned Match and Keep choices')
    pairs=[]
    for label, option in zip(labels, options, strict=True):
        m=re.fullmatch(r'\[Flip\] Tiles (\d+) and (\d+)\.',label)
        if m is None:raise ValueError('invalid Match and Keep flip label')
        i,j=map(int,m.groups());pairs.append((i,j))
        if not 0<=i<j<12 or tiles[i]['removed'] or tiles[j]['removed']:
            raise ValueError('invalid/removed tile action')
        option.clear();option.update(action=public.legal_actions[len(pairs)-1].as_dict(),
            kind='match_pair', tiles=[i,j])
    expected=[(i,j) for i in range(12) for j in range(i+1,12)
              if not tiles[i]['removed'] and not tiles[j]['removed']]
    if pairs!=expected:raise ValueError('incomplete or reordered Match and Keep action set')
    s['match_and_keep']=copy.deepcopy(board)
    return PolicyInput(out,public.legal_actions)


def pair_actions(public):
    from .run_options import public_run_options
    rows=public_run_options(public)
    return {tuple(row['tiles']): action for row,action in zip(rows,public.legal_actions,strict=True)
            if row['kind']=='match_pair'}


def learn(env, budget):
    from .local_event_search import EventSearchError, PlannedStep, _guard, _place, _unchanged
    public=env._current_public_input(); board=public.observation['screen'].get('match_and_keep')
    if not board or board['attempts_remaining']!=5 or any(t['removed'] for t in board['tiles']):
        raise EventSearchError('match_requires_initial_board')
    if budget.max_steps < 11 or budget.max_depth < 5:
        raise EventSearchError('match_budget_cannot_probe_and_finish')
    entry=env.event_checkpoint(); actions=pair_actions(public); cards={}; probes=[]
    for i in range(0,12,2):
        branch=env.clone_exact()
        if branch is env:raise EventSearchError('clone_aliases_live_environment')
        result=branch.step(actions[i,i+1]); after=branch._current_public_input()
        _guard(after,'Match and Keep',_place(public))
        if result.terminated or result.truncated:raise EventSearchError('match_probe_terminated')
        seen=after.observation['screen'].get('match_and_keep',{})
        if seen.get('attempts_remaining')!=4:raise EventSearchError('match_probe_attempt_mismatch')
        for j in (i,i+1):
            tile=seen['tiles'][j]
            if not tile['known']:raise EventSearchError('match_probe_did_not_reveal')
            cards[j]=dict(tile['card'])
        probes.append(PlannedStep(actions[i,i+1],public.fingerprint(),after.fingerprint()))
    _unchanged(env,entry)
    return cards,tuple(probes)


def candidates(public, cards, padding=False):
    """Keep every acquisition order; equivalent physical pairs use one binding.

    Failed matches have no RNG/resource callbacks, only reveal memory and consume
    an attempt. With the board known, all misses can be moved after acquisitions.
    Six pairs and five attempts give <=1955 planning transitions, plus six probes.
    """
    board=public.observation['screen']['match_and_keep']; actions=pair_actions(public)
    groups=defaultdict(list)
    for t in board['tiles']:
        if t['known'] and t['card']!=cards[t['index']]:raise ValueError('probed Match and Keep identity changed')
        if not t['removed']:groups[cards[t['index']]['id']].append(t['index'])
    result=[]
    if not padding:
        for indices in groups.values():
            if len(indices)>=2:result.append((actions[indices[0],indices[1]],False))
    for pair,action in actions.items():
        if cards[pair[0]]['id']!=cards[pair[1]]['id']:
            result.append((action,True));break
    return tuple(result)
