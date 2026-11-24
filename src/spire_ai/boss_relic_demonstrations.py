"""Forward-only, partial boss-relic demonstrations from reviewed run histories.

The last boss card reward is an inventory anchor, not the relic decision itself.
Apply that reward once, then stop BEFORE the boss chest's selected relic and its
effects. Resources and future act bosses remain unknown. No value supervision.
"""
import copy

from .actions import Action, ActionKind
from .boss_cards import reconstruct_win, FLOORS
from . import expert_cards as legacy
from .expert_cards import HistoryGap, NO_CARD, integer, card_key, obtained_card
from .policy_api import PolicyInput
from .rule_catalog import RuleCatalog
from .run_options import RUN_OPTIONS_VERSION

VERSION = 'human-boss-relic-partial-v1'
BUILD = '2020-07-30'
EXCLUDED_RELICS = frozenset(('RUNIC_DOME', 'PRISMATIC_SHARD'))


def offer(record, catalog):
    """noPick logs ONLY not_picked; recognize it only with a full valid menu."""
    if not isinstance(record, dict) or not isinstance(record.get('not_picked'), list):
        raise HistoryGap('missing_boss_offer')
    picked = record.get('picked', 'SKIP')
    if not isinstance(picked, str):
        raise HistoryGap('invalid_boss_pick')
    names = record['not_picked'] + ([] if picked == 'SKIP' else [picked])
    entries = [catalog.lookup('relic', name) if isinstance(name, str) else None for name in names]
    if any(e is None or e['tier'] != 'Boss' for e in entries):
        raise HistoryGap('unknown_or_nonboss_offer')
    ids = [e['id'] for e in entries]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise HistoryGap('incomplete_or_duplicate_boss_offer')
    chosen = None if picked == 'SKIP' else catalog.lookup('relic', picked)['id']
    return sorted(ids), chosen


def extract(run, *, catalog=None):
    catalog = catalog or RuleCatalog.bundled()
    if run.get('build_version') != BUILD:
        raise HistoryGap('unreviewed_boss_metric_build')
    bosses = run.get('boss_relics')
    if not isinstance(bosses, list) or len(bosses) > 2:
        raise HistoryGap('unexpected_boss_relic_count')
    adapted = copy.deepcopy(run)
    for record in adapted['boss_relics']:
        try:
            _, chosen = offer(record, catalog)
        except HistoryGap:
            continue  # The original reconstruction records an appropriate barrier.
        if chosen is None:
            record['picked'] = 'SKIP'
    replay = reconstruct_win(adapted, catalog=catalog)
    if replay['unexplained_final_mismatch'] or (replay['final_deck_matches'] is None and not replay['barriers']):
        raise HistoryGap('unverified_final_inventory_audit')
    if replay['barriers'] and replay['barriers'][0]['reason'].startswith(
            ('remove_missing_card:', 'ambiguous_or_missing_upgrade:')):
        raise HistoryGap('unlocalized_inventory_contradiction')
    rows, excluded = [], []
    for index, record in enumerate(bosses):
        act = index + 1
        boss_floor, chest_floor = FLOORS[act], FLOORS[act] + 1
        try:
            offered, chosen = offer(record, catalog)
            if chosen in EXCLUDED_RELICS:
                raise HistoryGap('chosen_relic_outside_normal_clear_scope')
            path = run['path_per_floor']
            if len(path) < chest_floor or path[boss_floor-1] != 'B' or path[chest_floor-1] is not None:
                raise HistoryGap('unverified_boss_chest_transition')
            if any(b['floor'] <= boss_floor for b in replay['barriers']):
                raise HistoryGap('tainted_pre_chest_inventory')
            anchors = [r for r in replay['rows'] if r['floor'] == boss_floor and r['act'] == act]
            if not anchors:
                raise HistoryGap('no_last_boss_reward_anchor')
            anchor = max(anchors, key=lambda r:r['choice_index'])
            if anchor['choice_index'] != max(i for i,c in enumerate(run['card_choices'])
                                             if integer(c['floor']) == boss_floor):
                raise HistoryGap('anchor_not_last_boss_reward')
            obs = copy.deepcopy(anchor['public']['observation'])
            deck = obs['inventory']['deck']
            relics = {r['id'] for r in obs['inventory']['relics']}
            picked_card = run['card_choices'][anchor['choice_index']]['picked']
            if picked_card not in NO_CARD:
                deck.append(obtained_card(picked_card, relics, catalog))
            deck.sort(key=card_key)
            if set(offered) & relics:
                raise HistoryGap('offered_relic_already_owned')
            if 'BLACK_BLOOD' in offered and 'BURNING_BLOOD' not in relics:
                raise HistoryGap('black_blood_missing_starter')
            obs['run']['floor'] = chest_floor
            # The next boss is not yet public. Do not relabel the defeated boss
            # as the upcoming boss or mine future damage_taken for its identity.
            obs['run'].pop('boss', None)
            obs['player'] = {}
            allowed = [r for r in offered if r not in EXCLUDED_RELICS]
            actions = [Action(ActionKind.CHOOSE, choice_index=i) for i in range(len(allowed))]
            options = [dict(kind='boss_relic', item_id=r, action=a.as_dict())
                       for r,a in zip(allowed, actions, strict=True)]
            actions.append(Action(ActionKind.PROCEED))
            options.append(dict(kind='leave', action=actions[-1].as_dict()))
            obs['screen'] = dict(type='BOSS_RELIC_REWARDS', run_options_version=RUN_OPTIONS_VERSION,
                                 decision_options=options)
            public = PolicyInput(obs, tuple(actions))
            rows.append(dict(version=VERSION, public=public.as_dict(), input_sha256=public.fingerprint(),
                target=allowed.index(chosen) if chosen is not None else len(allowed),
                label='relic' if chosen is not None else 'skip', boss_index=index,
                floor=chest_floor, act=act, anchor_choice_index=anchor['choice_index'],
                partial_fields=list(legacy.PARTIAL_FIELDS), relic_order_known=False,
                excluded_offered_relics=sorted(set(offered)&EXCLUDED_RELICS),
                limitations=['pre-choice resources unknown', 'next act boss unknown',
                             'normal-clear scope', 'human preference, not optimality or value']))
        except (HistoryGap, KeyError, TypeError, ValueError) as error:
            excluded.append(dict(boss_index=index, act=act, reason=str(error)))
    return dict(rows=rows, excluded=excluded)


def encode_row(row, encoder):
    if row.get('version') != VERSION:
        raise ValueError('boss relic partial contract mismatch')
    if row['public']['observation']['screen']['type'] != 'BOSS_RELIC_REWARDS':
        raise ValueError('boss relic screen required')
    return legacy.encode_row(dict(row, version=legacy.VERSION), encoder)
