"""Which card to SMITH after committing to an upgrade, from verified wins.

This is partial-state demonstration learning, not rest-vs-upgrade supervision.
No final deck, future resource series, seed or outcome repairs the observation.
Quiet intervals after an admitted same-act reward are the initial safe scope.
"""
from collections import Counter
import copy

from .actions import Action
from .boss_cards import reconstruct_win, FLOORS
from .expert_cards import (HistoryGap, PARTIAL_FIELDS, NO_CARD, card, card_key,
                          obtained_card, integer, act_for_floor)
from .conditional_removal import prefix_signature, BOTTLES
from .policy_api import PolicyInput
from .rule_catalog import RuleCatalog
from .run_afterstates import capture, local_transition

VERSION = 'conditional-smith-prefix-v1'
TASK = 'which_card_after_smith_committed'
PARTIAL = tuple(PARTIAL_FIELDS)


def candidate_groups(deck, catalog):
    groups = {}
    for index, current in enumerate(deck):
        parsed = card(current['id']+('+'+str(current['upgrades']) if current['upgrades'] else ''), catalog)
        entry = catalog.lookup('card', parsed['id'])
        if entry['color'] not in ('RED', 'COLORLESS', 'CURSE'):
            raise HistoryGap('outside_ironclad_card_scope')
        if entry['type'] not in ('ATTACK', 'SKILL', 'POWER'):
            continue
        if parsed['upgrades'] and parsed['id'] != 'SEARING_BLOW':
            continue
        groups.setdefault(card_key(parsed), []).append(index)
    if any(key[0] == 'RITUAL_DAGGER' and len(indices) > 1 for key,indices in groups.items()):
        raise HistoryGap('duplicate_persistent_misc_unknown')
    return [dict(card=dict(id=key[0], upgrades=key[1]), indices=indices)
            for key,indices in sorted(groups.items())]


def target_index(name, groups, catalog):
    target = card(name, catalog)
    candidates = [i for i,g in enumerate(groups) if g['card']['id'] == target['id'] and
        (g['card']['upgrades'] == target['upgrades'] if '+' in name else
         g['card']['upgrades'] == 0 or g['card']['id'] == 'SEARING_BLOW')]
    if len(candidates) != 1:
        raise HistoryGap('ambiguous_or_ineligible_smith_target')
    return candidates[0]


def extract(run, *, expected_prefixes=None, catalog=None):
    catalog = catalog or RuleCatalog.bundled()
    result = reconstruct_win(run, catalog=catalog)
    if expected_prefixes is not None and sorted(map(prefix_signature, result['rows'])) != sorted(expected_prefixes):
        raise ValueError('frozen accepted reward-prefix replay differs')
    if result['unexplained_final_mismatch'] or (result['final_deck_matches'] is None and not result['barriers']):
        raise HistoryGap('unverified_final_inventory_audit')
    if result['barriers'] and result['barriers'][0]['reason'].startswith(('remove_missing_card:', 'ambiguous_or_missing_upgrade:')):
        raise HistoryGap('unlocalized_inventory_contradiction')
    rewards = {integer(c['floor']) for c in run['card_choices']}
    purchases = {integer(f) for f in run['item_purchase_floors']}
    purges = {integer(f) for f in run['items_purged_floors']}
    events = {integer(c['floor']) for c in run['event_choices']}
    relics_at = {integer(c['floor']) for c in run['relics_obtained']}
    fires = Counter(integer(c['floor']) for c in run['campfire_choices'])
    edited_fires = {integer(c['floor']) for c in run['campfire_choices'] if c['key'] not in ('REST', 'RECALL', 'LIFT')}
    channels = rewards | purchases | purges | events | relics_at | edited_fires
    anchors = sorted(result['rows'], key=lambda r:(r['floor'], r['choice_index']))
    rows = []; excluded = []; count = 0
    for index, fire in enumerate(run['campfire_choices']):
        if fire['key'] != 'SMITH': continue
        count += 1; floor = integer(fire['floor']); reasons = []; act = act_for_floor(floor)
        path = run['path_per_floor']
        if not 1 <= floor <= len(path) or path[floor-1] != 'R': reasons.append('unverified_campfire_room')
        if act > 3: reasons.append('act4_outside_scope')
        if floor > FLOORS.get(act, -1): reasons.append('outside_reviewed_boss_visibility_window')
        if fires[floor] != 1: reasons.append('multiple_same_floor_campfire_actions')
        if floor in rewards | purchases | purges | events | relics_at:
            reasons.append('same_floor_other_inventory_channel')
        barrier = next((b for b in result['barriers'] if b['floor'] <= floor), None)
        if barrier: reasons.append('prefix_tainted:'+barrier['reason'])
        possible = [r for r in anchors if r['floor'] < floor and r['act'] == act]
        anchor = possible[-1] if possible else None
        if anchor is None: reasons.append('no_verified_same_act_reward_anchor')
        if anchor is not None:
            left = anchor['floor']
            same = [i for i,c in enumerate(run['card_choices']) if integer(c['floor']) == left]
            if anchor['choice_index'] != max(same): reasons.append('anchor_not_last_recorded_floor_choice')
            if any(left < f < floor for f in channels): reasons.append('intervening_inventory_channel')
        if not reasons:
            try:
                obs = copy.deepcopy(anchor['public']['observation'])
                deck = obs['inventory']['deck']; relics = [r['id'] for r in obs['inventory']['relics']]
                if 'FUSION_HAMMER' in relics: raise HistoryGap('smith_blocked_by_fusion_hammer')
                # A bottled and an ordinary copy are not strategically equivalent;
                # historical logs do not expose the bottle assignment.
                if BOTTLES & set(relics): raise HistoryGap('unknown_bottled_upgrade_identity')
                picked = run['card_choices'][anchor['choice_index']]['picked']
                if picked not in NO_CARD: deck.append(obtained_card(picked, relics, catalog))
                deck.sort(key=card_key); groups = candidate_groups(deck, catalog)
                if len(groups) < 2: raise HistoryGap('no_nontrivial_upgrade_choice')
                target = target_index(fire['data'], groups, catalog)
                obs['run']['floor'] = floor; obs['screen'] = {}
                public = PolicyInput(obs, ())
                rows.append(dict(version=VERSION, task=TASK, public=public.as_dict(),
                    input_sha256=public.fingerprint(), candidates=groups, target=target,
                    partial_fields=list(PARTIAL), smith_index=index, floor=floor, act=act,
                    anchor=dict(floor=left, choice_index=anchor['choice_index'], input_sha256=anchor['input_sha256']),
                    boss_context=copy.deepcopy(anchor['boss_context'])))
            except HistoryGap as error: reasons.append(str(error))
        if reasons: excluded.append(dict(smith_index=index, floor=floor, reasons=reasons))
    return dict(version=VERSION, rows=rows, excluded=excluded, upgrades=count,
                reward_prefixes=len(result['rows']), final_deck_matches=result['final_deck_matches'])


def encode_row(row, encoder):
    if row.get('version') != VERSION or row.get('task') != TASK or row.get('partial_fields') != list(PARTIAL):
        raise ValueError('conditional upgrade row contract')
    wire = row['public']; public = PolicyInput(wire['observation'], tuple(Action.from_dict(a) for a in wire['legal_actions']))
    if (public.fingerprint() != row['input_sha256'] or public.legal_actions
            or public.observation.get('player') != {} or public.observation.get('screen') != {}):
        raise ValueError('partial state-only smith observation required')
    if (BOTTLES | {'FUSION_HAMMER'}) & {r['id'] for r in public.observation['inventory']['relics']}:
        raise ValueError('smith eligibility/identity unresolved')
    groups = candidate_groups(public.observation['inventory']['deck'], encoder.catalog)
    if groups != row['candidates'] or type(row['target']) is not int or not 0 <= row['target'] < len(groups):
        raise ValueError('complete upgrade-equivalence groups required')
    root = capture(public, relic_order_known=False)
    encoded = tuple(encoder.encode(local_transition(root, 'upgrade', target_index=g['indices'][0])) for g in groups)
    for candidate in encoded:
        for state in (candidate.before, candidate.after):
            table = state.entities.tables['global']
            if any(table.known[0][table.columns.index(k)] for k in ('hp', 'max_hp', 'gold')):
                raise ValueError('unknown smith resources became known')
            relics = state.entities.tables['relics']; position = relics.columns.index('order')
            if any(mask[position] for mask in relics.known): raise ValueError('historical relic order leaked')
    return encoded
