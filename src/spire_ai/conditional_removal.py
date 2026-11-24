"""Partial-state imitation of WHICH card to purge after committing to removal.

Reuses verified reward-prefix reconstruction without modifying its frozen code.
The anchor is the last admitted reward BEFORE a later shop. Only an interval
with no intervening inventory channel is accepted. Current-floor purchases or
other edits have no cross-stream ordering, so they are excluded. Final decks,
per-floor gold/HP and outcomes never repair the observation.
"""
from collections import Counter
import copy

from .actions import Action
from .boss_cards import reconstruct_win, FLOORS
from .expert_cards import (HistoryGap, PARTIAL_FIELDS, NO_CARD, card, card_key,
    obtained_card, integer, act_for_floor, digest)
from .policy_api import PolicyInput
from .rule_catalog import RuleCatalog
from .run_afterstates import capture, local_transition, UNREMOVABLE

VERSION='conditional-paid-removal-prefix-v1'
BOTTLES=frozenset(('BOTTLED_FLAME','BOTTLED_LIGHTNING','BOTTLED_TORNADO'))
PARTIAL=(*PARTIAL_FIELDS,'removal_price','visible_shop_shelf')


def prefix_signature(row):
    return (row['choice_index'],row['floor'],row['input_sha256'],row['target'])


def candidate_groups(deck,catalog):
    """Identical ordinary copies give one effect; no guessed duplicate index."""
    groups={}
    for index,c in enumerate(deck):
        parsed=card(c['id']+('+'+str(c['upgrades']) if c['upgrades'] else ''),catalog)
        entry=catalog.lookup('card',parsed['id'])
        if entry['color'] not in ('RED','COLORLESS','CURSE'):
            raise HistoryGap('outside_ironclad_card_scope')
        if parsed['id'] in UNREMOVABLE:continue
        groups.setdefault(card_key(parsed),[]).append(index)
    # Historical metric IDs cannot distinguish different permanent dagger damage.
    if any(key[0]=='RITUAL_DAGGER' and len(indices)>1 for key,indices in groups.items()):
        raise HistoryGap('duplicate_persistent_misc_unknown')
    return [dict(card=dict(id=key[0],upgrades=key[1]),indices=indices)
            for key,indices in sorted(groups.items())]


def extract(run,*,expected_prefixes=None,catalog=None):
    catalog=catalog or RuleCatalog.bundled()
    result=reconstruct_win(run,catalog=catalog)
    if expected_prefixes is not None and sorted(map(prefix_signature,result['rows']))!=sorted(expected_prefixes):
        raise ValueError('frozen accepted reward-prefix replay differs')
    # Whole-run exclusions from the source dataset remain enforced by its reader.
    # Also keep this extractor safe for direct fixture/diagnostic callers.
    if result['unexplained_final_mismatch'] or (result['final_deck_matches'] is None and not result['barriers']):
        raise HistoryGap('unverified_final_inventory_audit')
    if result['barriers'] and result['barriers'][0]['reason'].startswith(('remove_missing_card:','ambiguous_or_missing_upgrade:')):
        raise HistoryGap('unlocalized_inventory_contradiction')
    names,floors=run['items_purged'],run['items_purged_floors']
    if len(names)!=len(floors):raise HistoryGap('unaligned_purge_history')
    purge_counts=Counter(integer(f) for f in floors)
    reward_floors={integer(c['floor']) for c in run['card_choices']}
    purchases={integer(f) for f in run['item_purchase_floors']}
    event_floors={integer(e['floor']) for e in run['event_choices']}
    relic_floors={integer(e['floor']) for e in run['relics_obtained']}
    fire_floors={integer(e['floor']) for e in run['campfire_choices']
                 if e['key'] not in ('REST','RECALL','LIFT')}
    anchors=sorted(result['rows'],key=lambda r:(r['floor'],r['choice_index']))
    rows=[];excluded=[]
    for index,(name,raw_floor) in enumerate(zip(names,floors,strict=True)):
        floor=integer(raw_floor);reasons=[]
        path=run['path_per_floor']
        if not 1<=floor<=len(path) or path[floor-1]!='$':reasons.append('unverified_shop_room')
        if act_for_floor(floor)>3:reasons.append('act4_outside_scope')
        if floor>FLOORS.get(act_for_floor(floor),-1):reasons.append('outside_reviewed_boss_visibility_window')
        if purge_counts[floor]!=1:reasons.append('multiple_same_floor_purges')
        if floor in purchases:reasons.append('same_floor_purchase_order_unknown')
        if floor in reward_floors|event_floors|relic_floors|fire_floors:
            reasons.append('same_floor_other_inventory_channel')
        barrier=next((b for b in result['barriers'] if b['floor']<=floor),None)
        if barrier:reasons.append('prefix_tainted:'+barrier['reason'])
        possible=[r for r in anchors if r['floor']<floor and r['act']==act_for_floor(floor)]
        anchor=possible[-1] if possible else None
        if anchor is None:reasons.append('no_verified_same_act_reward_anchor')
        if anchor is not None:
            left=anchor['floor']
            same=[i for i,c in enumerate(run['card_choices']) if integer(c['floor'])==left]
            if anchor['choice_index']!=max(same):reasons.append('anchor_not_last_recorded_floor_choice')
            channels=reward_floors|purchases|event_floors|relic_floors|fire_floors|set(purge_counts)
            if any(left<f<floor for f in channels):reasons.append('intervening_inventory_channel')
        if not reasons:
            try:
                obs=copy.deepcopy(anchor['public']['observation'])
                deck=obs['inventory']['deck'];relics=[r['id'] for r in obs['inventory']['relics']]
                if BOTTLES&set(relics):raise HistoryGap('unknown_bottled_removal_eligibility')
                picked=run['card_choices'][anchor['choice_index']]['picked']
                if picked not in NO_CARD:deck.append(obtained_card(picked,relics,catalog))
                deck.sort(key=card_key)
                groups=candidate_groups(deck,catalog)
                target=card(name,catalog)
                matches=[i for i,g in enumerate(groups) if g['card']==target]
                if len(matches)!=1:raise HistoryGap('purged_variant_missing_or_ineligible')
                if len(groups)<2:raise HistoryGap('no_nontrivial_removal_choice')
                obs['run']['floor']=floor
                # State-only encoding. Candidate references are separate decoder
                # metadata; do not invent a native menu, prices or a leave choice.
                obs['screen']={}
                public=PolicyInput(obs,())
                rows.append(dict(version=VERSION,public=public.as_dict(),input_sha256=public.fingerprint(),
                    candidates=groups,target=matches[0],partial_fields=list(PARTIAL),
                    task='which_card_after_paid_removal_committed',
                    anchor=dict(floor=left,choice_index=anchor['choice_index'],input_sha256=anchor['input_sha256']),
                    purge_index=index,floor=floor,act=act_for_floor(floor),
                    boss_context=copy.deepcopy(anchor['boss_context'])))
            except HistoryGap as error:reasons.append(str(error))
        if reasons:excluded.append(dict(purge_index=index,floor=floor,reasons=reasons))
    return dict(version=VERSION,rows=rows,excluded=excluded,purges=len(names),
                reward_prefixes=len(result['rows']),final_deck_matches=result['final_deck_matches'])


def encode_row(row,encoder):
    if (row.get('version')!=VERSION or row.get('task')!='which_card_after_paid_removal_committed' or
            row.get('partial_fields')!=list(PARTIAL)):
        raise ValueError('conditional removal row contract mismatch')
    wire=row['public'];public=PolicyInput(wire['observation'],tuple(Action.from_dict(a) for a in wire['legal_actions']))
    if (public.fingerprint()!=row['input_sha256'] or public.legal_actions or
            public.observation.get('player')!={} or public.observation.get('screen')!={}):
        raise ValueError('partial state-only removal observation required')
    if BOTTLES&{r['id'] for r in public.observation['inventory']['relics']}:
        raise ValueError('unknown bottle assignments')
    groups=candidate_groups(public.observation['inventory']['deck'],encoder.catalog)
    if groups!=row['candidates'] or type(row['target']) is not int or not 0<=row['target']<len(groups):
        raise ValueError('complete removal-equivalence groups required')
    root=capture(public,relic_order_known=False)
    encoded=tuple(encoder.encode(local_transition(root,'remove',target_index=g['indices'][0])) for g in groups)
    # Unknown decision resources and historical relic order must stay masked.
    for candidate in encoded:
        for state in (candidate.before,candidate.after):
            table=state.entities.tables['global']
            if any(table.known[0][table.columns.index(k)] for k in ('hp','max_hp','gold')):
                raise ValueError('unknown removal resources became known')
            relics=state.entities.tables['relics'];i=relics.columns.index('order')
            if any(mask[i] for mask in relics.known):raise ValueError('historical relic order leaked')
    return encoded
