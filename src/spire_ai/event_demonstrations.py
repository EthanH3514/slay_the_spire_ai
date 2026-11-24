"""Partial, pre-event commitments for reviewed free deck-edit menus.

This selects operation types, not hidden transform results or card target
instances. No resource-dependent menu is guessed from its observed outcome.
"""
from collections import Counter
import copy

from .actions import Action,ActionKind
from .boss_cards import reconstruct_win,FLOORS
from .boss_relic_demonstrations import offer,BUILD
from .conditional_removal import candidate_groups as removable, BOTTLES
from .conditional_upgrade import candidate_groups as upgradable, target_index
from . import expert_cards as legacy
from .expert_cards import HistoryGap,integer,act_for_floor,NO_CARD,obtained_card,card_key,card
from .event_options import event_choice
from .policy_api import PolicyInput
from .rule_catalog import RuleCatalog
from .run_options import RUN_OPTIONS_VERSION

VERSION='human-free-deck-event-commitment-v1'
MENUS={
    'Living Wall':('Living Wall',(('Forget','[Forget] Remove a card.'),
        ('Change','[Change] Transform a card.'),('Grow','[Grow] Upgrade a card.'))),
    'Back to Basics':('Ancient Writing',(('Elegance','[Elegance] Remove a card.'),
        ('Simplicity','[Simplicity] Upgrade all Strikes and Defends.'))),
}


def check_label(event,deck,catalog):
    label=event.get('player_choice')
    fields={k:list(event.get(k,[])) for k in ('cards_removed','cards_transformed','cards_upgraded','cards_obtained')}
    if label in ('Forget','Elegance'):
        field='cards_removed';groups=removable(deck,catalog)
    elif label=='Change':field='cards_transformed';groups=removable(deck,catalog)
    elif label=='Grow':field='cards_upgraded';groups=upgradable(deck,catalog)
    elif label=='Simplicity':
        expected=Counter(c['id'] for c in deck if c['id'] in ('STRIKE_RED','DEFEND_RED') and c['upgrades']==0)
        actual=Counter(card(n,catalog)['id'] for n in fields['cards_upgraded'])
        if expected!=actual or any(fields[k] for k in fields if k!='cards_upgraded'):
            raise HistoryGap('event_upgrade_set_record_mismatch')
        return
    else:raise HistoryGap('unknown_event_commitment')
    if len(fields[field])!=1:raise HistoryGap('event_target_record_missing_or_multiple')
    name=fields[field][0];identifier=card(name,catalog)['id']
    if identifier not in {g['card']['id'] for g in groups}:raise HistoryGap('event_target_not_in_prechoice_deck')
    if label=='Grow':target_index(name,groups,catalog)
    if label=='Change':
        if len(fields['cards_obtained'])!=1:raise HistoryGap('missing_transform_result_audit')
        card(fields['cards_obtained'][0],catalog)  # Audit only, never an input.
    expected_fields={field}|({'cards_obtained'} if label=='Change' else set())
    if any(v for k,v in fields.items() if k not in expected_fields):raise HistoryGap('unexpected_event_inventory_effect')


def extract(run,*,catalog=None):
    catalog=catalog or RuleCatalog.bundled()
    if run.get('build_version')!=BUILD:raise HistoryGap('unreviewed_event_metric_build')
    selected=[(i,e) for i,e in enumerate(run.get('event_choices',[])) if e.get('event_name') in MENUS]
    if not selected:return dict(rows=[],excluded=[],reviewed_events=0)
    adapted=copy.deepcopy(run)
    for r in adapted.get('boss_relics',[]):
        try:_,chosen=offer(r,catalog)
        except HistoryGap:continue
        if chosen is None:r['picked']='SKIP'
    replay=reconstruct_win(adapted,catalog=catalog)
    if replay['unexplained_final_mismatch'] or (replay['final_deck_matches'] is None and not replay['barriers']):
        raise HistoryGap('unverified_final_inventory_audit')
    if replay['barriers'] and replay['barriers'][0]['reason'].startswith(('remove_missing_card:','ambiguous_or_missing_upgrade:')):
        raise HistoryGap('unlocalized_inventory_contradiction')
    rewards={integer(c['floor']) for c in run['card_choices']}
    other={integer(f) for f in run['item_purchase_floors']+run['items_purged_floors']}
    other|={integer(r['floor']) for r in run['relics_obtained']}
    other|={integer(f['floor']) for f in run['campfire_choices'] if f['key'] not in ('REST','RECALL','LIFT')}
    counts=Counter(integer(e['floor']) for e in run['event_choices'])
    channels=rewards|other|set(counts)|{17,34}
    rows=[];excluded=[]
    for index,event in selected:
        floor=integer(event['floor']);act=act_for_floor(floor)
        try:
            if act>3 or floor>FLOORS[act]:raise HistoryGap('outside_reviewed_event_act')
            if floor<1 or len(run['path_per_floor'])<floor or run['path_per_floor'][floor-1]!='?':
                raise HistoryGap('unverified_event_room')
            if counts[floor]!=1 or floor in rewards|other:raise HistoryGap('same_floor_event_order_unknown')
            if any(b['floor']<floor for b in replay['barriers']):raise HistoryGap('tainted_pre_event_inventory')
            anchors=[r for r in replay['rows'] if r['floor']<floor and r['act']==act]
            if not anchors:raise HistoryGap('no_same_act_pre_event_anchor')
            anchor=max(anchors,key=lambda r:(r['floor'],r['choice_index']));left=anchor['floor']
            if anchor['choice_index']!=max(i for i,c in enumerate(run['card_choices']) if integer(c['floor'])==left):
                raise HistoryGap('anchor_not_last_floor_choice')
            if any(left<f<floor for f in channels):raise HistoryGap('intervening_inventory_channel')
            obs=copy.deepcopy(anchor['public']['observation']);deck=obs['inventory']['deck']
            relics={r['id'] for r in obs['inventory']['relics']}
            if relics&BOTTLES:raise HistoryGap('unknown_bottled_event_targets')
            picked=run['card_choices'][anchor['choice_index']]['picked']
            if picked not in NO_CARD:deck.append(obtained_card(picked,relics,catalog))
            deck.sort(key=card_key)
            if not removable(deck,catalog):raise HistoryGap('no_reviewed_purgeable_target')
            name,menu=MENUS[event['event_name']]
            if event['event_name']=='Living Wall' and not upgradable(deck,catalog):menu=menu[:2]
            labels=[label for label,_ in menu];label=event.get('player_choice')
            if label not in labels:raise HistoryGap('logged_event_choice_not_legal')
            check_label(event,deck,catalog)
            actions=tuple(Action(ActionKind.CHOOSE,choice_index=i) for i in range(len(menu)))
            options=[dict(kind='event_choice',event_choice=event_choice(name,description),action=a.as_dict())
                     for (_,description),a in zip(menu,actions,strict=True)]
            if any(o['event_choice']['status']!='parsed' for o in options):raise ValueError('event effect encoding missing')
            obs['run']['floor']=floor;obs['player']={}
            obs['screen']=dict(type='EVENT',decision_context=dict(event_name=name),
                               run_options_version=RUN_OPTIONS_VERSION,decision_options=options)
            public=PolicyInput(obs,actions)
            rows.append(dict(version=VERSION,public=public.as_dict(),input_sha256=public.fingerprint(),
                target=labels.index(label),label=label,event_index=index,event_name=event['event_name'],
                floor=floor,act=act,partial_fields=list(legacy.PARTIAL_FIELDS),relic_order_known=False,
                limitations=['operation commitment only; target choice separate','resources unknown',
                    'random transform result excluded','no leave option at these initial menus']))
        except (HistoryGap,ValueError,KeyError,TypeError) as e:
            excluded.append(dict(event_index=index,event_name=event['event_name'],floor=floor,reason=str(e)))
    return dict(rows=rows,excluded=excluded,reviewed_events=len(selected))


def encode_row(row,encoder):
    if row.get('version')!=VERSION or row['public']['observation']['screen']['type']!='EVENT':
        raise ValueError('event demonstration contract mismatch')
    return legacy.encode_row(dict(row,version=legacy.VERSION),encoder)
