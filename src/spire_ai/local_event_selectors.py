"""Candidate-only public selector contracts, preserving frozen model schemas."""
import copy
import re
from .contextual_event_options import public_card
from .policy_api import PolicyInput,_reject_forbidden_policy_keys
from .rule_catalog import RuleCatalog
from .run_options import public_run_options

VERSION='local-event-deck-copy-selector-v1'


def enrich(public,text,*,catalog=None):
    _reject_forbidden_policy_keys(public.observation,reject_seeds=True)
    obs=public.observation;screen=obs.get('screen',{})
    if screen.get('type')!='CARD_SELECT' or not screen.get('event_origin'):
        return public
    match=re.search(r'^Select Card to Duplicate\. \((\d+)/(\d+)\)$',text,re.M)
    if match is None:return public
    selected,count=map(int,match.groups())
    if not 0<=selected<count:raise ValueError('invalid duplicate selector progress')
    rows=public_run_options(public);labels=screen.get('choices',())
    if len(rows)!=len(labels):raise ValueError('unaligned duplicate selector labels')
    out=copy.deepcopy(public.as_dict()['observation']);target=out['screen']
    # Both native DUPLICATE and OBTAIN acquire the selected Card value, retaining
    # upgrades/misc. Reuse the existing obtain primitive; preserve provenance.
    target['decision_context'].update(selection_task='obtain',selection_source='deck_copy',
        selected_count=selected,selection_count=count)
    catalogue=catalog or RuleCatalog.bundled()
    for row,label in zip(target['decision_options'],labels,strict=True):
        card=public_card(label,obs['inventory'],catalogue)
        if card is None:
            row.update(kind='unknown',unknown_reason='ambiguous_public_deck_copy')
            row.pop('card',None);row.pop('task',None)
            continue
        row.update(kind='select_card',task='obtain',card=card,selection_source='deck_copy')
        row.pop('unknown_reason',None)
    target['local_event_selector_version']=VERSION
    return PolicyInput(out,public.legal_actions)
