"""Inference helpers from duplicate_selection_options.py."""

import re
from spire_ai.run_options import native_run_options as original
VERSION='public-duplicate-as-obtain-copy-v1'


def native_run_options(observation,options,text):
    if (observation.get('screen',{}).get('type')!='CARD_SELECT'
            or not re.search(r'^Select Card to Duplicate\. \(0/1\)$',text,re.M)):
        return original(observation,options,text)
    result=original(observation,options,text.replace('Select Card to Duplicate.', 'Select Card to Obtain.',1))
    deck=observation['inventory']['deck']
    rows=result['decision_options']
    if len(rows)!=len(deck):raise ValueError('duplicate candidates must cover the public deck')
    for i,(row,card) in enumerate(zip(rows,deck)):
        if (row['action']!={'kind':'choose','choice_index':i} or row['kind']!='select_card'
                or row['card']['id']!=card['id'] or row['card']['upgrades']!=card.get('upgrades',0)):
            raise ValueError('duplicate candidate/deck identity mismatch')
        row.update(card=dict(card),source_card_index=i,operation='duplicate')
    result['decision_context'].update(source_operation='duplicate',adapter_version=VERSION)
    return result


def install():
    import spire_ai.run_options as module
    module.native_run_options=native_run_options
