"""Opt-in event contracts bound only to displayed offers and public inventory.

Old fully parsed menus stay byte-for-byte unchanged. This extends the existing
effect vocabulary and encoder, not the frozen production observation contract.
"""
import re
from functools import lru_cache
from .event_options import event_choice as legacy_choice,effect,_clause
from .policy_api import PolicyInput,_reject_forbidden_policy_keys
from .rule_catalog import RuleCatalog
from .rule_coverage import _validate_input
from .run_options import public_run_options
from .run_learning_encoding import _copy_public

VERSION='public-contextual-event-options-v1'
FACES=('CULTIST_HEADPIECE','FACE_OF_CLERIC','GREMLIN_VISAGE','NLOTHS_HUNGRY_FACE','SSSERPENT_HEAD')


@lru_cache(maxsize=1)
def _catalog():
    # RuleCatalog owns immutable records. Many-choice events must not reload
    # and recursively freeze the entire bundled catalogue for every option.
    return RuleCatalog.bundled()


def public_card(name,inventory,catalog):
    """Resolve basic names/upgrades against visible copies, never a hidden index."""
    matches={}
    for card in inventory.get('deck',()):
        entry=catalog.lookup('card',card.get('id'))
        if entry is None:continue
        upgrades=card.get('upgrades',0)
        labels={entry['name']+('+' if upgrades else ''),entry['id']+('+' if upgrades else '')}
        if upgrades>1:labels.add(entry['name']+'+'+str(upgrades))
        if name in labels:
            key=(entry['id'],upgrades,card.get('misc'))
            matches[key]=dict(id=entry['id'],upgrades=upgrades,**({'misc':card['misc']} if 'misc' in card else {}))
    return next(iter(matches.values())) if len(matches)==1 else None


def public_relic(name,inventory,catalog):
    offered=catalog.lookup('relic',name)
    if offered is None:return None
    held=[r for r in inventory.get('relics',()) if (item:=catalog.lookup('relic',r.get('id'))) and item['id']==offered['id']]
    return offered['id'] if len(held)==1 else None


def face_effects(inventory,catalog):
    owned=set()
    for r in inventory.get('relics',()):
        item=catalog.lookup('relic',r.get('id'))
        if item is None:raise ValueError('unknown public face relic')
        owned.add(item['id'])
    pool=[r for r in FACES if r not in owned] or ['CIRCLET']
    # Original getRandomFace excludes owned faces, shuffles and takes one.
    # The advertised good/bad 50/50 is not used as the outcome distribution.
    return [effect('displayed_chance',numerator=1,denominator=len(pool),
        outcome=effect('obtain_relic',id=r,count=1),exclusive_group='face_reward',
        source='reviewed_face_pool_game_event_law') for r in pool]


def event_choice(event_name,label,observation,*,catalog=None):
    _reject_forbidden_policy_keys(observation,reject_seeds=True)
    catalog=catalog or _catalog();inventory=observation.get('inventory',{})
    old=legacy_choice(event_name,label)
    # Resolve every displayed card loss, including upgraded names. A known ID
    # alone must not hide its visible upgrade/misc information from the model.
    match=re.fullmatch(r'\[([^\]]+)\]\s*(.*)',label)
    if match is None or not event_name:return old
    option,body=match.groups();candidate=None;ops=[];unknown=[]
    needs_card=event_name=='Falling' and option in ('Land','Channel','Strike')
    if event_name=='Face Trader' and option=='Trade' and body=='50%: Good Face. 50%: Bad Face.':
        ops=face_effects(inventory,catalog)
    else:
        body=body.replace(', Become Cursed','. Become Cursed')
        for text in re.split(r'\.(?:\s+|$)',body):
            text=text.strip()
            if not text:continue
            if text=='Remove a card, then upgrade a random card':
                ops.extend([effect('select_remove',count=1),effect('random_upgrade',count=1)])
            elif text=='Lose this relic' and option.startswith('Offer '):
                item=public_relic(option[6:],inventory,catalog)
                if item:ops.append(effect('lose_relic',id=item,count=1))
                else:unknown.append('unresolved_public_offered_relic')
            elif text=="Obtain N'loth's Gift" and catalog.lookup('relic','NLOTHS_GIFT'):
                ops.append(effect('obtain_relic',id='NLOTHS_GIFT',count=1))
            elif text.startswith('Lose ') and (card:=public_card(text[5:],inventory,catalog)):
                ops.append(effect('lose_card',count=1,**card));candidate=card
            elif needs_card and text.startswith('Lose '):
                unknown.append('unresolved_public_card:'+text[5:])
            elif text=='Lose some gold':
                ops.append(effect('lose_gold',amount_known=False,amount_source='unspecified_in_old_display'))
            elif text=='Lose nothing?' and event_name=='Falling' and option=='Splat':
                ops.append(effect('leave'))
            else:
                parsed=_clause(text)
                if parsed is None:unknown.append(text)
                else:ops.extend(parsed)
        if not body:return old
    if old['status']=='parsed' and candidate is None and not needs_card:return old
    if event_name=='We Meet Again!' and not unknown and ops and all(op['kind'] in ('lose_card','lose_potion','lose_gold') for op in ops):
        ops.append(effect('random_relic',rarity='unspecified',count=1,source='reviewed_native_donation_rule'))
    cost=lambda op:op['kind'].startswith('lose_') or op['kind'] in {'damage','damage_fraction','add_curse','random_curse','disable_healing'}
    result=dict(version=VERSION,event_name=event_name,status='parsed' if ops and not unknown else 'partial' if ops else 'unknown',
        fidelity='display_contract_not_effect_certification',display_order=ops,
        costs=[op for op in ops if cost(op)],effects=[op for op in ops if not cost(op)],unknown_clauses=unknown)
    if candidate is not None:result['public_candidate_card']=candidate
    return result


def enrich(public,*,catalog=None):
    """For live simulator EVENT windows only; choices must match legal order."""
    _validate_input(public);obs=public.observation;screen=obs.get('screen',{})
    if screen.get('type')!='EVENT' or 'combat' in obs:return public
    # Tile actions are decoded by the dedicated public-board adapter. Running
    # every pair through the prose/card-offer parser is both semantically wrong
    # and repeats whole-observation validation for up to66 unrelated choices.
    if screen.get('decision_context',{}).get('event_name')=='Match and Keep':return public
    options=public_run_options(public);labels=screen.get('choices')
    if not isinstance(labels,(list,tuple)) or len(labels)!=len(options):
        raise ValueError('contextual event requires aligned public choice labels')
    name=screen.get('decision_context',{}).get('event_name');out=_copy_public(obs);changed=False
    for option,old,label in zip(out['screen']['decision_options'],options,labels,strict=True):
        parsed=event_choice(name,label,obs,catalog=catalog)
        if parsed==old.get('event_choice'):continue
        option['event_choice']=parsed
        if parsed['status']=='parsed' and option['kind']=='unknown':option['kind']='event_choice';option.pop('unknown_reason',None)
        if card:=parsed.get('public_candidate_card'):option['card']=dict(card)
        changed=True
    if not changed:return public
    out['screen']['contextual_event_version']=VERSION
    return PolicyInput(out,public.legal_actions)
