"""Verify a local selected target plus publicly offered queued effects."""
from .event_target_selection import verify_deck
from .run_options import public_run_options
from .run_afterstates import capture,local_transition


def followups(public,action):
    if public.observation.get('screen',{}).get('type')!='EVENT':return ()
    option=public_run_options(public)[public.legal_actions.index(action)]
    ops=option.get('event_choice',{}).get('display_order',())
    for i,op in enumerate(ops):
        if op.get('kind') in ('select_remove','select_upgrade'):
            return tuple(dict(x) for x in ops[i+1:])
    return ()


def verify_followups(transition,after,tail,*,catalog):
    if not tail:return verify_deck(transition,after)
    if len(tail)!=1 or tail[0]!=dict(kind='random_upgrade',count=1):
        raise ValueError('unreviewed effect after event target')
    # The chosen identity is random; accept any one legal remaining upgrade,
    # including Searing Blow, without consulting the actual RNG or picked index.
    candidates=[]
    # Reuse the local primitive on the intermediate public inventory solely to
    # verify settlement. This is not an extra player decision or model input.
    intermediate=capture(transition.after.public,relic_order_known=transition.after.relic_order_known,
                         run_public=transition.after.run_public)
    for i,card in enumerate(transition.after.public.observation['inventory']['deck']):
        entry=catalog.lookup('card',card['id'])
        if entry and entry['type'] in ('ATTACK','SKILL','POWER') and (not card['upgrades'] or entry['id']=='SEARING_BLOW'):
            candidates.append(local_transition(intermediate,'upgrade',target_index=i,catalog=catalog))
    if not candidates:return verify_deck(transition,after)
    for candidate in candidates:
        try:verify_deck(candidate,after)
        except ValueError:continue
        return
    raise ValueError('event target plus queued random upgrade differs')
