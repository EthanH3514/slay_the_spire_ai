"""Validated current-event identity, including its public overlay windows."""
import copy
from .policy_api import PolicyInput, _reject_forbidden_policy_keys

VERSION = 'public-event-origin-v1'
NATIVE_VERSION = 'public-event-context-v1'
WINDOWS = {'EVENT', 'CARD_SELECT', 'CARD_REWARD', 'COMBAT_REWARD'}


def is_terminal_death(public):
    """A finished run may retain its native event/reward screen, with no menu.

    Check the terminal facts together. Zero HP alone is not permission to cross
    an unrelated screen, nor is a live Fairy/Lizard Tail revival a death exit.
    """
    obs = public.observation
    return (obs.get('in_game') is False and obs.get('run', {}).get('outcome') == 'PLAYER_LOSS'
            and obs.get('player', {}).get('hp') == 0 and not public.legal_actions)


def attach(public, context):
    _reject_forbidden_policy_keys(context, reject_seeds=True)
    if not isinstance(context, dict) or context.get('version') != NATIVE_VERSION:
        raise ValueError('public event context version mismatch')
    if type(context.get('available')) is not bool:
        raise ValueError('event context availability must be explicit')
    if not context['available']:
        if set(context) != {'version', 'available'}:
            raise ValueError('unavailable event context includes stale identity')
        return public
    if set(context) != {'version','available','event_name','act','floor'}:
        raise ValueError('unexpected public event context fields')
    obs = public.observation; screen = obs['screen']; run = obs['run']
    if ('combat' in obs or (screen['type'] not in WINDOWS and not is_terminal_death(public))
            or (context['act'],context['floor']) != (run['act'],run['floor'])
            or (run.get('room_type') != 'EVENT' and not (run['floor']==0 and context['event_name']=='NEOW'))):
        raise ValueError('event identity does not own this public window')
    name = context['event_name']
    if not isinstance(name,str) or not name:
        raise ValueError('missing public event identity')
    displayed = screen.get('decision_context',{}).get('event_name')
    if displayed is not None and displayed != name:
        raise ValueError('displayed event identity differs from public context')
    out=copy.deepcopy(public.as_dict()['observation'])
    out['screen']['event_origin']=dict(version=VERSION,event_name=name,act=run['act'],floor=run['floor'])
    return PolicyInput(out,public.legal_actions)
