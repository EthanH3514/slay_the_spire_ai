"""Inference helpers from terminal_proof.py."""

from allocation_probe import validate_native as validate_parent
from spire_ai.actions import Action
ABI='public-terminal-proof-v1'


MODES=('control','proof')


def validate_native(native,legal,mode):
    if mode not in MODES:raise ValueError('undeclared proof mode')
    p=native.get('terminal_proof')
    if mode=='control':
        if p is not None:raise ValueError('disabled proof produced a receipt')
        return validate_parent(native,legal,'control')
    if not isinstance(p,dict) or p.get('version')!=ABI or p.get('extra_simulator_steps')!=0:
        raise ValueError('proof receipt missing or additional simulations')
    # Old strict oracle still checks every original root statistic/work counter.
    # Only its input action is projected back to the independently checked
    # visit-first recommendation; no scores/counts are rewritten.
    statistical=validate_parent(dict(native,action=p['statistical_action']),legal,'control').as_dict()
    if p['selected_action']!=native['action'] or p['changed']!=(p['used'] and native['action']!=statistical):
        raise ValueError('proof selected action mismatch')
    if p['used']:
        w=p['witness']
        if p['bound_gap'] or w['version']!=ABI or w['outcome']!=1 or not w['steps']:
            raise ValueError('unbounded/incomplete certificate selected')
        if w['steps'][0]['action']!=native['action'] or native['action'] not in legal:
            raise ValueError('proof first action illegal')
        root=w['steps'][0]['public'];end=w['terminal']
        if (end['player']['current_hp']!=root['player']['current_hp'] or end['player']['current_hp']<=0 or
            end['player']['max_hp']!=root['player']['max_hp'] or end['potions']!=root['potions']):
            raise ValueError('certificate does not reach the resource bound')
        if any(step['action']['kind']!='play_card' for step in w['steps']):raise ValueError('proof fragment changed')
    elif native['action']!=statistical or p['witness'] is not None:
        raise ValueError('unproved action override')
    return Action.from_dict(native['action'])
