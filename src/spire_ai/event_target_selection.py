"""Opt-in event target selection through the existing construction head.

The event has already committed to an operation. Rank only its observed legal
targets, from the current public state (costs already paid stay paid). Printed
selector indices are never treated as master-deck indices.
"""
from collections import Counter
from dataclasses import dataclass
import json

from .campfire_construction import tensorize
from .expert_card_policy import _card
from .expert_cards import card_key
from .outside_candidate_dispatch import OutsideCandidateDispatch
from .run_afterstates import capture, local_transition, UNREMOVABLE
from .run_learning_encoding import BOTTLES
from .run_options import public_run_options
from .policy_api import _thaw_public_value

VERSION = 'shared-construction-event-target-v2'


def task_for(public):
    obs = public.observation
    if 'combat' in obs or obs.get('run', {}).get('room_type') != 'EVENT':
        return None
    screen = obs.get('screen', {})
    if screen.get('type') != 'CARD_SELECT': return None
    context = screen.get('decision_context', {})
    task = context.get('selection_task')
    if task not in ('remove', 'upgrade'): return None
    if context.get('selected_count') != 0 or context.get('selection_count') != 1:
        raise ValueError('event target requires an observed single-card commitment')
    return task


@dataclass(frozen=True)
class Targets:
    task: str
    fingerprint: str
    candidates: tuple
    actions: tuple
    indices: tuple
    transitions: tuple
    action_targets: tuple


def public_selector(environment):
    export = getattr(environment.session, 'public_deck_selector', None)
    return json.loads(export()) if export is not None else None


def selector_bindings(selector, public, eligible, encoder, task):
    """Validate an explicit public instance map, never infer deck indices from labels."""
    if selector is None: return None
    if (set(selector) != {'version', 'available', 'task', 'selected_count', 'selection_count', 'cards'}
            or selector['version'] != 'public-deck-selector-v1' or selector['available'] is not True
            or selector['task'] != task or selector['selected_count'] != 0 or selector['selection_count'] != 1):
        raise ValueError('invalid public deck selector contract')
    expected = {i: key for key, values in eligible.items() for i, _ in values}
    deck = public.observation['inventory']['deck']; result = {}; seen = set()
    for row in selector['cards']:
        if set(row) != {'option_index', 'deck_index', 'card'}: raise ValueError('invalid selector fields')
        option, index = row['option_index'], row['deck_index']
        if (type(option) is not int or type(index) is not int or option < 0 or option in result
                or index in seen or index not in expected): raise ValueError('invalid selector instance binding')
        card = row['card']
        if (set(card) != {'id', 'upgrades', 'misc'} or card_key(_card(card, encoder.catalog)) != expected[index]
                or card.get('misc') != deck[index].get('misc', 0)):
            raise ValueError('stale public selector identity')
        result[option] = index; seen.add(index)
    if seen != set(expected) or set(result) != {a.choice_index for a in public.legal_actions}:
        raise ValueError('public selector coverage differs')
    return result


def build_targets(public, encoder, *, supplement=None, relic_order_known=True, selector=None):
    task = task_for(public)
    if task is None: raise ValueError('not a committed event removal/upgrade selector')
    root = capture(public, relic_order_known=relic_order_known, run_public=supplement)
    deck = public.observation['inventory']['deck']
    eligible = {}
    for i, card in enumerate(deck):
        key = card_key(_card(card, encoder.catalog))
        entry = encoder.catalog.lookup('card', key[0])
        if task == 'upgrade' and (entry['type'] not in ('ATTACK', 'SKILL', 'POWER') or
                                  (key[1] and key[0] != 'SEARING_BLOW')): continue
        if task == 'remove' and (key[0] in UNREMOVABLE or
                (supplement and any(supplement['deck'][i][k] for k in BOTTLES))): continue
        transition = local_transition(root, task, target_index=i, catalog=encoder.catalog)
        eligible.setdefault(key, []).append((i, transition))
    bindings = selector_bindings(selector, public, eligible, encoder, task)
    groups = {}
    for action, option in zip(public.legal_actions, public_run_options(public), strict=True):
        if option.get('kind') != 'select_card' or option.get('task') != task:
            raise ValueError('unexpected action in event target window')
        key = card_key(_card(option['card'], encoder.catalog))
        if key not in eligible: raise ValueError('legal target has no supported local transition')
        if bindings is not None and bindings[action.choice_index] not in {i for i, _ in eligible[key]}:
            raise ValueError('selector action identity differs')
        groups.setdefault(key, []).append(action)
    if not groups: raise ValueError('empty event target menu')
    candidates, actions, indices, transitions, action_targets = [], [], [], [], []
    for key, bound in sorted(groups.items()):
        possible = eligible[key]
        if len(possible) != len(bound): raise ValueError('event target multiplicity mismatch')
        # Native printed options omit persistent misc and bottle assignment.
        # Equivalent copies are interchangeable; otherwise no safe action binding.
        def identity(i):
            return json.dumps(_thaw_public_value(dict(card=deck[i],
                supplement=supplement['deck'][i] if supplement else None)),sort_keys=True)
        identities = {identity(i) for i, _ in possible}
        if len(identities) != 1 and bindings is None:
            raise ValueError('ambiguous public event target instances')
        subgroups = {}
        choices = dict(possible)
        for action in bound:
            i = bindings[action.choice_index] if bindings is not None else possible[0][0]
            subgroups.setdefault(identity(i), []).append((action, i))
        for values in subgroups.values():
            action, i = min(values, key=lambda x:json.dumps(x[0].as_dict(), sort_keys=True))
            if len(identities) == 1: i = possible[0][0]  # Preserve the existing equivalent-copy scoring.
            transition = choices[i]; target = len(candidates)
            candidates.append(encoder.encode(transition)); transitions.append(transition); indices.append(i)
            actions.append(action); action_targets.extend((a, target) for a, _ in values)
    return Targets(task, public.fingerprint(), tuple(candidates), tuple(actions), tuple(indices), tuple(transitions), tuple(action_targets))


def choose(public, construction, *, supplement=None, selector=None):
    import torch
    menu = build_targets(public, construction.encoder, supplement=supplement, selector=selector)
    with torch.inference_mode(): scores = construction.model(tensorize(menu.candidates, construction.encoder))
    if scores.shape != (len(menu.actions),) or not torch.isfinite(scores).all():
        raise ValueError('invalid event target scores')
    index = int(scores.argmax())
    return menu.actions[index], dict(controller=VERSION, mode='shared_construction_event_target',
        task=menu.task, selected=index, deck_index=menu.indices[index], probabilities=scores.softmax(0).tolist(),
        fingerprint=menu.fingerprint), menu.transitions[index]


def verify_deck(transition, after):
    # Native export adds derived rarity/type. Local edits deliberately drop stale
    # printed fields. Compare persistent identity, including known misc, instead.
    def cards(deck): return Counter((c['id'],c['upgrades'],c.get('misc')) for c in deck)
    expected=cards(transition.after.public.observation['inventory']['deck'])
    actual=cards(after.observation['inventory']['deck'])
    if expected != actual:
        raise ValueError('event target execution differs: '+str(dict(expected-actual))+' versus '+str(dict(actual-expected)))


class EventTargetDispatch(OutsideCandidateDispatch):
    """A separately selected controller; old dispatch and active runs stay pinned."""
    def step(self, env, *, reward_eligible=False, take=None):
        public = env._current_public_input()
        if task_for(public) is None: return super().step(env, reward_eligible=reward_eligible, take=take)
        if self.pending is not None and self.pending[2] != public.fingerprint():
            raise ValueError('stale event target handoff')
        supplement = json.loads(env.session.public_run_supplement())
        action, detail, transition = choose(public, self.construction, supplement=supplement, selector=public_selector(env))
        result = (take or (lambda a, d:env.step(a)))(action, detail)
        verify_deck(transition, env._current_public_input())
        self.pending = None; self.last_decision = detail
        return result
