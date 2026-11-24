"""Opt-in partial event commitments; never guesses an unobserved outcome."""
from dataclasses import dataclass
import json
from pathlib import Path

from .actions import Action, ActionKind
from .boss_cards import canonical_boss, FLOORS
from .conditional_removal import BOTTLES, candidate_groups as removable
from .conditional_upgrade import candidate_groups as upgradable
from .event_demonstrations import MENUS, VERSION as ROW_VERSION, encode_row
from .event_options import event_choice
from .expert_card_policy import _card
from .expert_cards import card_key, PARTIAL_FIELDS
from .policy_api import PolicyInput, _thaw_public_value
from .policy_scope import assess_policy_scope
from .rule_coverage import _validate_input
from .run_options import public_run_options, RUN_OPTIONS_VERSION

VERSION = 'free-deck-event-candidate-v1'
PUBLIC_NAMES = {value[0]: key for key, value in MENUS.items()}


def signature(contract):
    """Compare full parsed effects/costs, not labels or candidate positions."""
    return json.dumps(_thaw_public_value(contract), sort_keys=True, separators=(',', ':'))


def applicable(public):
    screen = public.observation.get('screen', {})
    if screen.get('type') != 'EVENT' or screen.get('decision_context', {}).get('event_name') not in PUBLIC_NAMES:
        return False
    # The post-choice leave screen is NOT a second commitment decision.
    options = public_run_options(public)
    return bool(options) and not all(o['kind'] == 'leave' for o in options)


@dataclass(frozen=True)
class EventProjection:
    row: dict
    encoded: object
    actions: tuple
    meanings: tuple

    def decode(self, index):
        if type(index) is not int or not 0 <= index < len(self.actions):
            raise ValueError('invalid projected event action')
        return self.actions[index]


def project(public, encoder):
    _validate_input(public)
    if not applicable(public) or 'combat' in public.observation:
        raise ValueError('reviewed initial event menu required')
    obs = public.as_dict()['observation']; run = obs['run']; inv = obs['inventory']
    if (run.get('character') != 'IRONCLAD' or type(run.get('act')) is not int or
            run['act'] not in (1, 2, 3) or type(run.get('floor')) is not int or
            not 1 <= run['floor'] <= FLOORS[run['act']] or
            type(run.get('ascension')) is not int or not 0 <= run['ascension'] <= 20):
        raise ValueError('outside reviewed event run scope')
    eligible = assess_policy_scope(public.observation, public.legal_actions).eligible_actions
    if eligible != public.legal_actions: raise ValueError('event scope changed candidate set')
    deck = sorted((_card(c, encoder.catalog) for c in inv['deck']), key=card_key)
    relics = []
    for r in inv['relics']:
        entry = encoder.catalog.lookup('relic', r.get('id'))
        if entry is None or entry['id'] in relics: raise ValueError('unknown or duplicate relic')
        relics.append(entry['id'])
    if BOTTLES & set(relics): raise ValueError('bottles outside partial event contract')
    if not removable(deck, encoder.catalog): raise ValueError('no purgeable event target')
    name = obs['screen']['decision_context']['event_name']; event = PUBLIC_NAMES[name]
    menu = MENUS[event][1]
    if event == 'Living Wall' and not upgradable(deck, encoder.catalog): menu = menu[:2]
    expected = {signature(event_choice(name, text)): label for label, text in menu}
    bound = {}
    for action, option in zip(public.legal_actions, public_run_options(public), strict=True):
        key = signature(option.get('event_choice'))
        if action.kind is not ActionKind.CHOOSE or key not in expected or expected[key] in bound:
            raise ValueError('event menu effects do not match reviewed initial menu')
        bound[expected[key]] = action
    labels = tuple(label for label, _ in menu)
    if set(bound) != set(labels): raise ValueError('incomplete initial event menu')
    actions = tuple(Action(ActionKind.CHOOSE, choice_index=i) for i in range(len(menu)))
    options = [dict(kind='event_choice', event_choice=event_choice(name, text), action=a.as_dict())
               for (_, text), a in zip(menu, actions, strict=True)]
    partial_run = {k:run[k] for k in ('character', 'act', 'floor', 'ascension')}
    boss = canonical_boss(run.get('boss'), run['act'], encoder.catalog)
    if boss is not None: partial_run['boss'] = boss
    partial = PolicyInput(dict(run=partial_run, player={},
        inventory=dict(deck=deck, relics=[dict(id=r) for r in sorted(relics)]),
        screen=dict(type='EVENT', decision_context=dict(event_name=name),
                    run_options_version=RUN_OPTIONS_VERSION, decision_options=options)), actions)
    row = dict(version=ROW_VERSION, public=partial.as_dict(), input_sha256=partial.fingerprint(),
               target=0, partial_fields=list(PARTIAL_FIELDS), relic_order_known=False)
    return EventProjection(row, encode_row(row, encoder), tuple(bound[k] for k in labels), labels)


class EventCandidatePolicy:
    def __init__(self, path, encoder):
        import torch
        from .run_learning_model import make_model
        path = Path(path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['version'] != VERSION or saved['schema'] != encoder.schema_id:
            raise ValueError('event checkpoint contract mismatch')
        self.encoder = encoder; self.model = make_model(encoder, width=saved['width'])
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval(); self.model.requires_grad_(False); self.last_decision = None

    def choose(self, public):
        import torch
        from .run_learning_model import tensorize
        p = project(public, self.encoder)
        with torch.inference_mode(): distribution, _ = self.model.distribution(tensorize([p.encoded], self.encoder))
        probs = distribution.probs[0]; index = int(probs.argmax()); action = p.decode(index)
        if action not in public.legal_actions: raise ValueError('event mapped action is not legal')
        self.last_decision = dict(version=VERSION, mode='greedy_partial_event',
            input_sha256=p.row['input_sha256'], meanings=p.meanings, probabilities=probs.tolist(), action=action.as_dict())
        return action
