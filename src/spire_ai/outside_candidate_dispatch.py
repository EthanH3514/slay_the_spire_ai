"""Opt-in candidate dispatch; leaves the active training controller unchanged."""
from collections import Counter, defaultdict
import json

from .actions import Action
from .construction_policy import execute
from .campfire_construction import tensorize
from .event_candidate_policy import applicable, project as project_event
from .expert_card_policy import _card
from .expert_cards import card_key
from .conditional_removal import candidate_groups as removable
from .conditional_upgrade import candidate_groups as upgradable
from .policy_api import PolicyInput, HierarchicalPolicy, _reject_forbidden_policy_keys
from .run_afterstates import capture, local_transition
from .run_options import public_run_options

VERSION = 'outside-candidate-dispatch-v1'
TARGET_TASKS = {'Forget':'remove', 'Elegance':'remove', 'Grow':'upgrade'}


def target_choice(before, selected, public, construction):
    """Rank legal targets using the SAME construction model, after commitment.

    The partial inventory projection matches conditional human construction data.
    Transform outcomes are not resolved or scored by this adapter.
    """
    task = TARGET_TASKS.get(selected)
    if task is None: raise ValueError('unsupported event target task')
    b = before.as_dict()['observation']; obs = public.as_dict()['observation']
    # Native room_phase changes EVENT -> CARD_SELECT when opening a selector.
    before_run = {k:v for k,v in b['run'].items() if k != 'room_phase'}
    after_run = {k:v for k,v in obs['run'].items() if k != 'room_phase'}
    if (before_run != after_run or b['inventory'] != obs['inventory'] or b.get('player') != obs.get('player')):
        raise ValueError('event commitment changed pre-target state')
    screen = obs['screen']; context = screen.get('decision_context', {})
    if (screen.get('type') != 'CARD_SELECT' or context.get('selection_task') != task or
            context.get('selected_count') != 0 or context.get('selection_count') != 1):
        raise ValueError('event target window/task mismatch')
    enc = construction.encoder; projection = project_event(before, enc.common.base)
    wire = projection.row['public']; partial_obs = wire['observation']; partial_obs['screen'] = {}
    root = capture(PolicyInput(partial_obs, ()), relic_order_known=False)
    deck = partial_obs['inventory']['deck']
    groups = (upgradable if task == 'upgrade' else removable)(deck, enc.catalog)
    bound = defaultdict(list)
    for action, option in zip(public.legal_actions, public_run_options(public), strict=True):
        if option['kind'] != 'select_card' or option.get('task') != task:
            raise ValueError('unexpected event target option')
        bound[card_key(_card(option['card'], enc.catalog))].append(action)
    if (set(bound) != {card_key(g['card']) for g in groups} or
            any(len(bound[card_key(g['card'])]) != len(g['indices']) for g in groups)):
        raise ValueError('event target identity/multiplicity mismatch')
    candidates = tuple(enc._wrap(enc.common.encode(local_transition(root, task, target_index=g['indices'][0]))) for g in groups)
    import torch
    with torch.inference_mode(): scores = construction.model(tensorize(candidates, enc))
    if scores.shape != (len(groups),) or not torch.isfinite(scores).all(): raise ValueError('event target scores invalid')
    index = int(scores.argmax()); target = groups[index]['card']
    action = min(bound[card_key(target)], key=lambda a:json.dumps(a.as_dict(), sort_keys=True))
    return action, dict(controller=VERSION, mode='shared_construction_event_target', task=task,
        selected_card=target, probabilities=scores.softmax(0).tolist(),
        candidates=[g['card'] for g in groups], action=action.as_dict())


def verify_target(before, after, task, target, catalog):
    def deck(public): return Counter(card_key(_card(c, catalog)) for c in public.observation['inventory']['deck'])
    expected = deck(before); key = card_key(target)
    expected[key] -= 1
    if task == 'upgrade': expected[(key[0], key[1]+1)] += 1
    if +expected != deck(after): raise ValueError('event target execution changed the wrong card')


class OutsideCandidateDispatch:
    """One explicit owner per decision. Combat remains owned by its caller.

    `base(public, supplement)` returns (action, detail). Unchanged windows use
    that same callback and RNG; candidates never consume its sampling stream.
    """
    def __init__(self, construction, base, *, event=None, boss=None, construction_select=None):
        self.construction = construction; self.base = base; self.event = event; self.boss = boss
        self.construction_select = construction_select or construction.choose
        self.pending = None; self.last_decision = None

    def step(self, env, *, reward_eligible=False, take=None):
        public = env._current_public_input()
        _reject_forbidden_policy_keys(public.observation, reject_seeds=True)
        if HierarchicalPolicy.specialist_for(public).value == 'combat':
            raise ValueError('combat must retain its original controller')
        supplement = json.loads(env.session.public_run_supplement())
        _reject_forbidden_policy_keys(supplement, reject_seeds=True)
        apply = take or (lambda a, d:env.step(a))
        if self.pending is not None:
            before, selected, fingerprint = self.pending
            if public.fingerprint() != fingerprint: raise ValueError('stale event target handoff')
            action, detail = target_choice(before, selected, public, self.construction)
            result = apply(action, detail)
            verify_target(public, env._current_public_input(), detail['task'], detail['selected_card'], self.construction.encoder.catalog)
            self.pending = None; self.last_decision = detail
            return result
        shelf = json.loads(env.session.public_shop_shelf()) if public.observation['screen']['type']=='SHOP' else None
        menu = self.construction.menu(public, supplement=supplement, shelf=shelf, reward_eligible=reward_eligible)
        if menu is not None:
            selected = self.construction_select(menu)
            self.last_decision = dict(controller=VERSION, mode='construction', menu=menu.kind, selected=selected)
            return execute(env, menu, selected, take=apply)
        screen = public.observation['screen']['type']
        if self.boss is not None and screen == 'BOSS_RELIC_REWARDS':
            action = self.boss.choose(public); self.last_decision = dict(self.boss.last_decision, controller=VERSION)
        elif self.event is not None and applicable(public):
            action = self.event.choose(public); self.last_decision = dict(self.event.last_decision, controller=VERSION)
            p = project_event(public, self.event.encoder); selected = p.meanings[p.actions.index(action)]
            result = apply(action, self.last_decision); after = env._current_public_input()
            if selected in TARGET_TASKS and after.observation['screen']['type'] == 'CARD_SELECT':
                # Validate before handing control back; no silently misrouted selector.
                target_choice(public, selected, after, self.construction)
                self.pending = (public, selected, after.fingerprint())
            return result
        else:
            action, detail = self.base(public, supplement)
            self.last_decision = dict(detail, controller=VERSION, owner='base')
        if action not in public.legal_actions: raise ValueError('dispatch returned illegal action')
        return apply(action, self.last_decision)
