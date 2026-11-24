"""Opt-in exact-entry event search, separate from the no-SL policy interface.

Only the environment adapter can clone hidden state. The evaluator receives a
public exit, never a session, audit record, seed, or future-room simulation.
No transposition table: equal public menus need not have equal hidden event state.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Callable, Protocol

from .actions import Action, ActionKind
from .environment import EnvironmentState, StepResult
from .policy_api import PolicyInput, _reject_forbidden_policy_keys
from .policy_scope import assess_policy_scope
from .potion_replacement import Replacement, plans as replacement_plans
from .local_event_context import VERSION as ORIGIN_VERSION, is_terminal_death
from .run_decision_rules import eligible_actions, LEAVE_ONLY_EVENTS, VERSION as RULES_VERSION


VERSION = "combat-nosl-local-event-rewind-v4"
# Mechanics classification, NOT a claim of complete simulator fidelity.
COMBAT_EVENTS = frozenset({"Dead Adventurer", "The Colosseum", "Masked Bandits",
    "Mindbloom", "Hypnotizing Colored Mushrooms", "Mysterious Sphere", "Secret Portal"})
MECHANISM_GAPS = {}
PURE_EVENTS = frozenset({"NEOW", "Match and Keep", "Ominous Forge", "Pleading Vagrant", "Ancient Writing",
    "Old Beggar", "Big Fish", "Bonfire Spirits", "Cursed Tome", "Designer In-Spire",
    "Augmenter", "Duplicator", "Face Trader", "Falling", "Forgotten Altar",
    "The Divine Fountain", "Council of Ghosts", "Golden Idol", "Golden Shrine",
    "Wing Statue", "Knowing Skull", "Lab", "The Ssssserpent", "Living Wall",
    "The Nest", "N'loth", "Purifier", "Scrap Ooze", "Sensory Stone", "Shining Light",
    "The Cleric", "The Joust", "The Library", "The Mausoleum", "The Moai Head",
    "The Woman in Blue", "Tomb of Lord Red Mask", "Transmorgrifier", "Upgrade Shrine",
    "Vampires(?)", "We Meet Again!", "Wheel of Change", "Winding Halls", "World of Goop"})
OWNED_SCREENS = frozenset({"EVENT", "CARD_SELECT", "CARD_REWARD", "COMBAT_REWARD"})
EVENT_ACTIONS = frozenset({ActionKind.CHOOSE, ActionKind.PROCEED, ActionKind.CANCEL})


class EventSearchError(ValueError):
    pass


class EventSearchEnvironment(Protocol):
    def clone_exact(self) -> EventSearchEnvironment: ...
    def _current_public_input(self) -> PolicyInput: ...
    def event_checkpoint(self) -> EnvironmentState: ...
    def event_supplement(self) -> Mapping: ...
    def step(self, action: Action) -> StepResult: ...
    def step_replacement(self, plan: Replacement) -> StepResult: ...


def _name(public):
    screen=public.observation.get('screen',{})
    origin=screen.get('event_origin',{})
    if origin:
        run=public.observation.get('run',{})
        displayed=screen.get('decision_context',{}).get('event_name')
        if (origin.get('version')!=ORIGIN_VERSION or not origin.get('event_name')
                or (origin.get('act'),origin.get('floor'))!=(run.get('act'),run.get('floor'))
                or (displayed is not None and displayed!=origin['event_name'])
                or (run.get('room_type')!='EVENT' and not (run.get('floor')==0 and origin['event_name']=='NEOW'))):
            raise EventSearchError('invalid_public_event_origin')
        return origin['event_name']
    return screen.get('decision_context',{}).get('event_name')


def eligibility(public: PolicyInput) -> str | None:
    """A reason means refusal, before cloning or looking at a hidden outcome."""
    _reject_forbidden_policy_keys(public.observation, reject_seeds=True)
    if "combat" in public.observation:
        return "active_combat"
    if is_terminal_death(public):
        return "terminal_event_state"
    screen=public.observation.get('screen',{})
    if screen.get('type') not in OWNED_SCREENS or (screen.get('type')!='EVENT' and not screen.get('event_origin')):
        return "event_entry_required"
    name = _name(public)
    if name in COMBAT_EVENTS:
        return "whole_event_can_enter_combat"
    if name in MECHANISM_GAPS:
        return MECHANISM_GAPS[name]
    if name not in PURE_EVENTS | LEAVE_ONLY_EVENTS:
        return "unknown_whole_event_scope"
    return None


def _combat_effect(value):
    if isinstance(value, Mapping):
        return (value.get("kind") in {"begin_combat", "travel_to_boss"}
                or any(_combat_effect(v) for v in value.values()))
    return isinstance(value, (list, tuple)) and any(_combat_effect(v) for v in value)


def _place(public):
    run = public.observation.get("run", {})
    return tuple(run.get(k) for k in ("act", "floor", "room_type"))


def _guard(public, owner, place):
    _reject_forbidden_policy_keys(public.observation, reject_seeds=True)
    obs = public.observation
    screen = obs.get("screen", {})
    if "combat" in obs or _combat_effect(screen):
        raise EventSearchError("combat_boundary_or_contract_violation")
    if _place(public) != place:
        raise EventSearchError("event_room_boundary_changed")
    if screen.get("type") not in OWNED_SCREENS | {"MAP"} and not is_terminal_death(public):
        raise EventSearchError("unowned_event_screen")
    if (screen.get("type") == "EVENT" or screen.get('event_origin')) and _name(public) != owner:
        raise EventSearchError("event_owner_changed")


@dataclass(frozen=True)
class Budget:
    max_steps: int = 4096
    max_depth: int = 64

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.max_depth)):
            raise ValueError("positive integer event budgets required")


@dataclass(frozen=True)
class PublicExit:
    public: PolicyInput
    supplement: Mapping
    dead: bool


@dataclass(frozen=True)
class PlannedStep:
    action: Action | Replacement
    before: str
    after: str


@dataclass(frozen=True)
class Plan:
    event_name: str
    path: tuple[PlannedStep, ...]
    exit: PublicExit
    score: float
    steps: int
    exits: int
    depth_cutoffs: int
    complete: bool
    scorer_version: str
    root_counts: tuple[tuple[Action | Replacement, int, int], ...]
    # Hidden RNG remains only inside EnvironmentState.audit_state. Never wire it.
    _entry: EnvironmentState = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)
    probes: tuple[PlannedStep, ...] = ()
    action_reduction: str | None = None

    def wire(self):
        return dict(version=VERSION, decision_rules_version=RULES_VERSION, event_name=self.event_name,
            path=[dict(action=_wire(s.action), before=s.before, after=s.after) for s in self.path],
            exit_fingerprint=self.exit.public.fingerprint(), dead=self.exit.dead, score=self.score,
            simulator_steps=self.steps, completed_exits=self.exits, depth_cutoffs=self.depth_cutoffs,
            complete=self.complete, scorer_version=self.scorer_version,
            probe_steps=len(self.probes), action_reduction=self.action_reduction,
            sl_probes=[dict(action=_wire(s.action),before=s.before,after=s.after) for s in self.probes],
            root_coverage=[dict(action=_wire(a), simulator_steps=n, completed_exits=e)
                           for a, n, e in self.root_counts],
            strict_no_sl=False, production_promoted=False, ppo_eligible=False)


def _actions(public):
    scope = assess_policy_scope(public.observation, public.legal_actions)
    result = tuple(a for a in scope.eligible_actions if a.kind in EVENT_ACTIONS)
    if any(a.kind not in EVENT_ACTIONS for a in scope.eligible_actions):
        raise EventSearchError("non_event_action_in_owned_menu")
    if not result:
        raise EventSearchError("event_has_no_scoped_action")
    result = eligible_actions(public, result, event_name=_name(public))
    # No bare discard: the alternative commits to both discard and acquisition
    # of an already visible potion. The unchanged executor validates both steps.
    if public.observation.get('screen',{}).get('type') == 'COMBAT_REWARD' and public.observation.get('inventory',{}).get('potions'):
        result += replacement_plans(public)
    return result


def _wire(action):
    return (dict(kind='replace_potion',plan=action.as_dict())
            if isinstance(action,Replacement) else action.as_dict())


def _cost(action):
    return 2 if isinstance(action,Replacement) else 1


def _take(env,action):
    if not isinstance(action,Replacement):
        return env.step(action)
    result=env.step_replacement(action)
    if result.info.get('potion_replacement',{}).get('primitive_steps')!=2:
        raise EventSearchError('incomplete_event_potion_replacement')
    return result


def _unchanged(env, entry):
    current = env.event_checkpoint()
    if (current.observation != entry.observation or current.legal_actions != entry.legal_actions
            or current.audit_state != entry.audit_state or current.info != entry.info):
        raise EventSearchError("stale_or_mutated_event_entry")


def search(env: EventSearchEnvironment, scorer: Callable[[PublicExit], float], *,
           scorer_version: str, budget: Budget = Budget()) -> Plan:
    """Depth-first branch enumeration with bounded memory and no seed resampling.

    Incomplete results choose only among completed exits. A cut-off menu is never
    scored as a completed event. Equal scores keep the shorter path. Neither
    exhaustive local enumeration nor the evaluator promises whole-run optimality.
    """
    entry = env.event_checkpoint()
    public = env._current_public_input()
    if reason := eligibility(public):
        raise EventSearchError(reason)
    if not scorer_version:
        raise ValueError("explicit evaluator provenance required")
    owner, place = _name(public), _place(public)
    _guard(public, owner, place)  # Including nested combat effects, BEFORE clone.
    cards = None; probes = (); reduction = None
    if owner == 'Match and Keep':
        from .match_and_keep import learn, candidates, REDUCTION
        cards, probes = learn(env, budget); reduction = REDUCTION
    def choices(p, padding=False):
        if cards is not None:return candidates(p,cards,padding)
        return tuple((a,False) for a in _actions(p))
    root = env.clone_exact()
    if root is env:
        raise EventSearchError("clone_aliases_live_environment")
    # Each frame owns ONE parent clone and a lazy iterator; siblings aren't kept.
    root_actions = tuple(a for a,_ in choices(public))
    counts = {a: [0, 0] for a in root_actions}
    stack = [(root, (), iter(choices(public)))]
    steps = len(probes); exits = cutoffs = 0
    budget_skipped = False
    best = None
    while stack and steps < budget.max_steps:
        parent, path, actions = stack[-1]
        candidate = next(actions, None)
        if candidate is None:
            stack.pop()
            continue
        action, padding = candidate
        cost=_cost(action)
        if steps+cost > budget.max_steps:
            budget_skipped=True
            continue
        depth=sum(_cost(s.action) for s in path)+cost
        if depth > budget.max_depth:
            cutoffs+=1
            continue
        before = parent._current_public_input()
        branch = parent.clone_exact()
        if branch is parent or branch is env:
            raise EventSearchError("clone_aliases_parent_environment")
        result = _take(branch,action)
        steps += cost
        root_action = path[0].action if path else action
        counts[root_action][0] += cost
        after = branch._current_public_input()
        _guard(after, owner, place)
        if result.truncated:
            raise EventSearchError("truncated_event_branch")
        if is_terminal_death(after) and not result.terminated:
            raise EventSearchError("event_death_transition_not_terminated")
        continuation = path + (PlannedStep(action, before.fingerprint(), after.fingerprint()),)
        dead = after.observation.get("player", {}).get("hp", 0) <= 0
        if result.terminated and not dead:
            raise EventSearchError("unexplained_event_terminal")
        if dead or after.observation["screen"]["type"] == "MAP":
            supplement = branch.event_supplement()
            _reject_forbidden_policy_keys(supplement, reject_seeds=True)
            outcome = PublicExit(after, supplement, dead)
            # Failure never benefits from residual resources or a critic error.
            score = 0.0 if dead else float(scorer(outcome))
            if not math.isfinite(score):
                raise EventSearchError("nonfinite_event_exit_score")
            exits += 1
            counts[root_action][1] += 1
            rank = (not dead, score, -depth)
            if best is None or rank > best[0]:
                best = (rank, continuation, outcome, score)
        elif depth >= budget.max_depth:
            cutoffs += 1
        else:
            stack.append((branch, continuation, iter(choices(after,padding))))
    # Drain exhausted iterators to distinguish an exact budget boundary from a
    # genuinely unvisited branch. This does not execute extra simulator steps.
    while stack:
        if next(stack[-1][2], None) is not None:
            break
        stack.pop()
    _unchanged(env, entry)
    if best is None:
        raise EventSearchError("budget_completed_no_event_exit")
    return Plan(owner, best[1], best[2], best[3], steps, exits, cutoffs,
                not stack and cutoffs == 0 and not budget_skipped, scorer_version,
                tuple((a, *n) for a, n in counts.items()), entry, env, probes, reduction)


def replay(env: EventSearchEnvironment, plan: Plan, *, take=None) -> StepResult:
    """Preflight on a clone, then commit exactly the selected event path.

    A mismatch stops explicitly. No automatic reset, alternate branch or fallback
    is applied to the live game. Callers must record this separate ruleset.
    """
    if plan._owner is not env or not plan.path:
        raise EventSearchError("plan_is_bound_to_another_entry")
    _unchanged(env, plan._entry)
    place = _place(env._current_public_input())

    def walk(target):
        result = None
        for step in plan.path:
            p = target._current_public_input()
            _guard(p, plan.event_name, place)
            if p.observation['screen']['type'] == 'MAP' or p.observation['player']['hp'] <= 0:
                raise EventSearchError('event_replay_continues_past_exit')
            if p.fingerprint() != step.before or step.action not in _actions(p):
                raise EventSearchError("event_replay_before_mismatch")
            # Live execution may journal each committed transition. Probes and
            # preflight stay on clones and never become training trajectories.
            result = take(step.action) if target is env and take is not None else _take(target,step.action)
            q = target._current_public_input()
            _guard(q, plan.event_name, place)
            if q.fingerprint() != step.after or result.truncated:
                raise EventSearchError("event_replay_after_mismatch")
        if target.event_supplement() != plan.exit.supplement:
            raise EventSearchError("event_replay_supplement_mismatch")
        return result

    check = env.clone_exact()
    if check is env:
        raise EventSearchError("clone_aliases_live_environment")
    walk(check)
    _unchanged(env, plan._entry)
    result = walk(env)
    from .simulator_environment import _reward_components
    reward = _reward_components(plan._entry.observation,
        env._current_public_input().observation, result.terminated)
    return replace(result, reward=reward, info=dict(result.info,
        local_event_search=plan.wire(), committed_primitive_steps=sum(_cost(s.action) for s in plan.path)))
