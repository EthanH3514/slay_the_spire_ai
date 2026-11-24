from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

from .actions import Action, ActionKind
from .card_recipes import DEFAULT_CARD_RECIPES, CardRecipeCatalog
from .combat_sampling import (
    CARD_HISTORY_VERSION, CARD_SAMPLING_STATUS, RESAMPLER_VERSION, CombatOutcome,
    CombatSampler, UnsupportedCombatSamplingError, public_fingerprint,
)
from .environment import EnvironmentError, InvalidActionError
from .dynamic_damage import dynamic_state_reason
from .play_triggers import play_trigger_state_reason
from .pile_retrieval import RETRIEVAL_TASKS
from .selected_copy import pending_copy_count
from .top_card_autoplay import autoplay_support_reason
from .random_generation import generation_support_reason
from .identifiers import canonical_identifier
from .mechanism_evidence import MECHANISM_EVIDENCE_VERSION, EvidenceCapability
from .policy_api import PolicyInput, hidden_enemy_intents
from .policy_scope import ScopeDecision, assess_policy_scope
from .rule_catalog import RuleCatalog
from .rule_coverage import RuleCoverage, _validate_input
from .combat_selections import SELECTION_TASKS
from .event_sampling import STRICT_ROOT_MODEL, EVENT_ROOT_MODEL, validate_event_result


SEARCH_VERSION = "combat-flat-rollout-v3-potion-aware"
# Common-random-number namespace is deliberately independent of policy versions.
# Retain v2 worlds for unchanged public roots when comparing this action expansion.
SEARCH_SAMPLING_NAMESPACE = "combat-flat-rollout-v2-risk-approx"
RULE_VERSION = "combat-recipe-rule-v56"
VALUE_VERSION = "combat-local-heuristic-v3-escape"
RANKING_VERSION = "combat-risk-lexicographic-v3-resources"


@dataclass(frozen=True, slots=True)
class SearchBudget:
    samples_per_action: int = 2
    rollout_horizon: int = 48
    max_branch_steps: int = 512
    max_samples: int = 256
    max_value_evaluations: int = 256
    max_seconds: float | None = 2.0

    def __post_init__(self) -> None:
        for name in (
            "samples_per_action", "rollout_horizon", "max_branch_steps",
            "max_samples", "max_value_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_seconds is not None and (
            isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float))
            or not math.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive, or None")


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    action: Action
    samples: int
    mean_value: float
    mean_ending_hp: float
    victories: int
    losses: int
    truncated: int
    victory_rate: float
    loss_rate: float
    truncation_rate: float
    minimum_ending_hp: int
    p10_ending_hp: float
    worst_20pct_mean_ending_hp: float
    ending_hp_histogram: tuple[tuple[int, int], ...]
    mean_resource_hp: float | None = None
    p10_resource_hp: float | None = None
    worst_20pct_resource_hp: float | None = None
    escapes: int = 0


@dataclass(slots=True)
class _Accumulator:
    samples: int = 0
    value_sum: float = 0.0
    hp_sum: int = 0
    victories: int = 0
    losses: int = 0
    truncated: int = 0
    ending_hps: list[int] = field(default_factory=list)
    resource_hps: list[float] = field(default_factory=list)
    escapes: int = 0

    def add(self, score: float, hp: int, outcome: CombatOutcome, resource_hp: float | None = None) -> None:
        self.samples += 1
        self.value_sum += score
        self.hp_sum += hp
        self.victories += outcome is CombatOutcome.VICTORY
        self.losses += outcome is CombatOutcome.LOSS
        self.truncated += outcome is CombatOutcome.ONGOING
        self.ending_hps.append(hp)
        self.resource_hps.append(hp if resource_hp is None else resource_hp)
        self.escapes += outcome is CombatOutcome.ESCAPED


@dataclass(frozen=True, slots=True)
class SearchReport:
    action: Action
    root_fingerprint: str
    stop_reason: str
    fallback_reason: str | None
    completed_rounds: int
    samples: int
    branch_steps: int
    value_evaluations: int
    rollout_policy_evaluations: int
    discarded_rollouts: int
    elapsed_seconds: float
    peak_live_branches: int
    excluded_potion_actions: int
    round_horizons: tuple[int, ...]
    candidates: tuple[CandidateEstimate, ...]
    budget: SearchBudget
    search_version: str = SEARCH_VERSION
    resampler_version: str = RESAMPLER_VERSION
    card_history_version: str = CARD_HISTORY_VERSION
    rollout_version: str = RULE_VERSION
    value_version: str = VALUE_VERSION
    ranking_version: str = RANKING_VERSION
    mechanism_evidence_version: str = MECHANISM_EVIDENCE_VERSION
    tree_nodes: int = 0
    model_evaluations: int = 0
    top_tie_count: int = 0
    tie_breaker: str | None = None
    tie_changed_action: bool = False

    def as_dict(self) -> dict:
        result = asdict(self)
        result["action"] = self.action.as_dict()
        for serialized, estimate in zip(result["candidates"], self.candidates):
            serialized["action"] = estimate.action.as_dict()
        return result

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _power(entity: Mapping, name: str) -> int:
    canonical_name = canonical_identifier(name, "power")
    for power in entity.get("powers", ()):
        if isinstance(power, Mapping) and canonical_identifier(power.get("id", ""), "power") == canonical_name:
            return int(power.get("amount", 0))
    return 0


def _player(public: PolicyInput) -> Mapping:
    combat = _mapping(public.observation.get("combat"))
    return _mapping(combat.get("player")) or _mapping(public.observation.get("player"))


def ending_hp(public: PolicyInput) -> int:
    player = _player(public)
    return int(player.get("current_hp", player.get("hp", 0)))


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _lower_tail_mean(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:count]) / count


def candidate_estimates(totals: Mapping[Action, _Accumulator]) -> tuple[CandidateEstimate, ...]:
    """Shared result aggregation for simulator and live native transports."""
    return tuple(CandidateEstimate(
        action=action, samples=rows.samples,
        mean_value=rows.value_sum / rows.samples,
        mean_ending_hp=rows.hp_sum / rows.samples,
        victories=rows.victories, losses=rows.losses, truncated=rows.truncated,
        victory_rate=rows.victories / rows.samples,
        loss_rate=rows.losses / rows.samples,
        truncation_rate=rows.truncated / rows.samples,
        minimum_ending_hp=min(rows.ending_hps),
        p10_ending_hp=_quantile(rows.ending_hps, 0.1),
        worst_20pct_mean_ending_hp=_lower_tail_mean(rows.ending_hps, 0.2),
        ending_hp_histogram=tuple(sorted(Counter(rows.ending_hps).items())),
        mean_resource_hp=sum(rows.resource_hps) / rows.samples,
        p10_resource_hp=_quantile(rows.resource_hps, 0.1),
        worst_20pct_resource_hp=_lower_tail_mean(rows.resource_hps, 0.2),
        escapes=rows.escapes,
    ) for action, rows in totals.items() if rows.samples)


class StarterRulePolicy:
    """Compatibility name for the native-only public combat rule controller.

    Python owns validation/transport, never card scoring. Custom Python recipes
    belong in the frozen offline oracle, not a production fallback.
    """

    def __init__(self, catalog: RuleCatalog | None = None,
                 recipes: CardRecipeCatalog | None = None) -> None:
        self.catalog = catalog or RuleCatalog.bundled()
        self.recipes = recipes or DEFAULT_CARD_RECIPES
        from .native_search import compatible_rule
        if not compatible_rule(self):
            raise ValueError("production combat requires bundled native rules; custom Python rules are offline-only")
        self.last_scope_decision: ScopeDecision | None = None
        self.last_backend: dict = {}

    def choose(self, public: PolicyInput) -> Action:
        _validate_input(public)
        self.last_backend = {"engine": "cpp", "status": "pending"}
        self.last_scope_decision = None
        self.last_scope_decision = assess_policy_scope(public.observation, public.legal_actions)
        public = PolicyInput(public.observation, self.last_scope_decision.eligible_actions)
        from .native_search import rule_decision
        try:
            action, gaps = rule_decision(public)
        except (EnvironmentError, InvalidActionError) as error:
            self.last_backend = {"engine": "cpp", "status": "refused", "reason": str(error)}
            raise
        self.last_backend = {"engine": "cpp", "status": "complete", "scope_gaps": list(gaps)}
        return action


def combat_value(public: PolicyInput, outcome: CombatOutcome) -> float:
    """Native leaf evaluator; retained as a public-only diagnostic interface."""
    from .native_search import leaf_value
    return leaf_value(public, outcome)


class CombatSearchPolicy:
    """Paired flat Monte Carlo rollouts, one live branch, no recursive tree.

    Deadlines are cooperative: a blocked native call requires an external process
    watchdog. No branch/audit/exact-RNG objects escape into a rollout/value policy.
    """

    @property
    def backend(self) -> str:
        return "cpp"

    @backend.setter
    def backend(self, value: str) -> None:
        if value not in {"cpp", "auto"}:
            raise ValueError("production backend is C++; Python search is offline-only")

    def __init__(
        self, sampler: CombatSampler, *, budget: SearchBudget | None = None,
        sampling_seed: int = 0, coverage: RuleCoverage | None = None,
        rollout: StarterRulePolicy | None = None,
        value: Callable[[PolicyInput, CombatOutcome], float] = combat_value,
        clock: Callable[[], float] = time.perf_counter,
        backend: str = "cpp",
        root_model: str = STRICT_ROOT_MODEL,
    ) -> None:
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int) or not (
            0 <= sampling_seed < 2**64
        ):
            raise ValueError("sampling_seed must be an unsigned 64-bit integer")
        if backend not in {"auto", "cpp"}:
            raise ValueError("production backend must be cpp (auto is a native-only alias); Python search is offline-only")
        self.backend = "cpp"
        if value is not combat_value or clock is not time.perf_counter:
            raise ValueError("custom Python value/clock is offline-only")
        self.last_backend: dict = {}
        self.sampler = sampler
        if root_model not in {STRICT_ROOT_MODEL, EVENT_ROOT_MODEL}:
            raise ValueError("unknown root probability model")
        if root_model == EVENT_ROOT_MODEL and not getattr(sampler, "conditional_roots", False):
            raise ValueError("event probability sampling requires the public session root service")
        self.root_model = root_model
        self.algorithm_version = SEARCH_VERSION
        self.tree_config = None  # Only the separately versioned experimental subclass sets this.
        self.budget = budget or SearchBudget()
        self.sampling_seed = sampling_seed
        self.coverage = coverage
        self.rollout = rollout or StarterRulePolicy()
        from .native_search import compatible_rule
        if not compatible_rule(self.rollout):
            raise ValueError("custom Python rollout is offline-only")
        for name in ("samples_per_action", "rollout_horizon", "max_branch_steps", "max_samples", "max_value_evaluations"):
            if getattr(self.budget, name) > 2**31 - 1:
                raise ValueError("native budget integer range exceeded: " + name)
        self.clock = clock
        self.last_report: SearchReport | None = None
        self.native_search_calls = 0

    def choose(self, public: PolicyInput) -> Action:
        self.last_report = None
        self.last_report = self.rank(public)
        return self.last_report.action

    def rank(self, public: PolicyInput) -> SearchReport:
        self.native_search_calls = 0  # Dispatch attempts, not samples or rule-only decisions.
        _validate_input(public)
        from .native_search import compatible_rule
        if not compatible_rule(self.rollout):
            raise ValueError("custom Python rollout is offline-only")
        started = self.clock()
        self.last_backend = {"engine": "cpp", "reason": "rule_only", "discarded_native_steps": 0}
        budget = self.budget
        deadline = None if budget.max_seconds is None else started + budget.max_seconds
        try:
            fallback = self.rollout.choose(public)
        except (EnvironmentError, InvalidActionError) as error:
            self.last_backend = {"engine": "cpp", "reason": str(error), "status": "refused"}
            if self.coverage is not None:
                self.coverage.record_issue("native_rule_refused", str(error), public, context=SEARCH_VERSION)
            raise
        fingerprint = public_fingerprint(public)
        candidates = public.legal_actions
        excluded = len(public.legal_actions) - len(candidates)
        totals = {action: _Accumulator() for action in candidates}
        samples = steps = evaluations = policy_evaluations = discarded = 0
        rounds = peak = 0
        horizons: list[int] = []

        def finish(reason: str, fallback_reason: str | None = None) -> SearchReport:
            resource = self.last_backend.get("resource_fallback")
            if (self.root_model == EVENT_ROOT_MODEL and reason == "deadline"
                    and fallback_reason == "no_complete_round" and self.native_search_calls == 0):
                resource = {"version": "event-resource-fallback-v1", "reason": "deadline"}
                self.last_backend.update(root_model=self.root_model, resource_fallback=resource)
            if (fallback_reason and getattr(self.sampler, "conditional_roots", False)
                    and isinstance(public.observation.get("combat"), Mapping)):
                # The session path is an explicit acceptance contract. Missing
                # history, partial rounds or unsupported effects stop the caller;
                # they are never silently counted as searched rule-policy games.
                self.last_backend["refused_work"] = {
                    "stop_reason": reason, "fallback_reason": fallback_reason,
                    "samples": samples, "branch_steps": steps, "value_evaluations": evaluations,
                    "rollout_policy_evaluations": policy_evaluations,
                    "discarded_rollouts": samples, "discarded_rounds": rounds,
                }
                if not (self.root_model == EVENT_ROOT_MODEL and resource):
                    raise EnvironmentError(f"conditional session refused ({reason}): {fallback_reason}")
                self.last_backend["status"] = "resource_rule_fallback"
            estimates = candidate_estimates(totals)
            selected = fallback
            top_tie_count = 0
            tie_breaker = None
            tie_changed_action = False
            if estimates and fallback_reason is None:
                from .native_search import ranked_choice
                choice = ranked_choice(public, estimates)
                selected = choice["action"]
                top_tie_count = choice["top_tie_count"]
                tie_breaker = choice["tie_breaker"]
                tie_changed_action = choice["tie_changed_action"]
            if fallback_reason and self.coverage is not None:
                self.coverage.record_issue(
                    "combat_search_fallback", fallback_reason, public,
                    context=f"{SEARCH_VERSION}; fallback={RULE_VERSION}; stop={reason}",
                )
            return SearchReport(
                selected, fingerprint, reason, fallback_reason, rounds, samples, steps,
                evaluations, policy_evaluations, discarded, max(0.0, self.clock() - started),
                peak, excluded, tuple(horizons), estimates, budget,
                search_version=self.algorithm_version,
                resampler_version=self.root_model if self.root_model == EVENT_ROOT_MODEL else RESAMPLER_VERSION,
                tree_nodes=self.last_backend.get("tree", {}).get("nodes", 0),
                top_tie_count=top_tie_count, tie_breaker=tie_breaker,
                tie_changed_action=tie_changed_action,
            )

        if self.coverage is not None:
            self.coverage.observe(public, context=f"{SEARCH_VERSION}/live-root")
        rule_gaps = self.rollout.last_backend.get("scope_gaps", ())
        if rule_gaps:
            reason = "rule_scope:" + "; ".join(rule_gaps)
            self.last_backend.update(reason=reason, scope_fallback=True)
            return finish("unsupported", reason)
        if not isinstance(public.observation.get("combat"), Mapping):
            return finish("not_combat", "not_combat")
        combat = _mapping(public.observation.get("combat"))
        input_state = str(combat.get("input_state", "")).upper()
        if input_state and input_state != "PLAYER_NORMAL" and not (
            input_state == "CARD_SELECT" and getattr(self.sampler, "conditional_roots", False)
        ):
            return finish("non_normal_combat")
        if len(candidates) == 1:
            return finish("single_candidate")
        unsupported = self.unsupported_root_reason(public)
        if unsupported:
            return finish("unsupported", unsupported)

        native_method = getattr(self.sampler, "native_tree_search" if self.tree_config else "native_search", None)
        if not callable(native_method):
            self.last_backend = {"engine": "cpp", "reason": "native_sampler_unavailable"}
            raise EnvironmentError("C++ search requires a native sampler; Python fallback is disabled")
        from .combat_sampling import sampling_seeds
        seed_public, seed_fingerprint = public, fingerprint
        if getattr(self.sampler, "conditional_roots", False):
            from .native_search import session_seed_input
            seed_public = session_seed_input(public)
            seed_fingerprint = public_fingerprint(seed_public)
        # Each complete round needs at least one step/sample/value per
        # candidate. Never eagerly allocate the requested (possibly huge)
        # round count when the actual work budgets permit only a few.
        seed_rounds = min(budget.samples_per_action, budget.max_samples // len(candidates),
                          budget.max_value_evaluations // len(candidates),
                          budget.max_branch_steps // len(candidates))
        seed_plan = []
        for r in range(seed_rounds):
            if deadline is not None and self.clock() >= deadline:
                return finish("deadline", "no_complete_round")
            seed_plan.append(sampling_seeds(seed_public, int.from_bytes(hashlib.sha256(
                f"{SEARCH_SAMPLING_NAMESPACE}:{self.sampling_seed}:{seed_fingerprint}:{r}".encode()
            ).digest()[:8], "big")))
        if self.tree_config is not None:
            # Tree-learning worlds are independent of the paired final evaluation
            # prefix. This only creates seeds, not a Python simulation/search loop.
            for r in range(self.tree_config.training_rollouts):
                if deadline is not None and self.clock() >= deadline:
                    return finish("deadline", "no_complete_round")
                seed_plan.append(sampling_seeds(seed_public, int.from_bytes(hashlib.sha256(
                    f"{getattr(self, 'planning_seed_namespace', self.algorithm_version)}:planning:{self.sampling_seed}:{seed_fingerprint}:{r}".encode()
                ).digest()[:8], "big")))
        for r in range(getattr(self, "verification_proposals", 0)):
            if deadline is not None and self.clock() >= deadline:
                return finish("deadline", "no_complete_round")
            seed_plan.append(sampling_seeds(seed_public, int.from_bytes(hashlib.sha256(
                    f"{getattr(self, 'verification_seed_namespace', self.algorithm_version)}:verification:{self.sampling_seed}:{seed_fingerprint}:{r}".encode()
            ).digest()[:8], "big")))
        limits = asdict(budget)
        if getattr(self, "combat_objective", None) is not None:
            limits["combat_objective"] = self.combat_objective
        if self.root_model == EVENT_ROOT_MODEL:
            limits["root_model"] = self.root_model
        if self.tree_config is not None:
            limits["tree"] = asdict(self.tree_config)
            from .history_tree_search import TREE_SEARCH_VERSION
            if self.algorithm_version != TREE_SEARCH_VERSION:
                limits["tree_algorithm"] = self.algorithm_version
        if deadline is not None:
            if self.clock() >= deadline:
                return finish("deadline", "no_complete_round")
            limits["max_seconds"] = max(0.0, deadline - self.clock())
        try:
            self.native_search_calls += 1
            native = native_method(public, seed_plan, limits)
        except UnsupportedCombatSamplingError as error:
            # Match the reference sampler refusal and its attempted-creation count.
            samples = 1
            return finish("unsupported", str(error))
        resource = None
        if self.root_model == EVENT_ROOT_MODEL and native is not None:
            resource = validate_event_result(native)
        elif native and native.get("root_model") == EVENT_ROOT_MODEL:
            raise EnvironmentError("unexpected event probability model in strict search")
        if native is not None and not native["unavailable"]:
            if native.get("combat_objective") != getattr(self, "combat_objective", None):
                raise EnvironmentError("C++ combat-objective contract mismatch")
            if self.tree_config is not None and (native.get("tree", {}).get("version") != self.algorithm_version
                    or (not native.get("refusal") and native.get("tree", {}).get("frozen") is not True)):
                raise EnvironmentError("C++ history-tree result contract mismatch; no flat fallback")
            samples, steps, evaluations = native["samples"], native["steps"], native["values"]
            policy_evaluations, discarded, peak = native["policies"], native["discarded"], native["peak"]
            outcomes = (CombatOutcome.ONGOING, CombatOutcome.VICTORY, CombatOutcome.LOSS, CombatOutcome.ESCAPED)
            for rows in native["rounds"]:
                for action, row in zip(candidates, rows, strict=True):
                    totals[action].add(row["value"], row["hp"], outcomes[row["outcome"]], row.get("resource_hp"))
            rounds = len(native["rounds"])
            horizons = native["horizons"]
            self.last_backend = {"engine": "cpp", "reason": None, "discarded_native_steps": 0}
            if "session" in native:
                self.last_backend.update(session=native["session"], root_sampling=native["root_sampling"])
            if "tree" in native:
                self.last_backend["tree"] = native["tree"]
            if "combat_objective" in native:
                self.last_backend["combat_objective"] = native["combat_objective"]
            if self.root_model == EVENT_ROOT_MODEL:
                self.last_backend.update(root_model=self.root_model, resource_fallback=resource)
            if resource:
                return finish("resource_limit", "event_resource:" + resource["reason"])
            return finish(native["stop"], native.get("refusal") or ("no_complete_round" if rounds == 0 else None))
        self.last_backend = {"engine": "cpp", "reason": native["unavailable"] if native else "native_extension_unavailable",
                             "discarded_native_steps": native["steps"] if native else 0,
                             "discarded_native_samples": native["samples"] if native else 0}
        if native and "public_failure" in native:
            self.last_backend["public_failure"] = native["public_failure"]
        if native and native.get("scope_gap") is True:
            # All-or-none for this root, including earlier completed rounds.
            # Choosing from only surviving samples would bias the estimator.
            samples, steps, evaluations = native["samples"], native["steps"], native["values"]
            policy_evaluations, peak, discarded = native["policies"], native["peak"], native["samples"]
            self.last_backend["scope_fallback"] = True
            return finish("unsupported_continuation", "unreviewed_continuation:" + native["unavailable"])
        # A narrowed native support envelope is a refusal, not permission to
        # restart a different implementation against the remaining deadline.
        if self.coverage is not None:
            self.coverage.record_issue("native_search_refused", self.last_backend["reason"], public,
                                       context=SEARCH_VERSION)
        raise EnvironmentError("C++ backend unavailable: " + self.last_backend["reason"])

    def unsupported_root_reason(self, public: PolicyInput) -> str | None:
        """Return the fail-closed public action/card reason, if this root has one."""

        if hidden_enemy_intents(public.observation):
            return "Runic Dome hides the current enemy intent from combat sampling"
        combat = _mapping(public.observation.get("combat"))
        from .public_piles import draw_order_sampling_reason
        conditioned = bool(getattr(self.sampler, "conditional_roots", False))
        order_reason = None if conditioned else draw_order_sampling_reason(combat)
        if order_reason is not None:
            return order_reason
        hand = combat.get("hand", ())
        from .hand_resident_curses import hand_resident_curse_reason
        resident_reason = hand_resident_curse_reason(combat, require_order_certainty=not conditioned)
        if resident_reason is not None:
            return resident_reason
        generation_reason = generation_support_reason(public.observation)
        if generation_reason is not None:
            return generation_reason
        from .forethought import free_play_state_reason, pending_bottom_choices
        dynamic_reason = dynamic_state_reason(combat) or play_trigger_state_reason(combat) or free_play_state_reason(combat, require_order_certainty=not conditioned)
        if dynamic_reason is not None:
            return dynamic_reason
        autoplay_reason = autoplay_support_reason(combat)
        if autoplay_reason is not None:
            return autoplay_reason
        if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and str(combat.get("card_select_task", "")).upper() == "FORETHOUGHT":
            try:
                pending_bottom_choices(combat)
            except ValueError as error:
                return str(error)
        if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and str(combat.get("card_select_task", "")).upper() == "EXHAUST_MANY":
            from .multi_exhaust import pending_exhaust_options
            try:
                pending_exhaust_options(combat)
            except ValueError as error:
                return str(error)
        if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and str(combat.get("card_select_task", "")).upper() == "DUAL_WIELD":
            try:
                pending_copy_count(combat)
            except ValueError as error:
                return str(error)
        for action in public.legal_actions:
            if action.kind is ActionKind.PLAY_CARD and action.card_index is not None:
                if not 0 <= action.card_index < len(hand):
                    raise InvalidActionError("legal play references an invalid hand index")
                card = _mapping(hand[action.card_index])
                status = self.rollout.catalog.status("card", card.get("id"))
                if status["sampling"] != CARD_SAMPLING_STATUS:
                    evidence = status["evidence"]["capabilities"][
                        EvidenceCapability.SEARCH.value
                    ]
                    layer = evidence["missing_layers"][0]
                    return (
                        f"search card evidence missing {layer}: {card.get('id')!r}"
                    )
            elif action.kind is ActionKind.CHOOSE:
                is_audited_selection = (
                    str(combat.get("input_state", "")).upper() == "CARD_SELECT"
                    and str(combat.get("card_select_task", "")).upper()
                    in SELECTION_TASKS
                )
                if not is_audited_selection:
                    return "search choose effects not audited for this selection"
            elif action.kind not in {
                ActionKind.PLAY_CARD, ActionKind.END_TURN,
                ActionKind.USE_POTION, ActionKind.DISCARD_POTION,
            }:
                return f"search action effects not audited: {action.kind.value}"
        return None
