"""Versioned frozen combat choices for training/evaluation, never actor labels."""
from collections import Counter
from dataclasses import asdict
import math

from .combat_sampling import BoundCombatSampler, CARD_HISTORY_VERSION, RESAMPLER_VERSION
from .combat_search import RULE_VERSION, SEARCH_VERSION, VALUE_VERSION, RANKING_VERSION, SEARCH_SAMPLING_NAMESPACE, SearchBudget
from .fast_policy import FAST_POLICY_VERSION, FastBaselinePolicy
from .native_search import NATIVE_SEARCH_ABI, SESSION_SEARCH_ABI
from .potion_contracts import POTION_CONTRACT_VERSION

FIXED_COMBAT_VERSION = "frozen-native-combat-v2-potion-search"
CONTROLLERS = ("native_rule", "native_search")
# Keep historical two-arm protocols unchanged. New experiments opt in by name;
# neither a new binary nor session_search=True silently changes an old contract.
SESSION_CONTROLLER = "native_session_search"
BOUNDARY_CONTROLLER = "native_session_boundaries"
SESSION_CONTROLLERS = (SESSION_CONTROLLER, BOUNDARY_CONTROLLER)
EXPERIMENT_CONTROLLERS = (*CONTROLLERS, *SESSION_CONTROLLERS)
SESSION_COMBAT_VERSION = "frozen-native-session-combat-v1"


def fixed_combat_contract(controller, sampling_seed=19000):
    if controller not in EXPERIMENT_CONTROLLERS:
        raise ValueError("unknown fixed combat controller")
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**64:
        raise ValueError("fixed combat sampling seed must be an unsigned 64-bit integer")
    search = controller != "native_rule"
    contract = {"version": FIXED_COMBAT_VERSION, "controller": controller,
        "fast_policy": FAST_POLICY_VERSION, "rule": RULE_VERSION,
        "native_abi": NATIVE_SEARCH_ABI, "potion_contracts": POTION_CONTRACT_VERSION,
        "history": CARD_HISTORY_VERSION, "search": SEARCH_VERSION,
        "value": VALUE_VERSION, "ranking": RANKING_VERSION, "resampler": RESAMPLER_VERSION,
        "search_budget": asdict(SearchBudget()) if search else None,
        "sampling_seed": sampling_seed if search else None,
        "sampling_namespace": SEARCH_SAMPLING_NAMESPACE if search else None,
        "actor_training_labels": False, "potion_search": search}
    if controller == SESSION_CONTROLLER:
        contract.update(version=SESSION_COMBAT_VERSION, root_service=SESSION_SEARCH_ABI,
            proposal_namespace="public-session-id-invariant-v1", fallback_policy="refuse",
            construction_limits={"work": 8_000_000, "random_words": 1_000_000})
    elif controller == BOUNDARY_CONTROLLER:
        from .session_boundaries import contract as boundary_contract, VERSION, MODEL
        contract.update(version="frozen-native-session-boundaries-v1", root_service=SESSION_SEARCH_ABI,
            search=VERSION, resampler=MODEL, root_model=MODEL, fallback_policy="refuse",
            proposal_namespace="public-session-id-invariant-v1",
            boundary_schedule=boundary_contract(sampling_seed))
    return contract


def make_fixed_environment(module_dir, *, controller="native_rule", session_search=None):
    """Explicit simulator-only adapter selection, independent of live deployment.

    A rule reference may use the same session adapter as its search candidate.
    Search contracts cannot be cross-wired; the binary is never auto-replaced.
    """
    fixed_combat_contract(controller)
    expected = controller in SESSION_CONTROLLERS
    if session_search is None:
        session_search = expected
    if type(session_search) is not bool or (controller != "native_rule" and session_search != expected):
        raise ValueError("fixed combat controller/root-service mismatch")
    from .simulator_environment import StsLightSpeedEnvironment
    from .native_search import compatible_versions, compatible_session
    environment = StsLightSpeedEnvironment(module_dir, session_search=session_search)
    if (not compatible_versions(environment.module)
            or (session_search and not compatible_session(environment.module))
            or hasattr(environment.module.SimulatorSession, "_test_fixture")):
        raise ValueError("fixed combat requires the declared native contract with test fixtures OFF")
    if controller == BOUNDARY_CONTROLLER:
        from .session_boundaries import BoundarySampler
        BoundarySampler(environment)  # Capability check before the first reset.
    return environment


def make_fixed_combat(environment=None, *, controller="native_rule", sampling_seed=19000, catalog=None):
    fixed_combat_contract(controller, sampling_seed)
    if controller == "native_rule":
        return FastBaselinePolicy(catalog=catalog)
    if environment is None:
        raise ValueError("native search requires its current source environment")
    if bool(getattr(environment, "session_search", False)) != (controller in SESSION_CONTROLLERS):
        raise ValueError("fixed combat controller/root-service mismatch")
    if controller == BOUNDARY_CONTROLLER:
        from .session_boundaries import BoundaryPolicy
        return BoundaryPolicy(environment, sampling_seed=sampling_seed, catalog=catalog)
    from .frozen_evaluation import ResourceAlignedSearchPolicy
    return ResourceAlignedSearchPolicy(BoundCombatSampler(environment), budget=SearchBudget(),
                                       sampling_seed=sampling_seed, catalog=catalog)


class FixedCombatAudit:
    """Accounting only; sanitized state/diagnostics, no environment or RNG access."""
    def __init__(self):
        self.counts, self.modes, self.gaps, self.stops, self.refusals = (Counter() for _ in range(5))
        self.seconds = []
        self.native_seconds = []
        self.examples = []
        self.last_error = None
        self.coverage = Counter()
        self.coverage_by_input = {}
        self.root_services = Counter()

    def record(self, public, policy, elapsed, error=None):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid fixed-controller latency")
        self.counts["decision_attempts"] += 1
        self.counts["errors"] += int(error is not None)
        self.seconds.append(elapsed)
        decision = getattr(policy, "last_decision", None)
        if isinstance(decision, dict):
            self.modes[decision["mode"]] += 1
            self.gaps.update(set(decision.get("gaps", ())))
            self.counts["reported_decisions"] += 1
            self.counts["fallbacks"] += int(bool(decision.get("fallback")))
        elif error is None:
            self.counts["unreported_decisions"] += 1
        diagnostic = getattr(policy, "last_search_diagnostic", None)
        bucket = combat_decision_kind(public, decision, diagnostic, error)
        self.coverage[bucket] += 1
        combat = public.observation.get("combat", {})
        input_kind = str(combat.get("input_state", "UNKNOWN"))
        if input_kind == "CARD_SELECT":
            input_kind += ":" + str(combat.get("card_select_task", "UNKNOWN"))
        self.coverage_by_input.setdefault(input_kind, Counter())[bucket] += 1
        if bucket == "searched":
            self.root_services[diagnostic["backend"].get("session", {}).get("version", NATIVE_SEARCH_ABI)] += 1
        reason = str(error) if error else None
        if diagnostic:
            self.counts["native_dispatches"] += diagnostic["native_calls"]
            if diagnostic["native_calls"]:
                self.native_seconds.append(elapsed)
            report = diagnostic["report"]
            if report:
                self.stops[report["stop_reason"]] += 1
                self.counts["completed_searches"] += int(report["completed_rounds"] > 0)
                self.counts["search_fallbacks"] += int(report["fallback_reason"] is not None)
                for name in ("samples", "branch_steps", "value_evaluations", "rollout_policy_evaluations",
                             "discarded_rollouts", "completed_rounds"):
                    self.counts[name] += report[name]
                self.counts["horizon_leaves"] += diagnostic["horizon_leaves"]
                reason = reason or report["fallback_reason"]
            else:
                for name in ("discarded_native_steps", "discarded_native_samples"):
                    self.counts[name] += diagnostic["backend"].get(name, 0)
                reason = reason or diagnostic.get("error") or diagnostic["backend"].get("reason")
                failure = diagnostic["backend"].get("refused_work")
                if failure:
                    self.stops[failure["stop_reason"]] += 1
                    # Rejected all-or-none requests have no usable estimates.
                    # Do not count their completed-but-discarded rounds as a
                    # successful decision, or lose their already-spent work.
                    for name in ("samples", "branch_steps", "value_evaluations", "rollout_policy_evaluations"):
                        self.counts["refused_" + name] += failure[name]
        if error is not None:
            # Fatal evidence must survive an already-full fallback example panel.
            self.last_error = {"floor": public.observation.get("run", {}).get("floor"),
                "fingerprint": public.fingerprint(), "reason": str(error), "search": diagnostic}
        if reason:
            self.refusals[reason] += 1
            if len(self.examples) < 8:
                self.examples.append({"floor": public.observation.get("run", {}).get("floor"),
                    "fingerprint": public.fingerprint(), "reason": reason, "search": diagnostic})

    def summary(self):
        from .training_readiness import latency_summary
        return {"counts": dict(self.counts), "modes": dict(self.modes), "gaps": dict(self.gaps),
            "search_stops": dict(self.stops), "refusal_reasons": dict(self.refusals),
            "decision_latency": latency_summary(self.seconds),
            "native_dispatch_latency": latency_summary(self.native_seconds), "examples": list(self.examples),
            "last_error": self.last_error,
            "coverage": {"version": "combat-decision-coverage-v1", "kinds": dict(self.coverage),
                "by_input": {k: dict(v) for k, v in self.coverage_by_input.items()},
                "root_services": dict(self.root_services),
                "note": "exclusive attempted-decision buckets; errors/refusals and forced actions are never searched; rule_only is intentional only in the rule reference"},
            "note": "dispatch attempts may refuse in preflight; samples include attempted creation; failed native work is separate; timings include rule/potion overhead, not pure C++ compute"}


def combat_decision_kind(public, decision, diagnostic, error=None):
    """Exclusive coverage denominators, not merely native dispatch counts."""
    if error is not None or (diagnostic and diagnostic.get("error")):
        return "refused"
    report = (diagnostic or {}).get("report") or {}
    if (decision or {}).get("fallback") or report.get("fallback_reason"):
        return "rule_fallback"
    if report.get("completed_rounds", 0) > 0 and (diagnostic or {}).get("native_calls", 0) > 0:
        return "searched"
    if len(public.legal_actions) == 1:
        return "forced"
    if diagnostic:
        return "unsearched"
    if decision is not None:
        return "rule_only"
    return "unreported"
