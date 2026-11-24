"""Inference helpers from tree_standalone_candidate.py."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from spire_ai.actions import Action
from spire_ai.combat_sampling import sampling_seeds, public_fingerprint, BoundCombatSampler, CombatOutcome
from spire_ai.combat_search import CombatSearchPolicy, candidate_estimates, _Accumulator, SearchReport
from spire_ai.environment import EnvironmentError
from spire_ai.event_sampling import EVENT_ROOT_MODEL
from spire_ai.frozen_evaluation import _SearchRule
from spire_ai.history_tree_search import COMBAT_OBJECTIVE_VERSION, TreeConfig
from spire_ai.native_search import session_projection_json, metadata_json, session_seed_input, compatible_session, compatible_versions
from spire_ai.rule_coverage import _validate_input
from spire_ai.fast_policy import FastBaselinePolicy
VERSION = "public-history-tree-v9-standalone"


def canonical(action):
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def objective_key(e):
    return (-e.loss_rate, e.victory_rate, e.worst_20pct_resource_hp,
            e.p10_resource_hp, e.mean_resource_hp, e.mean_value)


def validate_result(result, public, budget, tree_config, *, version=VERSION):
    """Recompute recommendations from actual outcomes; never invent flat rounds."""
    if (result.get("contract") != version or result.get("root_model") != EVENT_ROOT_MODEL
            or result.get("combat_objective") != COMBAT_OBJECTIVE_VERSION
            or "rounds" in result or "horizons" in result or result.get("resource_fallback")
            or result.get("unavailable")):
        raise EnvironmentError("standalone result contract mismatch")
    if result.get("refusal") or result.get("action") is None:
        raise EnvironmentError("standalone refused: " + str(result.get("refusal") or result.get("stop")))
    t = result["tree"]
    if (t["version"] != version or t["frozen"] is not True or t["flat_delegations"] != 0
            or t["verification_simulations"] != 0 or t["calibrated_confidence"] is not False
            or t["new_node_cap"] != 3 or t["max_new_nodes_per_simulation"] > 3):
        raise EnvironmentError("standalone tree contract mismatch")
    n = len(public.legal_actions)
    cap = min(tree_config.training_rollouts, budget.max_samples, budget.max_value_evaluations, budget.max_branch_steps)
    if (t["simulation_cap"] != cap or t["simulation_horizon"] != min(budget.rollout_horizon, budget.max_branch_steps//n)
            or t["covered_root_actions"] != n or t["root_action_count"] != n):
        raise EnvironmentError("standalone coverage/horizon contract mismatch")
    for field, limit in (("steps", budget.max_branch_steps), ("samples", cap), ("values", budget.max_value_evaluations)):
        if type(result[field]) is not int or not 0 < result[field] <= limit:
            raise EnvironmentError("standalone work cap: " + field)
    complete = t["completed_simulations"]
    if (not n <= complete <= result["values"] <= result["samples"]
            or result["discarded"] != result["samples"]-complete
            or t["interrupted_simulations"] != result["samples"]-complete
            or t["tree_policy_steps"]+t["rule_leaf_steps"] != result["steps"]
            or t["rule_calls"] != result["policies"] or not 0 <= result["policies"] <= result["steps"]
            or not complete <= t["backed_up_edges"] <= t["tree_policy_steps"]
            or t["unused_steps"] != budget.max_branch_steps-result["steps"]
            or t["nodes"] > tree_config.max_nodes or t["retained_payload_bytes"] > tree_config.arena_bytes):
        raise EnvironmentError("standalone incomplete work accounting")
    sampling = result["root_sampling"]
    if (sampling.get("probability_model") != EVENT_ROOT_MODEL
            or sampling["construction_work"] > sampling["construction_work_limit"]
            or sampling["random_words"] > sampling["random_word_limit"]
            or (sampling["valid"] is not True and result["stop"] != "deadline")):
        raise EnvironmentError("standalone conditional root accounting")
    totals = {};native_keys = {}
    outcomes = (CombatOutcome.ONGOING, CombatOutcome.VICTORY, CombatOutcome.LOSS, CombatOutcome.ESCAPED)
    for entry in result["root_evidence"]:
        action = Action.from_dict(entry["action"])
        if action in totals or action not in public.legal_actions or entry["visits"] != len(entry["outcomes"]) or not entry["visits"]:
            raise EnvironmentError("standalone root evidence mismatch")
        acc = _Accumulator()
        for row in entry["outcomes"]:
            if (type(row["outcome"]) is not int or not 0 <= row["outcome"] <= 3
                    or any(not isinstance(row[k], (int, float)) or isinstance(row[k], bool)
                           or not math.isfinite(row[k]) for k in ("hp", "resource_hp", "value"))
                    or row["hp"] < 0 or int(row["hp"]) != row["hp"]):
                raise EnvironmentError("standalone invalid outcome")
            acc.add(row["value"], int(row["hp"]), outcomes[row["outcome"]], row["resource_hp"])
        totals[action] = acc;native_keys[action] = tuple(entry["objective_key"])
    estimates = candidate_estimates(totals)
    if set(totals) != set(public.legal_actions) or sum(e.samples for e in estimates) != complete:
        raise EnvironmentError("standalone missing or unaccounted completed outcomes")
    if any(objective_key(e) != native_keys[e.action] for e in estimates):
        raise EnvironmentError("standalone objective aggregation mismatch")
    ordered = sorted(estimates, key=lambda e: canonical(e.action.as_dict()))
    best = max(ordered, key=objective_key)
    if result["action"] != best.action.as_dict():
        raise EnvironmentError("standalone recommendation does not follow its evidence")
    return best.action, estimates


class StandaloneSampler(BoundCombatSampler):
    def native_tree_search(self, public, seeds, budget, *, trace=False):
        env = self._environment
        if (not env.session_search or not compatible_versions(env.module) or not compatible_session(env.module)
                or budget.get("tree_algorithm") != VERSION
                or getattr(env.module, "public_standalone_tree_search_abi", None) != VERSION
                or budget.get("combat_objective") != COMBAT_OBJECTIVE_VERSION
                or getattr(env.module, "public_combat_objective_abi", None) != COMBAT_OBJECTIVE_VERSION
                or budget.get("root_model") != EVENT_ROOT_MODEL
                or getattr(env.module, "public_event_offer_abi", None) != EVENT_ROOT_MODEL):
            raise ValueError("explicit standalone/event/objective contract required")
        started = time.perf_counter()
        current = env._prepare_combat_sampling(public, conditioned=True)
        limits = dict(budget)
        if limits["max_seconds"] is not None:
            limits["max_seconds"] = max(0., limits["max_seconds"]-(time.perf_counter()-started))
        self.request = (public, seeds, dict(limits))
        self.result = json.loads(env.session.search_session_standalone_tree(seeds, session_projection_json(current),
            json.dumps([a.as_dict() for a in current.legal_actions]), json.dumps(limits), metadata_json(), trace))
        return self.result


@dataclass(frozen=True, slots=True)
class StandaloneReport(SearchReport):
    completed_simulations: int = 0


class StandaloneSearch(CombatSearchPolicy):
    version = VERSION
    def __init__(self, sampler, *, tree_config=None, **kwargs):
        super().__init__(sampler, **kwargs)
        if self.root_model != EVENT_ROOT_MODEL or not getattr(sampler, "conditional_roots", False):
            raise ValueError("standalone requires event-conditioned session roots")
        self.tree_config = tree_config or TreeConfig(training_rollouts=256)
        self.algorithm_version = self.version

    def configure_request(self, public, limits):
        return limits

    def validate_native(self, result, public):
        return validate_result(result, public, self.budget, self.tree_config)

    def rank(self, public):
        started = time.perf_counter();self.native_search_calls = 0
        self.last_backend = {"engine": "cpp", "reason": None}
        _validate_input(public)
        if not public.observation.get("combat"):
            raise EnvironmentError("standalone is a combat controller")
        budget = self.budget;native = None;estimates = ()
        fingerprint = public_fingerprint(public)
        try:
            if len(public.legal_actions) == 1:
                action = public.legal_actions[0]
            else:
                unsupported = self.unsupported_root_reason(public)
                if unsupported:raise EnvironmentError("standalone scope refusal: " + unsupported)
                seed_public = session_seed_input(public);seed_fingerprint = public_fingerprint(seed_public)
                count = min(self.tree_config.training_rollouts, budget.max_samples,
                            budget.max_value_evaluations, budget.max_branch_steps)
                seeds = [sampling_seeds(seed_public, int.from_bytes(hashlib.sha256(
                    f"{VERSION}:planning:{self.sampling_seed}:{seed_fingerprint}:{i}".encode()
                ).digest()[:8], "big")) for i in range(count)]
                limits = dict(asdict(budget), tree=asdict(self.tree_config), tree_algorithm=self.algorithm_version,
                              combat_objective=COMBAT_OBJECTIVE_VERSION, root_model=self.root_model)
                limits = self.configure_request(seed_public, limits)
                if budget.max_seconds is not None:
                    limits["max_seconds"] = max(0., budget.max_seconds-(time.perf_counter()-started))
                self.native_search_calls += 1
                native = self.sampler.native_tree_search(public, seeds, limits)
                self.last_backend.update({k: native[k] for k in
                    ("tree", "session", "root_sampling", "combat_objective", "root_model", "resource_fallback")})
                action, estimates = self.validate_native(native, public)
        except Exception as error:
            self.last_backend.update(status="refused", reason=str(error))
            if native is not None:
                self.last_backend["refused_work"] = {k: native[k] for k in
                    ("steps", "samples", "values", "policies", "discarded", "stop", "refusal")}
            raise
        complete = native["tree"]["completed_simulations"] if native else 0
        return StandaloneReport(action, fingerprint, native["stop"] if native else "single_candidate", None,
            0, native["samples"] if native else 0, native["steps"] if native else 0,
            native["values"] if native else 0, native["policies"] if native else 0,
            native["discarded"] if native else 0, time.perf_counter()-started, native["peak"] if native else 0,
            0, (), estimates, budget, search_version=self.algorithm_version, resampler_version=EVENT_ROOT_MODEL,
            tree_nodes=native["tree"]["nodes"] if native else 0, tie_breaker="canonical-action-json" if native else None,
            completed_simulations=complete)


class StandalonePolicy:
    search_class = StandaloneSearch
    def __init__(self, sampler, *, budget, sampling_seed, root_model, tree_config=None, catalog=None):
        self.fast = FastBaselinePolicy(catalog=catalog);rule = self.fast.combat.rule
        self.search = self.search_class(sampler, budget=budget, sampling_seed=sampling_seed,
            root_model=root_model, tree_config=tree_config, rollout=rule)
        self.fast.combat.rule = _SearchRule(self.search, rule);self.fast.combat.potion_override = False
        self.last_decision = self.last_search_diagnostic = None

    def choose(self, public):
        self.last_decision = self.last_search_diagnostic = None
        self.search.last_report = None;self.search.last_backend = {};self.search.native_search_calls = 0
        started = time.perf_counter();error = None
        try:
            action = self.fast.choose(public)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}";raise
        finally:
            if self.search.last_backend:
                report = self.search.last_report
                self.last_search_diagnostic = dict(native_calls=self.search.native_search_calls,
                    backend=dict(self.search.last_backend), error=error,
                    report=report.as_dict() if report else None,
                    horizon_leaves=sum(e.truncated for e in report.candidates) if report else 0,
                    controller_seconds=time.perf_counter()-started)
        self.last_decision = dict(self.fast.last_decision)
        report = self.search.last_report
        if report:
            self.last_decision.update(mode="combat_search" if report.completed_simulations else "combat_forced",
                                      search_calls=self.search.native_search_calls)
            if report.completed_simulations:
                self.last_decision["gaps"] = [g for g in self.last_decision["gaps"] if g != "combat_selection_rule_only"]
                self.last_decision["root_service"] = self.search.last_backend["session"]["version"]
        return action
