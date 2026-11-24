"""Public-only capability audit and bounded full-run throughput measurement.

Exposure is not correctness evidence. A successful driver run does not certify
that every native card, potion, relic or event agrees with the original game.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import time
from typing import Callable

from .corpus_policy import CorpusDecision
from .environment import InvalidActionError
from .evaluation import _is_win
from .fast_policy import FAST_POLICY_VERSION
from .policy_api import policy_input_from_state
from .policy_scope import UnsupportedPolicyScopeError
from .public_map import public_map_status


READINESS_VERSION = "public-training-readiness-v6-combat-support"
CAPABILITIES = (
    ("combat", "native_rule_and_potion_aware_search", "root_selections_rule_only_and_conditional_order_gaps"),
    ("card_reward", "profile_heuristic_partial", "group_x_generation_upgrade_valuation_pending"),
    ("route", "public_current_act_map_and_action_links", "full_graph_value_and_live_map_parity_pending"),
    ("campfire", "typed_choices_and_hp_heuristic", "relic_interactions_partial"),
    ("shop", "typed_items_prices_remove_leave", "relic_and_resource_budget_valuation_pending"),
    ("event", "public_resource_effect_heuristic_partial", "random_multistage_value_and_effect_fidelity_pending"),
    ("card_select", "typed_upgrade_remove_transform_candidates", "task_specific_valuation_partial"),
    ("relic_reward", "typed_identity_stable_choice", "full_relic_effect_catalog_and_fidelity_audit_pending"),
    ("potion", "33_ironclad_contracts_search_history_and_lifecycle", "offclass_and_exhaustive_interaction_fidelity_not_certified"),
)


def capability_matrix() -> list[dict]:
    return [{"category": name, "implemented": implementation, "remaining_gap": gap,
             "certified_training_ready": False} for name, implementation, gap in CAPABILITIES]


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "total_seconds": 0.0, "mean_seconds": None,
                "p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered) - 1) * q
        low = int(index)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {"count": len(values), "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values), "p50_seconds": quantile(.5),
            "p95_seconds": quantile(.95), "max_seconds": max(values)}


class TimedNativeSession:
    """Benchmark-only instrumentation; these timings NEST within adapter timings."""
    def __init__(self, session, samples: dict, clock: Callable[[], float]):
        self.session, self.samples, self.clock = session, samples, clock

    def __getattr__(self, name):
        value = getattr(self.session, name)
        if name not in {"get_state", "get_observation", "get_actions_text", "take_action", "reset"}:
            return value
        def timed(*args, **kwargs):
            started = self.clock()
            try:
                return value(*args, **kwargs)
            finally:
                self.samples["native_" + name].append(self.clock() - started)
        return timed


def probe_episode(environment, policy, *, seed: int, max_steps: int,
                  clock: Callable[[], float] = time.perf_counter, encoder=None) -> dict:
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    samples: dict[str, list[float]] = defaultdict(list)
    if hasattr(environment, "session"):
        environment.session = TimedNativeSession(environment.session, samples, clock)
    screens, modes, gaps, actions = Counter(), Counter(), Counter(), Counter()
    maps, map_links, event_options = Counter(), Counter(), Counter()
    selected_event_statuses = Counter()
    search_counts, search_stops, search_refusals = Counter(), Counter(), Counter()
    search_examples = []
    encoded_steps = encoded_batches = 0
    encoding_unknowns, encoding_gaps, encoding_max_rows = Counter(), Counter(), Counter()
    pending_batch = []

    def flush_batch():
        nonlocal encoded_batches
        if pending_batch:
            from .public_encoding import collate_decisions
            at = clock()
            collate_decisions(pending_batch)
            samples["public_collate"].append(clock() - at)
            encoded_batches += 1
            pending_batch.clear()
    examples = []
    chosen_unknown = 0
    fallbacks = 0
    unreported_fallbacks = 0
    steps = attempts = 0
    error = None
    status = "truncated"
    state = None
    public = None
    stage = "reset"
    started = clock()
    try:
        reset_at = clock()
        state = environment.reset(seed=seed, character="IRONCLAD", ascension=0)
        samples["environment_reset"].append(clock() - reset_at)
        for _ in range(max_steps):
            stage = "public_input"
            if not state.legal_actions:
                raise InvalidActionError("nonterminal state has no legal actions")
            at = clock()
            public = policy_input_from_state(state)
            samples["public_conversion"].append(clock() - at)
            encoded = None
            if encoder is not None:
                at = clock()
                encoded = encoder.encode(public)
                samples["public_encode"].append(clock() - at)
                encoded_steps += 1
                encoding_unknowns.update(encoded.unknown_categories)
                encoding_gaps.update(encoded.gaps)
                for name, table in encoded.tables.items():
                    encoding_max_rows[name] = max(encoding_max_rows[name], len(table.values))
                pending_batch.append(encoded)
                if len(pending_batch) == 8:
                    flush_batch()
            screen = str(public.observation.get("screen", {}).get("type", "UNKNOWN"))
            screens[screen] += 1
            maps[public_map_status(public.observation)] += 1
            screen_data = public.observation.get("screen", {})
            context = screen_data.get("decision_context", {})
            if screen == "MAP":
                map_links[context.get("map_link_status", "missing_map_links")] += 1
            if screen == "EVENT":
                for row, label in zip(screen_data.get("decision_options", ()), screen_data.get("choices", ())):
                    event_options[json.dumps({"event": context.get("event_name", "UNKNOWN"),
                        "label": label, "kind": row.get("kind"),
                        "semantic_status": row.get("event_choice", {}).get("status", "missing")},
                        sort_keys=True)] += 1
            at = clock()
            stage = "policy_choose"
            try:
                action = policy.choose(public)
            finally:
                samples["policy_choose"].append(clock() - at)
                diagnostic = getattr(policy, "last_search_diagnostic", None)
                if diagnostic is not None:
                    search_counts["decisions"] += 1
                    search_counts["native_calls"] += diagnostic["native_calls"]
                    search_counts["controller_errors"] += int(bool(diagnostic.get("error")))
                    report = diagnostic["report"]
                    if report is not None:
                        search_stops[report["stop_reason"]] += 1
                        search_counts["completed_searches"] += int(report["completed_rounds"] > 0)
                        search_counts["fallbacks"] += int(report["fallback_reason"] is not None)
                        for name in ("samples", "branch_steps", "value_evaluations", "rollout_policy_evaluations",
                                     "discarded_rollouts", "completed_rounds"):
                            search_counts[name] += report[name]
                        search_counts["horizon_leaves"] += diagnostic["horizon_leaves"]
                        reason = report["fallback_reason"]
                    else:
                        search_counts["errors"] += 1
                        backend = diagnostic["backend"]
                        reason = diagnostic.get("error") or backend.get("reason", "unknown_native_failure")
                        for name in ("discarded_native_steps", "discarded_native_samples"):
                            search_counts[name] += backend.get(name, 0)
                    if diagnostic["native_calls"]:
                        # Includes failed attempts; separate from successful report elapsed.
                        samples["search_controller_with_native_call"].append(diagnostic["controller_seconds"])
                    if reason:
                        search_refusals[reason] += 1
                        if len(search_examples) < 8:
                            search_examples.append({"floor": public.observation.get("run", {}).get("floor"),
                                "fingerprint": public.fingerprint(), "diagnostic": diagnostic})
            if action not in public.legal_actions:
                raise InvalidActionError("probe policy returned illegal action")
            if encoded is not None and encoded.decode(public.legal_actions.index(action)) != action:
                raise InvalidActionError("encoded action round trip changed the chosen action")
            decision = getattr(policy, "last_decision", None)
            if isinstance(decision, CorpusDecision):
                # The existing corpus baseline reports a dataclass, not the new
                # policy's dictionary. Its first-legal choices are still fallbacks.
                fallback = decision.mode != "combat_rule"
                decision = {"mode": decision.mode, "fallback": fallback,
                            "gaps": ["corpus_first_legal_choice"] if fallback else []}
            if isinstance(decision, dict):
                modes[decision["mode"]] += 1
                fallbacks += int(decision.get("fallback", False))
                unreported_fallbacks += int("fallback" not in decision)
                gaps.update(set(decision.get("gaps", ())))
                if decision.get("fallback") and len(examples) < 8:
                    examples.append({"floor": public.observation.get("run", {}).get("floor"),
                                     "screen": screen, "mode": decision["mode"],
                                     "gaps": decision.get("gaps", ()), "action": action.as_dict()})
            else:
                modes[getattr(decision, "mode", "unreported_policy")] += 1
                unreported_fallbacks += 1
            options = public.observation.get("screen", {}).get("decision_options", ())
            chosen_unknown += sum(row.get("kind") == "unknown" and row.get("action") == action.as_dict()
                                  for row in options)
            if screen == "EVENT":
                selected = next((row for row in options if row.get("action") == action.as_dict()), {})
                selected_event_statuses[selected.get("event_choice", {}).get("status", "missing")] += 1
            attempts += 1
            at = clock()
            stage = "environment_step"
            result = environment.step(action)
            samples["environment_step"].append(clock() - at)
            state = result.state
            steps += 1
            actions[action.kind.value] += 1
            if result.done:
                if result.truncated:
                    status = "truncated"
                elif _is_win(state, result.terminated, "act3_boss"):
                    status = "win"
                elif state.observation.get("run", {}).get("outcome") == "PLAYER_LOSS":
                    status = "loss"
                else:
                    raise ValueError("terminal native outcome is neither victory nor loss")
                break
        flush_batch()
    except UnsupportedPolicyScopeError as exc:
        status, error = "unsupported", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # Keep failed episodes in the manifest denominator. Errors are neither
        # fallback actions nor native losses and make the probe fail its gate.
        status, error = "error", f"{type(exc).__name__}: {exc}"
    elapsed = max(0.0, clock() - started)
    observation = state.observation if state else {}
    return {
        "status": status, "seed": seed, "steps": steps, "step_attempts": attempts,
        "reload_count": 0, "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed else None,
        "final_floor": observation.get("run", {}).get("floor"),
        "final_hp": observation.get("player", {}).get("hp"), "error": error,
        "failure_context": {"stage": stage, "public": public.as_dict() if public else None,
            "search": getattr(policy, "last_search_diagnostic", None) if stage == "policy_choose" else None} if error else None,
        "search_audit": {"counts": dict(search_counts), "stop_reasons": dict(search_stops),
                         "refusal_reasons": dict(search_refusals), "examples": search_examples},
        "screens": dict(screens), "modes": dict(modes), "gaps": dict(gaps),
        "actions": dict(actions), "fallback_decisions": fallbacks,
        "unreported_fallback_decisions": unreported_fallbacks,
        "chosen_unknown_semantics": chosen_unknown, "fallback_examples": examples,
        "public_maps": dict(maps), "map_action_links": dict(map_links),
        "selected_event_display_statuses": dict(selected_event_statuses),
        "encoding": {"enabled": encoder is not None, "steps": encoded_steps, "batches": encoded_batches,
                     "unknown_categories": dict(encoding_unknowns), "gaps": dict(encoding_gaps),
                     "max_rows": dict(encoding_max_rows)},
        "observed_event_options": [{**json.loads(key), "count": count} for key, count in sorted(event_options.items())],
        "timings": {name: latency_summary(values) for name, values in samples.items()},
        "timing_samples": dict(samples),
    }


def summarize_probe(episodes: list[dict]) -> dict:
    statuses, screens, modes, gaps = Counter(), Counter(), Counter(), Counter()
    maps, map_links, event_options = Counter(), Counter(), Counter()
    selected_event_statuses = Counter()
    encoding_unknowns, encoding_gaps, encoding_max_rows = Counter(), Counter(), Counter()
    samples = defaultdict(list)
    search_totals = {name: Counter() for name in ("counts", "stop_reasons", "refusal_reasons")}
    for episode in episodes:
        statuses[episode["status"]] += 1
        for name, counts in search_totals.items():
            counts.update(episode.get("search_audit", {}).get(name, {}))
        for count, field in ((screens, "screens"), (modes, "modes"), (gaps, "gaps")):
            count.update(episode.get(field, {}))
        for name, values in episode.get("timing_samples", {}).items():
            samples[name].extend(values)
        maps.update(episode.get("public_maps", {}))
        map_links.update(episode.get("map_action_links", {}))
        selected_event_statuses.update(episode.get("selected_event_display_statuses", {}))
        encoded = episode.get("encoding", {})
        encoding_unknowns.update(encoded.get("unknown_categories", {}))
        encoding_gaps.update(encoded.get("gaps", {}))
        for name, count in encoded.get("max_rows", {}).items():
            encoding_max_rows[name] = max(encoding_max_rows[name], count)
        for row in episode.get("observed_event_options", ()):
            event_options[json.dumps({k: v for k, v in row.items() if k != "count"}, sort_keys=True)] += row["count"]
    complete = statuses["win"] + statuses["loss"]
    total_steps = sum(e.get("steps", 0) for e in episodes)
    elapsed = sum(e.get("elapsed_seconds", 0) for e in episodes)
    workers = sum(e.get("worker_wall_seconds", 0) for e in episodes)
    return {
        "attempts": len(episodes), "statuses": dict(statuses), "completed": complete,
        "observed_clear_fraction_all_attempts": statuses["win"] / len(episodes) if episodes else None,
        "control_flow_gate_passed": bool(episodes) and complete == len(episodes),
        "training_ready": False, "steps": total_steps, "episode_seconds": elapsed,
        "work_counters_complete_for_all_attempts": all("steps" in e for e in episodes),
        "worker_wall_seconds": workers,
        "steps_per_worker_second": total_steps / workers if workers and all("steps" in e for e in episodes) else None,
        "search_audit": {name: dict(counts) for name, counts in search_totals.items()},
        "search_counting_note": "native_calls are native-sampler dispatch attempts (preflight may refuse); completed_searches have at least one complete paired round; samples retain attempted-creation accounting; controller latency includes rule/potion overhead, failed calls and cold startup",
        "steps_per_second": total_steps / elapsed if elapsed else None,
        "screens": dict(screens), "modes": dict(modes), "gaps": dict(gaps),
        "fallback_decisions": sum(e.get("fallback_decisions", 0) for e in episodes),
        "unreported_fallback_decisions": sum(e.get("unreported_fallback_decisions", 0) for e in episodes),
        "chosen_unknown_semantics": sum(e.get("chosen_unknown_semantics", 0) for e in episodes),
        "public_maps": dict(maps), "map_action_links": dict(map_links),
        "selected_event_display_statuses": dict(selected_event_statuses),
        "encoding": {"steps": sum(e.get("encoding", {}).get("steps", 0) for e in episodes),
                     "batches": sum(e.get("encoding", {}).get("batches", 0) for e in episodes),
                     "unknown_categories": dict(encoding_unknowns), "gaps": dict(encoding_gaps),
                     "max_rows": dict(encoding_max_rows)},
        "observed_event_options": [{**json.loads(key), "count": count} for key, count in sorted(event_options.items())],
        "timings": {name: latency_summary(values) for name, values in samples.items()},
        "timing_note": "native_* calls nest inside environment_*; do not add overlapping totals; includes cold calls",
    }
