"""Outcome-blind combat capture/selection and bounded fixed-controller diagnosis."""
from __future__ import annotations

from collections import Counter
import time

from .actions import Action, ActionKind
from .combat_sampling import BoundCombatSampler, public_fingerprint
from .combat_search import CombatSearchPolicy
from .fast_policy import FastBaselinePolicy
from .policy_api import policy_input_from_state
from .training_readiness import latency_summary


EVALUATION_VERSION = "frozen-run-combat-diagnostic-v2-potion-aware"


def hp(observation):
    player = observation.get("combat", {}).get("player") or observation.get("player", {})
    return int(player.get("current_hp", player.get("hp", 0)))


class RecordingEnvironment:
    """Observer wrapper. Only sanitized public snapshots go into the capture.

    No session attribute: the old timing probe must not replace this wrapper's
    native session. Its step/reset times include capture overhead, explicitly.
    """
    def __init__(self, environment, *, attempt_id, notify=None):
        self.environment, self.attempt_id = environment, attempt_id
        self.notify = notify or (lambda: None)
        self.actions, self.roots, self.deck_changes, self.choices = [], [], [], []
        self.state = None
        self.final_inventory = None

    def reset(self, **kwargs):
        if self.state is not None:
            raise ValueError("no reset/retry within an evaluation attempt")
        self.state = self.environment.reset(**kwargs)
        self._observe(None, self.state)
        self.notify()
        return self.state

    def step(self, action):
        previous = self.state
        public = policy_input_from_state(previous)
        if action not in public.legal_actions:
            raise ValueError("capture received an illegal action")
        result = self.environment.step(action)
        self.actions.append(action.as_dict())
        if "combat" not in public.observation:
            screen = public.observation.get("screen", {})
            screen_data = public.as_dict()["observation"].get("screen", {})
            options = screen_data.get("decision_options", []) + screen_data.get("card_reward_options", [])
            self.choices.append({"step": len(self.actions), "floor": public.observation.get("run", {}).get("floor"),
                "screen": screen.get("type"), "action": action.as_dict(),
                "option": next((dict(o) for o in options
                                if o.get("action") == action.as_dict()), None)})
        self.state = result.state
        changed = self._observe(previous, self.state)
        if changed or result.done or len(self.actions) % 25 == 0:
            self.notify()
        return result

    def _observe(self, previous, state):
        public = policy_input_from_state(state)
        observation = public.as_dict()["observation"]
        self.final_inventory = observation.get("inventory", {})
        previous_deck = previous.observation.get("inventory", {}).get("deck", ()) if previous else ()
        deck = state.observation.get("inventory", {}).get("deck", ())
        changed = False
        if previous is not None and deck != previous_deck:
            self.deck_changes.append({"step": len(self.actions), "floor": observation.get("run", {}).get("floor"),
                                      "deck": self.final_inventory.get("deck", [])})
        if "combat" in observation and (previous is None or "combat" not in previous.observation):
            combat = observation["combat"]
            self.roots.append({"id": f"{self.attempt_id}/combat-{len(self.roots):02d}",
                "prefix_length": len(self.actions), "public": public.as_dict(),
                "fingerprint": public_fingerprint(public), "hp": hp(observation),
                "floor": observation.get("run", {}).get("floor"),
                "encounter": "+".join(str(m.get("id", "UNKNOWN")) for m in combat.get("monsters", ()))})
            changed = True
        return changed

    def capture(self):
        return {"actions": self.actions, "roots": self.roots, "deck_changes": self.deck_changes,
                "choices": self.choices, "final_inventory": self.final_inventory,
                "final_player": policy_input_from_state(self.state).as_dict()["observation"].get("player", {})
                                if self.state else None}


def coverage_order(roots):
    """Greedy PUBLIC coverage only; outcome/ending HP/search results are ignored.

    Prefer unseen source policy, then unseen encounter, then new public feature
    tags. Stable root ID breaks ties. Identical fingerprints are audited once.
    """
    unique = {}
    for root in sorted(roots, key=lambda r: r["id"]):
        unique.setdefault(root["fingerprint"], root)
    pending, ordered, seen = list(unique.values()), [], set()

    def features(root):
        observation = root["public"]["observation"]
        run = observation.get("run", {})
        return {"policy:" + root["policy"], "encounter:" + root["encounter"],
                "act:" + str(run.get("act")), "room:" + str(run.get("room_type")),
                "hp_bucket:" + str(root["hp"] // 20)} | {
                    "card:" + str(c.get("id")) for c in observation.get("inventory", {}).get("deck", ())}

    while pending:
        selected = max(pending, key=lambda r: (
            int("policy:" + r["policy"] not in seen), int("encounter:" + r["encounter"] not in seen),
            len(features(r) - seen)))
        pending.remove(selected)
        seen.update(features(selected))
        ordered.append(selected)
    return ordered


def replay_root(environment, *, seed, actions, root):
    state = environment.reset(seed=seed, character="IRONCLAD", ascension=0)
    count = root["prefix_length"]
    if type(count) is not int or not 0 <= count <= len(actions):
        raise ValueError("invalid replay prefix length")
    for serialized in actions[:count]:
        result = environment.step(Action.from_dict(serialized))
        if result.done:
            raise ValueError("replay terminated before root")
        state = result.state
    public = policy_input_from_state(state)
    if public_fingerprint(public) != root["fingerprint"] or public.as_dict() != root["public"]:
        raise ValueError("public combat root did not reproduce exactly")
    return state


class _SearchRule:
    def __init__(self, search, rule):
        self.search, self.catalog, self.recipes = search, rule.catalog, rule.recipes

    def choose(self, public):
        return self.search.choose(public)


class ResourceAlignedSearchPolicy:
    """Same eligible resources, with potions inside native search/continuations.

    No heuristic may override a completed search decision after the fact.
    """
    def __init__(self, sampler, *, budget, sampling_seed, catalog=None, root_model="native-seed-conditioned-v1"):
        self.fast = FastBaselinePolicy(catalog=catalog)
        rule = self.fast.combat.rule
        self.search = CombatSearchPolicy(sampler, budget=budget, sampling_seed=sampling_seed, rollout=rule, root_model=root_model)
        self.fast.combat.rule = _SearchRule(self.search, rule)
        self.fast.combat.potion_override = False
        self.last_decision = None
        self.last_search_diagnostic = None

    def choose(self, public):
        self.last_decision = self.last_search_diagnostic = None
        self.search.last_report = None
        self.search.last_backend = {}
        self.search.native_search_calls = 0
        started = time.perf_counter()
        error = None
        try:
            action = self.fast.choose(public)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Includes a failing native call; no stale report from the previous
            # combat leaks into a route/reward/event decision.
            if self.search.last_backend:
                report = self.search.last_report
                fields = ("stop_reason", "fallback_reason", "samples", "branch_steps",
                          "value_evaluations", "rollout_policy_evaluations", "discarded_rollouts",
                          "completed_rounds", "elapsed_seconds", "rollout_version", "search_version")
                self.last_search_diagnostic = {
                    "native_calls": self.search.native_search_calls,
                    "backend": dict(self.search.last_backend),
                    "error": error,
                    "report": {k: getattr(report, k) for k in fields} if report else None,
                    "horizon_leaves": sum(c.truncated for c in report.candidates) if report else 0,
                    "controller_seconds": time.perf_counter() - started,
                }
        self.last_decision = dict(self.fast.last_decision)
        if self.last_search_diagnostic:
            report = self.search.last_report
            fallback = report.fallback_reason is not None
            if self.last_decision["mode"] == "combat_rule":
                self.last_decision["mode"] = "combat_search" if report.completed_rounds else "combat_search_rule_only"
            if report.completed_rounds and self.search.last_backend.get("session"):
                self.last_decision["gaps"] = [g for g in self.last_decision["gaps"]
                                               if g != "combat_selection_rule_only"]
                self.last_decision["root_service"] = self.search.last_backend["session"]["version"]
            self.last_decision.update(search_calls=self.search.native_search_calls,
                                      fallback=self.last_decision["fallback"] or fallback)
            if fallback:
                self.last_decision["gaps"] = sorted(set(self.last_decision["gaps"]) |
                                                     {"search_fallback:" + report.fallback_reason})
        return action


def audit_root(environment, state, *, budget, sampling_seed):
    public = policy_input_from_state(state)
    source = environment._read_state().as_dict()  # Audit-only, never supplied to a policy.
    try:
        search = CombatSearchPolicy(BoundCombatSampler(environment), budget=budget, sampling_seed=sampling_seed)
        reason = search.unsupported_root_reason(public)
        if reason:
            return {"status": "unsupported", "reason": reason}
        # A start-of-combat root must be a real normal choice, not a synthetic reveal.
        if public.observation.get("combat", {}).get("input_state") != "PLAYER_NORMAL":
            return {"status": "unsupported", "reason": "non_normal_combat_start"}
        branch = BoundCombatSampler(environment).sample(public, sample_seed=sampling_seed)
        del branch
        return {"status": "supported"}
    finally:
        if environment._read_state().as_dict() != source:
            raise ValueError("root support probe mutated its source")


def combat_episode(environment, state, *, policy_name, budget, sampling_seed, max_steps, notify=None):
    if policy_name not in {"fast", "fast_with_search"}:
        raise ValueError("unknown combat comparator")
    policy = (FastBaselinePolicy() if policy_name == "fast" else ResourceAlignedSearchPolicy(
        BoundCombatSampler(environment), budget=budget, sampling_seed=sampling_seed))
    report = {"status": "running", "policy": policy_name, "root_hp": hp(state.observation),
              "decisions": [], "potions_used": 0, "potions_discarded": 0,
              "replay_is_offline_diagnostic_not_gameplay_sl": True}
    notify = notify or (lambda report: None)
    notify(report)
    for _ in range(max_steps):
        public = policy_input_from_state(state)
        source = environment._read_state().as_dict()
        at = time.perf_counter()
        action = policy.choose(public)
        elapsed = time.perf_counter() - at
        if environment._read_state().as_dict() != source or action not in public.legal_actions:
            raise ValueError("combat decision mutated source or returned illegal action")
        search_report = policy.search.last_report.as_dict() if policy_name == "fast_with_search" else None
        report["decisions"].append({"fingerprint": public_fingerprint(public), "action": action.as_dict(),
                                     "elapsed_seconds": elapsed, "search": search_report})
        report["potions_used"] += action.kind is ActionKind.USE_POTION
        report["potions_discarded"] += action.kind is ActionKind.DISCARD_POTION
        result = environment.step(action)
        state = result.state
        report.update(ending_hp=hp(state.observation), max_hp=state.observation.get("player", {}).get("max_hp"),
                      native_combat_outcome=state.info.get("combat_outcome"))
        if result.truncated:
            report["status"] = "truncated"
        elif state.observation.get("run", {}).get("outcome") == "PLAYER_LOSS" or hp(state.observation) <= 0:
            report["status"] = "loss"
        elif state.info.get("combat_outcome") == "PLAYER_ESCAPED":
            report["status"] = "escaped"
        elif "combat" not in state.observation:
            if state.info.get("combat_outcome") not in {None, "PLAYER_VICTORY"}:
                raise ValueError("unexpected native combat exit outcome")
            report["status"] = "win"
        elif result.terminated:
            raise ValueError("unrecognized terminal combat result")
        notify(report)
        if report["status"] != "running":
            break
    else:
        report["status"] = "truncated"
    report["timings"] = latency_summary([d["elapsed_seconds"] for d in report["decisions"]])
    report["hp_note"] = "net HP after exit includes Burning Blood/other exit effects, NOT raw damage taken"
    notify(report)
    return report


def paired_summary(attempts):
    """Failures stay in denominators, never converted to deaths or dropped silently."""
    pairs = {}
    for attempt in attempts:
        pairs.setdefault(attempt["root_id"], {})[attempt["policy"]] = attempt
    rows = []
    for root_id, pair in pairs.items():
        if set(pair) != {"fast", "fast_with_search"}:
            continue
        fast, search = pair["fast"], pair["fast_with_search"]
        completed = all(p["status"] in {"win", "loss", "escaped"} for p in (fast, search))
        rows.append({"root_id": root_id, "completed_pair": completed,
            "fast_status": fast["status"], "search_status": search["status"],
            "net_exit_hp_difference": search["ending_hp"] - fast["ending_hp"] if completed else None,
            "potion_use_difference": search["potions_used"] - fast["potions_used"] if completed else None})
    return {"attempts": len(attempts), "statuses": dict(Counter(a["status"] for a in attempts)),
            "completed_pairs": sum(r["completed_pair"] for r in rows), "pairs": rows,
            "teacher_strength_established": False}
