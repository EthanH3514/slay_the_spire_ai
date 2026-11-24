"""Versioned run-input PPO collector; frozen legacy implementation is retained.

PPO/GAE math is copied unchanged from training_rollout.py. Only the tensor
adapter, explicit frozen controller and public trajectory callbacks differ.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time

import torch
from .advantage_normalization import normalize_advantages

from .run_learning_model import tensorize
from .policy_api import HierarchicalPolicy, policy_input_from_state
from .public_encoding import EncodedDecision
from .training_contract import TrainingReward, generalized_advantages


class FixedCombatAudit:
    """The new distribution controller must not masquerade as old flat search."""
    def __init__(self):
        self.counts = Counter()
        self.seconds = 0.0

    def record(self, public, policy, elapsed, error=None):
        self.seconds += elapsed
        self.counts["attempts"] += 1
        native = policy.last_native
        if error:
            self.counts["errors"] += 1
        elif native and native.get("forced"):
            self.counts["forced"] += 1
        elif native and native.get("steps", 0) > 0:
            self.counts["searched"] += 1
            self.counts["simulator_steps"] += native["steps"]
        else:
            raise ValueError("frozen controller omitted search evidence")

    def summary(self):
        return dict(counts=dict(self.counts), seconds=self.seconds, fallback="refuse")


@dataclass
class RolloutStep:
    encoded: EncodedDecision
    action_row: int
    learned: bool
    specialist: str
    screen: str
    old_log_prob: float | None
    value: float
    reward: float
    terminated: bool
    truncated: bool
    next_value: float = 0.0


class RolloutCollector:
    def __init__(self, environment, model, encoder, seeds, config, *, fixed, supplement,
                 capture=None, notify=None, generator=None):
        self.environment, self.model, self.encoder = environment, model, encoder
        self.seeds, self.config = seeds, config
        config.validate()
        self.fixed_contract = fixed.contract
        if self.fixed_contract.get("actor_training_labels") is not False:
            raise ValueError("fixed controller must not supply actor labels")
        self.fixed, self.supplement, self.capture = fixed, supplement, capture
        self.last_supplement = None
        self.combat_audit = FixedCombatAudit()
        self.pending_decision = None
        self.failure_snapshot = None
        self.reward = TrainingReward(config.progress_cap)
        self.generator = generator or torch.Generator().manual_seed(config.model_seed + 1)
        self.notify = notify or (lambda: None)
        self.state = None
        self.episodes: list[dict] = []
        self.steps = 0
        self.counts, self.unknowns, self.gaps, self.timings = Counter(), Counter(), Counter(), Counter()
        self.first_input = None

    def _encode(self, public):
        self.last_supplement = self.supplement()
        return self.encoder.encode(public, run_public=self.last_supplement)

    def prepare_initial_probe(self):
        """Start the first episode once, without choosing an action or drawing policy RNG."""
        if self.steps != 0:
            raise ValueError("initial probe must precede collection")
        if self.state is None:
            if self.episodes:
                raise ValueError("cannot retry a failed initial reset")
            self._reset()
        if self.first_input is None:
            encoded = self._encode(policy_input_from_state(self.state))
            self.first_input = tensorize([encoded], self.encoder).payload()
        return self.first_input

    def _reset(self):
        seed = self.seeds.next_seed()
        self.episodes.append({"seed": seed, "status": "running", "steps": 0,
                              "reload_count": 0, "terminal_reward": 0.0, "progress_reward": 0.0})
        self.counts["reset_attempts"] += 1
        self.pending_decision = {"stage": "environment_reset", "episode_index": len(self.episodes) - 1}
        self.notify()  # Account for the attempt before native reset can crash/block.
        self.state = self.environment.reset(seed=seed, character="IRONCLAD", ascension=0)
        self.pending_decision = None
        self.reward.reset(self.state.observation)
        if self.capture:
            self.capture.reset(seed, self.state, self.fixed_contract)

    def _end(self, status, reason=None):
        episode = self.episodes[-1]
        episode.update(status=status, final_floor=self.state.observation.get("run", {}).get("floor"),
                       final_hp=self.state.observation.get("player", {}).get("hp"))
        if reason:
            episode["reason"] = reason
        if self.capture:
            self.capture.finish(episode, self.state)
        self.state = None
        self.notify()

    def collect(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("positive rollout step count required")
        records = []
        self.model.eval()
        for _ in range(count):
            if self.steps >= self.config.max_env_steps:
                break
            if self.state is None:
                self._reset()
            at = time.perf_counter()
            public = policy_input_from_state(self.state)
            encoded = self._encode(public)
            batch = tensorize([encoded], self.encoder)
            self.timings["public_tensor_seconds"] += time.perf_counter() - at
            if self.first_input is None:
                self.first_input = batch.payload()
            at = time.perf_counter()
            with torch.no_grad():
                distribution, value = self.model.distribution(batch)
                value = float(value.item())
            self.timings["inference_seconds"] += time.perf_counter() - at
            # A preceding nonterminal step bootstraps from this exact public state,
            # not the new episode after reset and not a hidden simulator clone.
            if records and not records[-1].terminated and not records[-1].truncated:
                records[-1].next_value = value
            specialist = HierarchicalPolicy.specialist_for(public).value
            learned = specialist != "combat"
            if learned:
                row = int(torch.multinomial(distribution.probs[0], 1, generator=self.generator).item())
                log_prob = float(distribution.log_prob(torch.tensor([row])).item())
                action = encoded.decode(row)
                self.counts[specialist + "_samples"] += 1
                self.counts[specialist + "_nontrivial_choices"] += int(batch.action_mask.sum() > 1)
            else:
                self.pending_decision = {"stage": "fixed_combat", "floor": public.observation.get("run", {}).get("floor"),
                                         "fingerprint": public.fingerprint()}
                self.notify()  # Persist before a native call can block/crash.
                at = time.perf_counter()
                error = None
                try:
                    action = self.fixed.choose(public)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    elapsed = time.perf_counter() - at
                    self.timings["fixed_combat_seconds"] += elapsed
                    self.combat_audit.record(public, self.fixed, elapsed, error)
                row, log_prob = public.legal_actions.index(action), None
                if encoded.decode(row) != action:
                    raise ValueError("fixed action failed public round trip")
                self.counts["fixed_combat_samples"] += 1
            screen = str(public.observation.get("screen", {}).get("type", "UNKNOWN"))
            self.counts["screen:" + screen] += 1
            self.unknowns.update(encoded.unknown_categories)
            self.gaps.update(encoded.gaps)
            self.pending_decision = {"stage": "environment_step", "action": action.as_dict(),
                                     "fingerprint": public.fingerprint()}
            self.counts["step_attempts"] += 1
            self.notify()
            at = time.perf_counter()
            if self.capture:
                self.capture.begin(public, self.last_supplement, action, learned, log_prob, value, self.fixed)
            result = self.environment.step(action)
            if self.capture:
                self.capture.append(result)
            self.timings["environment_step_seconds"] += time.perf_counter() - at
            self.steps += 1
            episode = self.episodes[-1]
            episode["steps"] += 1
            components = self.reward.step(result)
            episode["terminal_reward"] += components["terminal"]
            episode["progress_reward"] += components["progress"]
            self.state = result.state
            self.pending_decision = None
            truncated = bool(result.truncated or (not result.terminated and
                             episode["steps"] >= self.config.max_episode_steps))
            record = RolloutStep(encoded, row, learned, specialist, screen, log_prob, value,
                                 components["total"], bool(result.terminated), truncated)
            records.append(record)
            if result.terminated:
                self._end("win" if components["terminal"] > 0 else "loss")
            elif truncated:
                record.next_value = self._bootstrap()
                self._end("truncated", "environment_truncation" if result.truncated else "episode_step_budget")
        if records and not records[-1].terminated and not records[-1].truncated:
            records[-1].next_value = self._bootstrap()
        if not records:
            raise ValueError("no transitions collected")
        return records

    def _bootstrap(self):
        encoded = self._encode(policy_input_from_state(self.state))
        with torch.no_grad():
            _, value = self.model(tensorize([encoded], self.encoder))
        return float(value.item())

    def close(self, *, error=None):
        if error and self.state is not None and self.failure_snapshot is None:
            from .public_failure_capture import public_failure_snapshot
            self.failure_snapshot = public_failure_snapshot(self.state,
                stage=(self.pending_decision or {}).get("stage", "training"), error=error)
        if self.episodes and self.episodes[-1]["status"] == "running":
            if self.state is None:
                self.episodes[-1].update(status="error", reason=error or "reset_failed")
                self.notify()
            else:
                self._end("error" if error else "truncated", error or "training_budget")

    def summary(self):
        return {"env_steps": self.steps, "counts": dict(self.counts), "timings": dict(self.timings),
                "unknown_categories": dict(self.unknowns), "encoding_gaps": dict(self.gaps),
                "fixed_combat": self.fixed_contract, "combat_audit": self.combat_audit.summary(),
                "pending_decision": self.pending_decision, "failure_snapshot": self.failure_snapshot,
                "episodes": self.episodes, "episode_statuses": dict(Counter(e["status"] for e in self.episodes)),
                "reload_count": 0}


def ppo_update(model, optimizer, encoder, records, config, *, generator, auxiliary_loss=None):
    """Actor loss ONLY on actual learner samples; value loss on the whole trace."""
    advantages, targets = generalized_advantages(
        [s.reward for s in records], [s.value for s in records], [s.next_value for s in records],
        [s.terminated for s in records], [s.truncated for s in records],
        gamma=config.gamma, lam=config.gae_lambda)
    actor_mask = torch.tensor([s.learned for s in records], dtype=torch.bool)
    advantage = torch.tensor(advantages, dtype=torch.float32)
    # Never normalize actor advantages using fixed-combat samples.
    if actor_mask.sum() > 1:
        selected = advantage[actor_mask]
        advantage[actor_mask] = normalize_advantages(selected)
    target = torch.tensor(targets, dtype=torch.float32)
    if any(s.learned and (s.old_log_prob is None or not math.isfinite(s.old_log_prob)) for s in records):
        raise ValueError("missing learner behavior probability")
    if any(not s.learned and s.old_log_prob is not None for s in records):
        raise ValueError("fixed combat must not be labeled as learner behavior")
    old_log = torch.tensor([s.old_log_prob if s.learned else 0.0 for s in records])
    model.train()
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    logs = []
    stopped_for_kl = False
    for _ in range(config.epochs):
        order = torch.randperm(len(records), generator=generator)
        for start in range(0, len(records), config.minibatch_size):
            ids = order[start:start + config.minibatch_size]
            batch_records = [records[i] for i in ids.tolist()]
            batch = tensorize([s.encoded for s in batch_records], encoder)
            distribution, values = model.distribution(batch)
            active = actor_mask[ids]
            value_loss = .5 * (values - target[ids]).square().mean()
            policy_loss = values.sum() * 0
            entropy = policy_loss
            kl = values.new_zeros(())
            if active.any():
                # Indexing active rows also avoids fake likelihood ratios for rules.
                logits = distribution.logits[active]
                actor = torch.distributions.Categorical(logits=logits)
                selected_rows = torch.tensor([s.action_row for s in batch_records])[active]
                log_ratio = actor.log_prob(selected_rows) - old_log[ids][active]
                ratio = log_ratio.exp()
                if not torch.isfinite(ratio).all():
                    raise ValueError("non-finite PPO ratio")
                adv = advantage[ids][active]
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv).mean()
                entropy = actor.entropy().mean()
                kl = ((ratio - 1) - log_ratio).mean()
                if float(kl.detach()) > config.target_kl:
                    stopped_for_kl = True
                    break
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            # Opt-in experiment only. No extra step/sampling after the PPO KL
            # guard, no demonstration labels in GAE or behavior likelihoods.
            auxiliary = None
            if auxiliary_loss is not None:
                auxiliary = auxiliary_loss(model)
                if not isinstance(auxiliary, torch.Tensor) or auxiliary.ndim != 0:
                    raise ValueError("auxiliary PPO loss must be a scalar tensor")
                loss = loss + auxiliary
            if not torch.isfinite(loss):
                raise ValueError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            logs.append({"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                         "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                         "approx_kl": float(kl.detach()), "gradient_norm": float(grad),
                         "actor_samples": int(active.sum())})
            if auxiliary is not None:
                logs[-1]["auxiliary_loss"] = float(auxiliary.detach())
        if stopped_for_kl:
            break
    if not logs:
        raise ValueError("PPO made no optimizer step")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("non-finite parameters after update")
    changes = {}
    for group in ("run_head", "route_head", "value_head", "shared"):
        changes[group] = sum(float((p.detach() - initial[name]).square().sum()) for name, p in model.named_parameters()
            if (name.split(".")[0] == group if group != "shared" else
                name.split(".")[0] not in {"run_head", "route_head", "value_head"})) ** .5
    return {"samples": len(records), "actor_samples": int(actor_mask.sum()),
            "fixed_combat_samples": int((~actor_mask).sum()), "optimizer_steps": len(logs),
            "stopped_for_kl": stopped_for_kl, "parameter_l2_changes": changes,
            "learner_screens": dict(Counter(s.screen for s in records if s.learned)),
            "specialists": {name: {"samples": sum(s.specialist == name for s in records),
                "mean_return": sum(t for s, t in zip(records, targets) if s.specialist == name) /
                               max(1, sum(s.specialist == name for s in records))}
                for name in ("run", "route", "combat")},
            "mean_metrics": {key: sum(row[key] for row in logs) / len(logs)
                             for key in (("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm")
                                         + (("auxiliary_loss",) if auxiliary_loss is not None else ()))}}
