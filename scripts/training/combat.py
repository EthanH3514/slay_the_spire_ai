"""Fit the released combat heads using fresh public decisions and fight results."""

import json

import torch

from learned_continuation_contract import canonical
from play import FrozenPolicy, MODELS, read
from spire_ai.native_search import session_projection_json
from spire_ai.potion_replacement import constrain

from . import critic_state_value, escape_model, player_value
from .combat_batches import update
from .rich_combat import Encoder, pack


def classify_prior(public, report):
    if report is None or report["terminal_proof"]["changed"]:
        return None
    counts = {
        canonical(edge["action"]): edge["visits"] for edge in report["root_evidence"]
    }
    legal = [canonical(action.as_dict()) for action in public.legal_actions]
    if counts.keys() != set(legal):
        raise ValueError("search visits do not match the public legal menu")
    visits = [counts[key] for key in legal]
    total = sum(visits)
    if total != report["tree"]["completed_simulations"] or not total:
        raise ValueError("invalid root visit total")
    return [count / total for count in visits]


class CombatTrainer:
    def __init__(self, checkpoint, *, training):
        self.prior = escape_model.load(read(MODELS / "prior.json"))
        self.value_artifact = read(MODELS / "value.json")
        self.value = critic_state_value.load(self.value_artifact["model"])
        self.prior.load_state_dict(checkpoint["prior"]["model"], strict=True)
        self.value.load_state_dict(checkpoint["value"]["model"], strict=True)
        self.optimizers = {}
        if training:
            self.generator = torch.Generator().set_state(checkpoint["generator"])
            for kind, model, parameters in (
                ("prior", self.prior, self.prior.actor_parameters()),
                ("value", self.value, self.value.value_parameters()),
            ):
                model.requires_grad_(False)
                for parameter in parameters:
                    parameter.requires_grad_(True)
                optimizer = torch.optim.Adam(parameters)
                optimizer.load_state_dict(checkpoint[kind]["optimizer"])
                self.optimizers[kind] = optimizer

    def weights(self):
        return escape_model.export(self.prior), dict(
            self.value_artifact, model=critic_state_value.export(self.value)
        )

    def fit(self, positions, batch_size):
        report, state = {}, {}
        for kind, model in (("prior", self.prior), ("value", self.value)):
            rows = positions[kind]
            if not rows:
                raise ValueError(f"no completed combat supplied {kind} labels")
            optimizer = self.optimizers[kind]
            order = torch.randperm(len(rows), generator=self.generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [rows[index] for index in order[start : start + batch_size]]
                losses.append(update(model, optimizer, batch, kind))
            report[kind] = {
                "positions": len(rows),
                "updates": len(losses),
                "mean_loss": sum(losses) / len(losses),
            }
            state[kind] = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        state["generator"] = self.generator.get_state()
        return state, report


class CombatTrainingPolicy(FrozenPolicy):
    def __init__(self, module_dir, construction_state, trainer):
        super().__init__(
            module_dir,
            construction_state=construction_state,
            combat_weights=trainer.weights(),
        )
        if trainer.prior.spec != trainer.value.spec:
            raise ValueError("combat prior and value use different public encoders")
        self.encoder = Encoder(trainer.prior.spec)
        self.rows = {"prior": [], "value": []}
        self.battles = 0

    def choose_combat(self, public):
        action = super().choose_combat(public)
        public = constrain(public)
        view = json.loads(session_projection_json(public))
        rich = json.loads(self.environment.session.rich_public_view())
        legal = [action.as_dict() for action in public.legal_actions]
        packed = pack(self.encoder.encode(view, legal, rich))
        prior = classify_prior(public, self.combat.search.sampler.result)
        row = dict(
            packed=packed,
            player=player_value.numbers(view),
            critic_state=critic_state_value.prepare_state(packed["old"]),
        )
        self.pending.append((row, prior))
        return action

    def record_outcome(self, count, before, action, after):
        boundary = self.environment.boundaries[-1]
        if boundary.index != count - 1:
            raise ValueError("combat labels do not align with executed actions")
        if boundary.battle is None:
            return
        value = escape_model.terminal_target(boundary.battle)
        for row, prior in self.pending:
            self.rows["value"].append(dict(row, label=value))
            if prior is not None:
                self.rows["prior"].append(
                    dict(packed=row["packed"], visit_target=prior)
                )
        self.pending.clear()
        self.battles += 1

    def collect(self, seed, max_actions):
        self.pending = []
        count, outcome = self.play(
            seed, 2 * seed + 1, 2 * seed + 2, max_actions, self.record_outcome
        )
        self.pending.clear()
        return count, outcome
