"""Continue the released deck-building and route policies on complete runs."""

from dataclasses import asdict

import torch

from play import FrozenPolicy
from potion_replacement_combat import _with_dependencies
from spire_ai.construction_learning import ConstructionLearner
from spire_ai.ordinary_reward_targets import (
    DEFAULT,
    OrdinaryEpisode,
    complete_episode,
    targets,
    validate,
)
from spire_ai.ordinary_route_learning import OrdinaryRouteLearner
from spire_ai.run_choice_policy import state_hash
from spire_ai.run_learning_encoding import RunLearningEncoder
from spire_ai.run_learning_model import make_model
from spire_ai.training_contract import TrainingConfig


class ConstructionTrainer(ConstructionLearner):
    _validate = _with_dependencies(
        ConstructionLearner._validate,
        episode_objective=lambda episode: episode.schedule.objective
        if isinstance(episode, OrdinaryEpisode)
        else None,
    )
    _update = _with_dependencies(ConstructionLearner.update, terminal_targets=targets)

    def __init__(self, policy, **settings):
        super().__init__(policy, **settings)
        self.objective = DEFAULT.objective

    def update(self, episodes, *, generator):
        report = self._update(episodes, generator=generator)
        report.pop("stage_clears")
        report["target"] = "remaining +1/+2/+3 ordinary-act rewards"
        report["act_clears"] = {
            str(act): sum(len(validate(episode)) >= act for episode in episodes)
            for act in range(1, 4)
        }
        return report


class OutsideTrainingPolicy(FrozenPolicy):
    def __init__(self, module_dir, checkpoint, combat_weights):
        super().__init__(
            module_dir, construction_state=checkpoint, combat_weights=combat_weights
        )
        if checkpoint["objective"] != DEFAULT.objective:
            raise ValueError("checkpoint has a different run reward objective")
        config = TrainingConfig(**checkpoint["config"])
        self.learner = ConstructionTrainer(
            self.construction,
            config=config,
            startup_enabled=checkpoint["startup_enabled"],
        )
        self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        self.construction_generator = torch.Generator().set_state(
            checkpoint["generator"]
        )

        saved_route = checkpoint["route"]
        route_config = TrainingConfig(**saved_route["config"])
        base = make_model(RunLearningEncoder(), width=route_config.width)
        base.load_state_dict(saved_route["model"], strict=True)
        self.route_learner = OrdinaryRouteLearner(base, config=route_config)
        self.route_learner.model.load_state_dict(saved_route["model"], strict=True)
        self.route_learner.snapshot = state_hash(self.route_learner.model)
        self.route_learner.optimizer.load_state_dict(saved_route["optimizer"])
        self.route_generator = torch.Generator().set_state(saved_route["generator"])

    def choose_construction(self, menu, generator):
        sample = self.learner.sample(
            self.learner.cache(menu), split="train", generator=generator
        )
        self.construction_choices.append((len(self.environment.boundaries), sample))
        return sample.selected

    def choose_route(self, public, supplement, generator):
        action, sample = self.route_learner.sample(
            public, supplement, generator=generator
        )
        self.route_choices.append((len(self.environment.boundaries), sample))
        return action

    def collect(self, seed, max_actions):
        self.construction_choices = []
        self.route_choices = []
        count, outcome = self.play(
            seed, 2 * seed + 1, 2 * seed + 2, max_actions, lambda *_: None
        )
        if outcome == "action_limit":
            return count, outcome, None, None
        boundaries = self.environment.boundaries
        if len(boundaries) != count:
            raise ValueError(
                "public reward milestones do not align with executed actions"
            )
        episodes = []
        for choices in (self.construction_choices, self.route_choices):
            indices = [index for index, _ in choices]
            samples = [sample for _, sample in choices]
            episodes.append(complete_episode(samples, indices, boundaries))
        return count, outcome, *episodes

    def update(self, construction_episodes, route_episodes, checkpoint):
        if any(episode.samples for episode in construction_episodes):
            construction_report = self.learner.update(
                construction_episodes, generator=self.construction_generator
            )
        else:
            construction_report = {
                "samples": 0,
                "optimizer_steps": 0,
                "reason": "no construction choices in these runs",
            }
        route_report = self.route_learner.update(
            route_episodes, generator=self.route_generator
        )
        if (
            not construction_report["optimizer_steps"]
            and not route_report["optimizer_steps"]
        ):
            raise ValueError("no construction or route decisions to update")
        route = self.route_learner
        state = dict(
            checkpoint,
            model=self.learner.model.state_dict(),
            optimizer=self.learner.optimizer.state_dict(),
            generator=self.construction_generator.get_state(),
            route=dict(
                checkpoint["route"],
                model=route.model.state_dict(),
                optimizer=route.optimizer.state_dict(),
                config=asdict(route.config),
                generator=self.route_generator.get_state(),
            ),
        )
        return state, {"construction": construction_report, "route": route_report}
