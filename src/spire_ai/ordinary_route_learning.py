"""Route PPO on fresh complete three-act games, with the existing frozen encoder."""
import torch

from .advantage_normalization import VERSION as NORMALIZATION
from .ordinary_reward_targets import DEFAULT, OrdinaryEpisode, returns, validate
from .policy_api import PolicyInput
from .route_behavior import route_input, validate_record
from .route_learning import RouteLearner, frozen_hash, VERSION as SAMPLE_VERSION
from .run_choice_policy import state_hash
from .run_learning_model import tensorize
from .run_learning_rollout import RolloutStep, ppo_update

VERSION = "route-three-act-ppo-v1"


def stable_ppo():
    return ppo_update


_update = stable_ppo()


class OrdinaryRouteLearner(RouteLearner):
    """Keep old sampling/architecture; never reinterpret old Act1 episodes."""

    objective = DEFAULT.objective

    def update(self, episodes, *, generator):
        if self.inference_only:
            raise ValueError("inference-only route learner cannot update")
        if self.failed:
            raise ValueError("partially updated route learner cannot resume")
        steps, seen, exact_targets = [], set(), []
        self.model.eval()
        for episode in episodes:
            if not isinstance(episode, OrdinaryEpisode) or episode.schedule != DEFAULT:
                raise ValueError("matching ordinary route episode required")
            validate(episode)
            values = returns(episode)
            current = []
            for j, sample in zip(episode.sample_indices, episode.samples, strict=True):
                identity = sample["ordinal"]
                if (
                    sample["version"] != SAMPLE_VERSION
                    or sample["generation"] != self.generation
                    or sample["split"] != "train"
                    or type(identity) is not int
                    or identity < 0
                    or identity in seen
                    or not sample["behavior"]["training_admitted"]
                ):
                    raise ValueError("stale/duplicate/held-out route sample")
                seen.add(identity)
                raw = sample["public"]
                from .actions import Action

                public = PolicyInput(
                    raw["observation"],
                    tuple(Action.from_dict(a) for a in raw["legal_actions"]),
                )
                if dict(public.observation["run"]) != episode.boundaries[j].before:
                    raise ValueError("route sample does not match trajectory boundary")
                behavior = sample["behavior"]
                validate_record(
                    behavior,
                    public,
                    sample["supplement"],
                    self.encoder,
                    expected_behavior_id=self.snapshot,
                )
                encoded = route_input(public, sample["supplement"], self.encoder)
                with torch.no_grad():
                    dist, predicted = self.model.distribution(
                        tensorize([encoded], self.encoder)
                    )
                if abs(
                    float(predicted) - behavior["value"]
                ) > 2e-6 or not torch.allclose(
                    dist.probs[0],
                    torch.tensor(behavior["probabilities"]),
                    atol=2e-6,
                    rtol=0,
                ):
                    raise ValueError(
                        "route behavior weights/probabilities/value differ"
                    )
                current.append(
                    RolloutStep(
                        encoded,
                        behavior["action_row"],
                        True,
                        "route",
                        "MAP",
                        behavior["old_log_prob"],
                        behavior["value"],
                        0.0,
                        False,
                        False,
                    )
                )
            for i, step in enumerate(current):
                # Attribute only milestones between these consecutive route decisions.
                # At gamma=lambda=1, GAE then reproduces the remaining actual return.
                following = i + 1 < len(current)
                step.reward = values[i] - (values[i + 1] if following else 0.0)
                step.next_value = current[i + 1].value if following else 0.0
                step.terminated = not following
            steps.extend(current)
            exact_targets.extend(values)
        try:
            result = (
                _update(
                    self.model,
                    self.optimizer,
                    self.encoder,
                    steps,
                    self.config,
                    generator=generator,
                )
                if steps
                else dict(
                    samples=0,
                    optimizer_steps=0,
                    reason="no route decision before actual terminal",
                )
            )
            if frozen_hash(self.model) != self.frozen:
                raise ValueError("frozen route features changed")
        except BaseException:
            self.failed = True
            raise
        self.generation += 1
        self.snapshot = state_hash(self.model)
        self.model.eval()
        return dict(
            result,
            version=VERSION,
            generation=self.generation,
            episodes=len(episodes),
            objective=self.objective,
            target="remaining actual +1/+2/+3 rewards",
            target_sum=sum(exact_targets),
            encoder_unchanged=True,
            advantage_normalization=NORMALIZATION,
            model_sha256=self.snapshot,
        )
