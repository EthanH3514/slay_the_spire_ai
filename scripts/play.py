"""Run the frozen Ironclad policy in the headless simulator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from allocation_probe import request as allocation_request
from autoplay_target_combat import validate_target_receipt
from current_combat import proof_validator
from duplicate_selection_options import install
from learned_continuation_contract import digest
from local_event_environment import LocalEventEnvironment
from potion_replacement_combat import PolicyEnvironment, decision_accounting
from root_recommendation import make_policy
from run_learning_combat_history_guard import validate_receipt
from spire_ai.canonical_event_dispatch import CanonicalEventOutside
from spire_ai.construction_learning import ConstructionLearner
from spire_ai.construction_policy import execute, policy_from_checkpoint
from spire_ai.expert_card_policy import RewardOrigin
from spire_ai.headless_crt import configure
from spire_ai.local_event_search import Budget, _take, eligibility, replay, search
from spire_ai.local_event_value import ConstructionEventValue
from spire_ai.policy_api import HierarchicalPolicy
from spire_ai.potion_replacement import constrain
from spire_ai.route_behavior import sample_route
from spire_ai.route_learning import RouteEncoder
from spire_ai.run_choice_model import RunChoiceEncoder
from spire_ai.run_choice_policy import state_hash
from spire_ai.run_learning_model import make_model, tensorize
from spire_ai.training_contract import TrainingConfig
from spire_ai.tree_capacity import TreeCapacity

MODELS = ROOT / "models"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Combat:
    def __init__(self, environment, files):
        self.module = environment.module
        # The native interface uses these identifiers to associate the model bindings.
        model_ids = {name: files[name]["sha256"] for name in ("prior", "value", "continuation")}
        actor = read(MODELS / files["continuation"]["file"])
        prior = read(MODELS / files["prior"]["file"])
        value = read(MODELS / files["value"]["file"])
        self.module.cold_bind(json.dumps(actor), json.dumps(actor["schema"]), model_ids["continuation"])
        self.module.neural_prior_bind(json.dumps(prior), json.dumps(prior["schema"]),
            digest(actor), model_ids["prior"], model_ids["continuation"])
        self.module.value_leaf_bind(json.dumps(value), json.dumps(value["model"]["schema"]),
            digest(value["model"]["parent"]["actor"]), model_ids["value"])
        self.module.lazy_rule_choice_enable(True)
        self.module.value_memo_enable(True)
        self.module.prior_memo_enable(True)
        self.search = make_policy(PolicyEnvironment(environment), "visits", "count").search
        self.capacity = TreeCapacity(1024, 3073, 128 * 1024 * 1024)
        self.search.tree_config = self.capacity.config(self.search.tree_config)
        self.validate_proof = proof_validator(self.capacity)
        configure_request = self.search.configure_request

        def configured(public, limits):
            return dict(configure_request(public, limits),
                history_work_guard="public-history-byte-work-guard-v1",
                allocation_probe=allocation_request("control"),
                terminal_proof="public-terminal-proof-v1",
                autoplay_target_prefix="public-autoplay-target-prefix-v1",
                resource_utility="combat-resource-utility-v2-linear")
        self.search.configure_request = configured
        self.search.validate_native = self.validate

    def validate(self, result, public):
        if result.get("refusal"):
            raise ValueError(f"Native search refused the public position: {result['refusal']}")
        self.capacity.validate(result)
        observed = decision_accounting(result)
        legality = [action.as_dict() for action in public.legal_actions]
        action = self.validate_proof(observed, legality, "proof")
        validate_target_receipt(result, True)
        validate_receipt(result["root_sampling"])
        tree = result["tree"]
        if (tree["actor_counts"]["calls"] or result["run_sampling"] != "public-run-sampling-v1"
                or result["offer_replay"] != "public-offer-replay-v1"):
            raise ValueError("Combat search contract drift")
        return action, ()

    def choose(self, public):
        self.search.sampler.result = None
        self.module.potion_replacement_enable(True)
        try:
            return self.search.rank(constrain(public)).action
        finally:
            self.module.potion_replacement_enable(False)


def choose_base(public, supplement, encoder, model, _candidate, *, eligible, arm, rng):
    if eligible or arm != "control":
        raise ValueError("Invalid base policy dispatch")
    encoded = encoder.encode(public, run_public=supplement)
    with torch.inference_mode():
        distribution, _ = model.distribution(tensorize([encoded], encoder))
    index = int(torch.multinomial(distribution.probs[0], 1, generator=rng))
    return encoded.decode(index), {"mode": "sampled_base"}


class FrozenPolicy:
    def __init__(self, module_dir):
        manifest = read(MODELS / "manifest.json")
        files = manifest["files"]
        saved = load(MODELS / files["construction"]["file"])
        self.construction = policy_from_checkpoint(saved)
        self.learner = ConstructionLearner(self.construction,
            config=TrainingConfig(**saved["config"]), startup_enabled=saved["startup_enabled"],
            inference_only=True)
        saved_route = saved["route"]
        self.route_encoder = RouteEncoder()
        self.route_model = make_model(self.route_encoder, width=saved_route["width"])
        self.route_model.load_state_dict(saved_route["model"], strict=True)
        self.route_model.eval().requires_grad_(False)
        self.route_identity = state_hash(self.route_model)
        if self.route_encoder.schema_id != saved_route["schema"]:
            raise ValueError("Route weight schema mismatch")
        encoder = RunChoiceEncoder().base
        base = load(MODELS / files["base"]["file"])
        if base["schema_id"] != encoder.schema_id:
            raise ValueError("Base weight schema mismatch")
        base_model = make_model(encoder, width=base["width"])
        base_model.load_state_dict(base["model"], strict=True)
        base_model.eval().requires_grad_(False)
        self.outside = CanonicalEventOutside.load(
            MODELS / files["event"]["file"], MODELS / files["boss"]["file"],
            self.construction, encoder, base_model, choose_base)
        self.event_value = ConstructionEventValue(self.construction.encoder, self.construction.model,
            objective=saved["objective"], checkpoint_sha256=files["construction"]["sha256"])
        self.environment = LocalEventEnvironment(self.construction.encoder, module_dir=module_dir)
        self.combat = Combat(self.environment, files)

    def play(self, seed, outside_seed, construction_seed, max_actions, on_action):
        env = self.environment
        env.reset(character="IRONCLAD", ascension=10, seed=seed)
        outside_rng = torch.Generator().manual_seed(outside_seed)
        construction_rng = torch.Generator().manual_seed(construction_seed)
        origin = RewardOrigin()
        actions = 0
        result = None

        def take(action, _detail=None):
            nonlocal actions, result
            public = env._current_public_input()
            if actions >= max_actions:
                raise ActionLimit
            result = _take(env, action)
            actions += 1
            if result.truncated:
                raise ValueError("Simulator truncated the run")
            on_action(actions, public, action, env._current_public_input())
            return result

        try:
            while result is None or not result.terminated:
                public = env._current_public_input()
                eligible = origin.observe(public)
                if actions >= max_actions:
                    raise ActionLimit
                if HierarchicalPolicy.specialist_for(public).value == "combat":
                    take(self.combat.choose(public))
                    continue
                if eligibility(public) is None:
                    plan = search(env, self.event_value, scorer_version=self.event_value.version,
                                  budget=Budget(max_steps=4096, max_depth=64))
                    replay(env, plan, take=take)
                    continue
                supplement = json.loads(env.session.public_run_supplement())
                screen = public.observation["screen"]["type"]
                shelf = json.loads(env.session.public_shop_shelf()) if screen == "SHOP" else None
                menu = self.construction.menu(public, supplement=supplement, shelf=shelf,
                                              reward_eligible=eligible)
                if menu is not None:
                    with torch.no_grad():
                        dist, _ = self.learner.distribution(self.learner.cache(menu))
                        selected = int(torch.multinomial(dist.probs, 1, generator=construction_rng))
                    execute(env, menu, selected, take=take)
                elif screen == "MAP":
                    action, _ = sample_route(public, supplement, self.route_encoder, self.route_model,
                                             generator=outside_rng, behavior_id=self.route_identity)
                    take(action)
                else:
                    action, _ = self.outside.choose(public, supplement, rng=outside_rng)
                    take(action)
        except ActionLimit:
            return actions, "action_limit"
        return actions, env._current_public_input().observation["run"]["outcome"]


class ActionLimit(Exception):
    """Requested bounded preview reached its action limit."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--outside-seed", type=int, default=1)
    parser.add_argument("--construction-seed", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--module-dir", type=Path, default=ROOT / "runtime/native")
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    configure()
    install()
    torch.set_num_threads(1)
    policy = FrozenPolicy(args.module_dir)

    def print_action(i, public, action, after):
        print(json.dumps(dict(index=i, floor=public.observation["run"]["floor"],
                              screen=public.observation["screen"]["type"],
                              action=action.as_dict(), fingerprint=after.fingerprint())), flush=True)

    count, outcome = policy.play(args.seed, args.outside_seed, args.construction_seed,
                                 args.max_actions, print_action)
    print(json.dumps(dict(actions=count, outcome=outcome)), flush=True)


if __name__ == "__main__":
    main()
