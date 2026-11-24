"""Small opt-in route-head PPO, using sealed complete Act1 outcomes.

The existing entity encoder/backbone and other policies stay frozen. Only the
route output and its value head learn. Ascension is masked in a new input schema.
Terminal credit is undiscounted over route choices, not diluted by combat clicks.
"""
import copy
from dataclasses import replace
import hashlib
import json
from types import MappingProxyType

import torch

from .actions import Action
from .construction_learning import complete_act1_episode, tensor_hash
from .policy_api import PolicyInput
from .public_encoding import FeatureTable
from .route_behavior import route_input, sample_route, validate_record
from .run_choice_policy import state_hash
from .run_learning_encoding import RunLearningEncoder
from .run_learning_model import make_model, tensorize
from .run_learning_rollout import RolloutStep, ppo_update
from .training_contract import TrainingConfig

VERSION = 'route-head-complete-act1-ppo-v1'


class RouteEncoder(RunLearningEncoder):
    def __init__(self):
        super().__init__()
        self.source_schema = self.schema_id
        self.schema_id = hashlib.sha256(json.dumps(dict(version=VERSION, source=self.source_schema,
            mask=['global.ascension.value', 'global.ascension.known']), sort_keys=True).encode()).hexdigest()

    def encode(self, public, *, run_public=None):
        encoded = super().encode(public, run_public=run_public)
        table = encoded.tables['global']; index = table.columns.index('ascension')
        values, known = [list(r) for r in table.values], [list(r) for r in table.known]
        for row, mask in zip(values, known, strict=True): row[index], mask[index] = 0., False
        tables = dict(encoded.tables)
        tables['global'] = FeatureTable(table.columns, tuple(map(tuple, values)), tuple(map(tuple, known)))
        return replace(encoded, tables=MappingProxyType(tables))


def frozen_hash(model):
    h=hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(('route_head.', 'value_head.')):
            h.update(name.encode()); h.update(tensor_hash(value).encode())
    return h.hexdigest()


class RouteLearner:
    def __init__(self, base, *, config=None, inference_only=False):
        self.encoder = RouteEncoder()
        if base.schema_id != self.encoder.source_schema:
            raise ValueError('reviewed base run/route schema required')
        self.model = make_model(self.encoder, width=base.width)
        self.model.load_state_dict(base.state_dict(), strict=True)
        self.model.eval().requires_grad_(False)
        self.model.route_head.requires_grad_(True)
        self.model.value_head.requires_grad_(True)
        # New objective; keep the nonzero hidden layer, reset scalar output only.
        with torch.no_grad():
            self.model.value_head[-1].weight.zero_(); self.model.value_head[-1].bias.zero_()
        self.config = config or TrainingConfig(width=base.width, gamma=1., gae_lambda=1., progress_cap=0.,
                                               epochs=2, minibatch_size=32, learning_rate=.0003)
        if (self.config.gamma, self.config.gae_lambda, self.config.progress_cap)!=(1.,1.,0.):
            raise ValueError('complete-episode undiscounted objective required')
        self.inference_only = inference_only
        self.optimizer = None if inference_only else torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.config.learning_rate)
        if inference_only: self.model.requires_grad_(False)
        self.frozen = frozen_hash(self.model)
        self.generation = 0; self.ordinal = 0; self.failed = False
        self.snapshot = state_hash(self.model)

    def sample(self, public, supplement, *, generator, split='train'):
        if self.failed or split!='train': raise ValueError('active TRAIN route learner required')
        self.model.eval()
        action, behavior = sample_route(public, supplement, self.encoder, self.model,
            generator=generator, behavior_id=self.snapshot)
        behavior['training_admitted'] = True
        sample = dict(version=VERSION, public=public.as_dict(), supplement=copy.deepcopy(supplement),
            behavior=behavior, generation=self.generation, ordinal=self.ordinal, split=split)
        self.ordinal += 1
        return action, sample

    def episode(self, samples, **terminal):
        return complete_act1_episode(samples, **terminal)

    def update(self, episodes, *, generator):
        if self.inference_only: raise ValueError('inference-only route learner cannot update')
        if self.failed: raise ValueError('partially updated learner cannot resume')
        steps, seen = [], set()
        self.model.eval()
        for episode in episodes:
            check = self.episode(episode.samples, before_run=episode.before_run, after_run=episode.after_run,
                battle_result=episode.battle_result, terminated=episode.terminated,
                truncated=episode.truncated, split=episode.split)
            if check.outcome!=episode.outcome: raise ValueError('route objective result changed')
            current=[]
            for sample in episode.samples:
                identity=sample['ordinal']
                if (sample['version']!=VERSION or sample['generation']!=self.generation or sample['split']!='train' or
                        type(identity) is not int or identity<0 or identity in seen or not sample['behavior']['training_admitted']):
                    raise ValueError('stale/duplicate/held-out route sample')
                seen.add(identity)
                p=PolicyInput(sample['public']['observation'],tuple(Action.from_dict(a) for a in sample['public']['legal_actions']))
                behavior=sample['behavior']; supplement=sample['supplement']
                validate_record(behavior,p,supplement,self.encoder,expected_behavior_id=self.snapshot)
                encoded=route_input(p,supplement,self.encoder)
                with torch.no_grad():
                    dist,value=self.model.distribution(tensorize([encoded],self.encoder))
                if (abs(float(dist.log_prob(torch.tensor([behavior['action_row']])))-behavior['old_log_prob'])>2e-6 or
                        abs(float(value)-behavior['value'])>2e-6 or
                        not torch.allclose(dist.probs[0],torch.tensor(behavior['probabilities']),atol=2e-6,rtol=0)):
                    raise ValueError('route behavior weights/probability/value differ')
                current.append(RolloutStep(encoded,behavior['action_row'],True,'route','MAP',
                    behavior['old_log_prob'],behavior['value'],0.,False,False))
            for left,right in zip(current,current[1:]): left.next_value=right.value
            if current:
                current[-1].reward=1. if episode.outcome=='ACT1_CLEAR' else -1.
                # End of the Act1 training objective, not an ordinary-run victory.
                current[-1].terminated=True
            steps.extend(current)
        if not steps: raise ValueError('no route decisions in this complete batch')
        try:
            result=ppo_update(self.model,self.optimizer,self.encoder,steps,self.config,generator=generator)
            if frozen_hash(self.model)!=self.frozen: raise ValueError('frozen route features changed')
        except BaseException:
            self.failed=True; raise
        self.generation+=1; self.snapshot=state_hash(self.model); self.model.eval()
        return dict(result,version=VERSION,generation=self.generation,episodes=len(episodes),
            objective='act1_clear',target='actual complete-episode +/-1, gamma=lambda=1',
            model_sha256=self.snapshot,encoder_unchanged=True)
