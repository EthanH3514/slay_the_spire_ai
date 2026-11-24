"""Frozen card-only inference for the paired actor/effect preference models.

Preserve the historical missing-field projection and the actual legal-action
decoder. Labels, run IDs, RNG state and simulated future results are not inputs.
This adapter does not make the card preference head an outcome critic.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from .actions import Action
from .boss_cards import project_reward
from .metrics_stream import file_sha
from .policy_api import PolicyInput
from .run_afterstates import capture,card_reward_candidates
from .run_choice_model import RunChoiceEncoder,SharedRunChoiceModel,tensorize_choices
from .run_learning_model import make_model,tensorize
from . import run_choice_cache as numeric

VERSION='frozen-paired-card-preference-policy-v1'
TRAINING_VERSION='full-run-choice-paired-validation-v1'
ARMS=('candidate_actor','effect_preference')


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str((tensor.shape,tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ChoiceProjection:
    reward:object
    choices:tuple
    operations:tuple


def project_choices(public,encoder):
    projected=project_reward(public,encoder.base)
    wire=projected.row['public']
    canonical=PolicyInput(wire['observation'],tuple(Action.from_dict(a) for a in wire['legal_actions']))
    candidates=card_reward_candidates(capture(canonical,relic_order_known=False),encoder.catalog)
    if tuple(c.action for c in candidates)!=projected.encoded.legal_actions:
        raise ValueError('live preference candidate/action alignment mismatch')
    choices=tuple(encoder.encode(c.transition) for c in candidates)
    root=choices[0].before.entities
    if (root.tables!=projected.encoded.tables or root.extra_tables!=projected.encoded.extra_tables or
            root.legal_mask!=projected.encoded.legal_mask or root.scope_mask!=projected.encoded.scope_mask):
        raise ValueError('live preference root/actor features differ')
    return ChoiceProjection(projected,choices,tuple(c.operation for c in candidates))


class FrozenRunChoicePolicy:
    def __init__(self,checkpoint,expected_sha,arm,encoder=None,*,width=32,expected_epoch=None,expected_data_receipt=None,
                 expected_contract=None):
        if arm not in ARMS:raise ValueError('unknown frozen card arm')
        self.encoder=encoder or RunChoiceEncoder();path=Path(checkpoint)
        if file_sha(path)!=expected_sha:raise ValueError('frozen card checkpoint SHA mismatch')
        saved=torch.load(path,weights_only=True,map_location='cpu');contract=saved.get('contract',{})
        if (saved.get('version')!=TRAINING_VERSION or contract.get('version')!=TRAINING_VERSION or
                saved.get('artifact_type')!='offline_preference_training_only' or
                saved.get('critic_fitted') is not False or saved.get('production_ready') is not False or
                contract.get('schema')!=self.encoder.schema_id or
                (expected_contract is None and any(k in contract for k in ('continuation','input_migration','model_compute')))):
            raise ValueError('fixed original paired card checkpoint required')
        if expected_contract is not None and contract != expected_contract:
            raise ValueError('frozen selected training contract mismatch')
        self.epoch=saved['state']['history'][-1]['epoch'];self.data_receipt=contract['data_receipt']
        if (type(self.epoch) is not int or self.epoch<1 or
                (expected_epoch is not None and self.epoch!=expected_epoch) or
                (expected_data_receipt is not None and self.data_receipt!=expected_data_receipt)):
            raise ValueError('frozen card selected epoch/data mismatch')
        state=saved['state'];entry=state['history'][-1]
        if (entry.get('checkpoint')!=path.name or state['total_updates']!=entry['total_updates'] or
                not ((state['phase']=='train' and state['next_batch']==0 and state['epoch']==self.epoch+1) or
                     (state['phase']=='complete' and state['epoch']==self.epoch))):
            raise ValueError('validated epoch boundary checkpoint required')
        self.compute_mode=contract.get('model_compute',{}).get('mode','reference')
        if self.compute_mode not in ('reference','roots_cards','roots_rows'):
            raise ValueError('unsupported frozen computation mode')
        self.arm=arm;self.checkpoint_sha=expected_sha
        self.model=make_model(self.encoder.base,width=width) if arm=='candidate_actor' else SharedRunChoiceModel(self.encoder,width=width)
        self.model.load_state_dict(saved['models'][arm],strict=True)
        if self.compute_mode=='roots_rows':
            from .run_choice_rows import StaticRowEncoder
            backbone=self.model if arm=='candidate_actor' else self.model.states.backbone
            for name,row in list(backbone.rows.items()):
                backbone.rows[name]=StaticRowEncoder(row,empty=True,unique=(name=='cards'))
        self.model.eval();self.model.requires_grad_(False)
        if state_hash(self.model)!=saved['state_hashes'][arm]:raise ValueError('frozen card model state mismatch')
        self._probe_input=saved['probe_input'];self._probe_logits=saved['probe_logits'][arm]
        self.last_decision=None

    def verify_probe(self,record):
        if record['metadata']!=self._probe_input:raise ValueError('frozen card public probe input mismatch')
        with torch.inference_mode():actual=self.batch_scores(numeric.collate([record],self.encoder))
        if any(not torch.equal(a,b) for a,b in zip(actual,self._probe_logits,strict=True)):
            raise ValueError('frozen card exact public probe mismatch')

    def batch_scores(self,batch):
        if self.arm=='effect_preference' and self.compute_mode!='reference':
            from .run_choice_compute import choice_scores
            return list(choice_scores(self.model,batch.choice,mode='roots_cards',sizes=batch.sizes).split(batch.sizes))
        return numeric.model_scores(self.model,batch,self.arm)

    def scores(self,projected):
        with torch.inference_mode():
            if self.arm=='candidate_actor':
                values,_=self.model(tensorize([projected.reward.encoded],self.encoder.base))
                result=values[0,:len(projected.choices)]
            elif self.compute_mode!='reference':
                from .run_choice_compute import choice_scores
                result=choice_scores(self.model,tensorize_choices(projected.choices,self.encoder),
                    mode='roots_cards',sizes=(len(projected.choices),))
            else:result=self.model(tensorize_choices(projected.choices,self.encoder))
        if result.shape!=(len(projected.reward.actions),) or not torch.isfinite(result).all():
            raise ValueError('invalid frozen card scores')
        return result

    def choose(self,public):
        projected=project_choices(public,self.encoder);scores=self.scores(projected)
        probabilities=scores.softmax(0);index=int(scores.argmax());reward=projected.reward
        action=reward.decode(index)
        if action not in public.legal_actions:raise ValueError('frozen card action no longer legal')
        self.last_decision=dict(version=VERSION,mode='greedy_fixed_partial_card_preference',arm=self.arm,
            checkpoint_epoch=self.epoch,checkpoint_sha256=self.checkpoint_sha,input_sha256=reward.row['input_sha256'],
            action=action.as_dict(),meaning=reward.meanings[index],probabilities=probabilities.tolist(),
            candidates=reward.meanings,operations=projected.operations,fallback=False)
        return action
