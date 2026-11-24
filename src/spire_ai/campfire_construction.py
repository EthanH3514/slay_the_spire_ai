"""Add complete campfire candidates to the ONE shared construction head.

Independent schema/migration leaves running v1 fits/checkpoints untouched.
The old feature extractor is reused; there is no private campfire scoring head.
New operation/context columns are zero initialized on migration. Complete room
commitments still require new labels; conditional human targets do not supply
rest-versus-smith supervision.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .construction_model import (ConstructionEncoder, SharedConstructionModel,
    tensorize_construction, OPERATIONS, EFFECT_SCALES)
from .campfire_transactions import KINDS, PENDING, transact_choices
from .run_afterstates import _digest

VERSION = 'shared-construction-with-campfire-v1'
CONTEXT = (*KINDS, 'card_reward_count', 'relic_reward_count', 'closed', 'is_campfire')


@dataclass(frozen=True, slots=True)
class EncodedChoice:
    schema_id: str
    core: object
    activity: str | None
    context: tuple

    @property
    def before(self):return self.core.before

    @property
    def after(self):return self.core.after

    @property
    def operation(self):return self.core.operation


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    schema_id: str
    candidates: tuple
    choices: tuple  # Decoder only; original activity plus exact target instance.


class CampfireConstructionEncoder:
    def __init__(self,common=None):
        self.common=common or ConstructionEncoder()
        self.states,self.base,self.catalog=self.common.states,self.common.base,self.common.catalog
        self.schema_id=_digest(dict(version=VERSION,common=self.common.schema_id,context=CONTEXT,
            missing_legacy_operation='zero old operation one-hot for rest/lift/dig'))

    def _wrap(self,core):
        return EncodedChoice(self.schema_id,core,None,(0.,)*len(CONTEXT))

    def encode(self,transition):
        return self._wrap(self.common.encode(transition))

    def encode_shop(self,state):
        d=self.common.encode_shop(state)
        return EncodedDecision(self.schema_id,tuple(self._wrap(c) for c in d.candidates),d.choices)

    def encode_campfire(self,state):
        before=self.states.encode(state.run);result=[];transitions=transact_choices(state)
        for t in transitions:
            kind=t.choice.kind
            # The v1 tensorizer has a bounded operation vocabulary. Skip is
            # ONLY a transport placeholder for new operations; tensor features
            # explicitly erase it below before the neural head sees the data.
            operation={'smith':'upgrade','remove':'remove'}.get(kind,'skip')
            effects=tuple(0. if t.effects[k] is None else t.effects[k]/scale for k,scale in EFFECT_SCALES.items())
            known=tuple(t.effects[k] is not None for k in EFFECT_SCALES)
            left=t.choice.target_index+1 if t.choice.target_index is not None else 0
            right=left if kind=='smith' else 0
            core=self.common._choice(before,self.states.encode(t.after.run),OPERATIONS.index(operation),
                                     effects,known,left,right)
            pending={p['kind']:p['count'] for p in t.after.pending}
            context=tuple(float(kind==k) for k in KINDS)+tuple(pending.get(k,0)/10. for k in PENDING)+(float(t.after.closed),1.)
            result.append(EncodedChoice(self.schema_id,core,kind,context))
        return EncodedDecision(self.schema_id,tuple(result),tuple(t.choice for t in transitions))


@dataclass
class ChoiceBatch:
    schema_id: str
    core: object
    context: torch.Tensor
    new_operation: torch.Tensor


def tensorize(choices,encoder):
    if not choices or any(not isinstance(c,EncodedChoice) or c.schema_id!=encoder.schema_id for c in choices):
        raise ValueError('versioned complete-construction candidates required')
    for c in choices:
        if c.activity is None:
            if c.context!=(0.,)*len(CONTEXT):raise ValueError('non-campfire context must be absent')
        elif (c.activity not in KINDS or len(c.context)!=len(CONTEXT) or c.context[-2:]!=(1.,1.) or
                c.context[:len(KINDS)]!=tuple(float(c.activity==k) for k in KINDS) or
                c.context[len(KINDS):len(KINDS)+2] not in ((0.,0.),(.1,0.),(0.,.1))):
            raise ValueError('invalid campfire operation/context')
    return ChoiceBatch(encoder.schema_id,tensorize_construction([c.core for c in choices],encoder.common),
        torch.tensor([c.context for c in choices],dtype=torch.float32),
        torch.tensor([c.activity in ('rest','lift','dig') for c in choices],dtype=torch.bool))


class CampfireConstructionModel(nn.Module):
    def __init__(self,encoder,*,width=32,seed=261101,warm_common=None):
        super().__init__();self.schema_id=encoder.schema_id
        self.common=SharedConstructionModel(encoder.common,width=width,seed=seed)
        if warm_common is not None:
            if warm_common.schema_id!=encoder.common.schema_id or warm_common.width!=width:
                raise ValueError('incompatible construction v1 migration')
            self.common.load_state_dict(warm_common.state_dict(),strict=True)
        old=self.common.preference
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.preference=nn.Sequential(nn.Linear(old[0].in_features+len(CONTEXT),width),nn.Tanh(),nn.Linear(width,1))
        with torch.no_grad():
            self.preference[0].weight.zero_()
            self.preference[0].weight[:,:old[0].in_features].copy_(old[0].weight)
            self.preference[0].bias.copy_(old[0].bias)
            self.preference[2].load_state_dict(old[2].state_dict())
        self.common.preference=nn.Identity()  # Exactly ONE trainable preference head.

    def features(self,batch):
        if not isinstance(batch,ChoiceBatch) or batch.schema_id!=self.schema_id:
            raise ValueError('versioned complete-construction batch required')
        features=self.common.features(batch.core)
        if batch.new_operation.any():
            features=features.clone();start=4*self.common.width+2
            features[batch.new_operation,start:start+len(OPERATIONS)]=0
        return torch.cat((features,batch.context),-1)

    def forward(self,batch):
        scores=self.preference(self.features(batch)).squeeze(-1)
        if not torch.isfinite(scores).all():raise ValueError('non-finite complete-construction preference')
        return scores
