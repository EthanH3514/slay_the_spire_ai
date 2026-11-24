"""Opt-in preference P(choice | public state, local effect, local afterstate).

NOT an outcome critic, full room simulator, new production policy or a claim
that card BC transfers to upgrade/shop choices. State and card embeddings are
shared. Exact public-instance joins are references, never numeric indexes.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from . import learning_model as legacy
from .run_afterstates import LocalTransition, RunStateEncoder, _digest
from .run_state_model import SharedRunStateModel, StateBatch, tensorize_states


VERSION = "shared-run-action-effect-preference-v1"
OPERATIONS = ("take_card", "skip", "singing_bowl", "upgrade", "remove")
EFFECT_SCALES = dict(hp_delta=100, max_hp_delta=100, gold_delta=1000,
                     healing_requested=100, max_hp_loss_requested=100)


@dataclass(frozen=True, slots=True)
class EncodedLocalChoice:
    schema_id: str
    before: object
    after: object
    operation: int
    effects: tuple
    known: tuple
    before_card_ref: int
    after_card_ref: int


class RunChoiceEncoder:
    def __init__(self, state_encoder=None):
        self.states = state_encoder or RunStateEncoder()
        self.base, self.catalog = self.states.base, self.states.catalog
        self.schema_id = _digest(dict(version=VERSION, states=self.states.schema_id,
            operations=OPERATIONS, effect_scales=EFFECT_SCALES,
            card_refs="one-based public deck instance; zero means no changed card"))

    def encode(self, transition):
        if not isinstance(transition, LocalTransition) or transition.operation not in OPERATIONS:
            raise TypeError("resolved versioned local transition required")
        if transition.before.phase != "decision" or transition.after.phase != "local_effect_resolved":
            raise ValueError("local transition phase mismatch")
        if set(transition.effects) != set(EFFECT_SCALES):
            raise ValueError("local effect fields mismatch")
        values, known = [], []
        for name, scale in EFFECT_SCALES.items():
            value = transition.effects[name]
            if value is not None and (type(value) is not int or abs(value)>1000000):
                raise ValueError("bounded integer effect or unknown required")
            values.append(0. if value is None else value/scale)
            known.append(value is not None)
        before, after = self.states.encode(transition.before), self.states.encode(transition.after)
        target, left, right = transition.target_index, 0, 0
        if transition.operation in {"upgrade", "remove"}:
            size = len(transition.before.public.observation["inventory"]["deck"])
            if type(target) is not int or not 0 <= target < size:
                raise ValueError("invalid public target instance")
            left = target+1
            right = left if transition.operation=="upgrade" else 0
        elif target is not None:
            raise ValueError("non-deck operation has target instance")
        if transition.operation=="take_card":
            right = len(transition.after.public.observation["inventory"]["deck"])
        # PublicEncoder emits deck rows first and unordered. Check this join,
        # not a card-ID-only match that could confuse upgrades/misc/bottles.
        for state, ref in ((before,left),(after,right)):
            if ref:
                table=state.entities.tables["cards"]
                i=table.columns.index("pile_id")
                if ref>len(table.values) or table.values[ref-1][i] != self.base.vocabularies["pile"].index("deck")+2:
                    raise ValueError("public deck join mismatch")
        return EncodedLocalChoice(self.schema_id,before,after,OPERATIONS.index(transition.operation),
                                  tuple(values),tuple(known),left,right)


@dataclass
class ChoiceBatch:
    schema_id: str
    before: StateBatch
    after: StateBatch
    effects: torch.Tensor
    before_card_ref: torch.Tensor
    after_card_ref: torch.Tensor


def tensorize_choices(choices, encoder):
    if not choices or any(not isinstance(c,EncodedLocalChoice) or c.schema_id!=encoder.schema_id for c in choices):
        raise ValueError("versioned local choice batch required")
    features=[]
    for c in choices:
        if (type(c.operation) is not int or not 0<=c.operation<len(OPERATIONS) or
                len(c.effects)!=len(EFFECT_SCALES) or len(c.known)!=len(EFFECT_SCALES) or
                any(type(k) is not bool for k in c.known) or
                any(type(v) not in (float,int) or not math.isfinite(v) for v in c.effects) or
                any(v!=0 for v,k in zip(c.effects,c.known,strict=True) if not k)):
            raise ValueError("invalid local choice effects")
        if any(type(r) is not int or r<0 for r in (c.before_card_ref,c.after_card_ref)):
            raise ValueError("invalid public choice reference")
        features.append([float(i==c.operation) for i in range(len(OPERATIONS))]+
                        list(c.effects)+list(map(float,c.known)))
    return ChoiceBatch(encoder.schema_id,
        tensorize_states([c.before for c in choices],encoder.states),
        tensorize_states([c.after for c in choices],encoder.states),
        torch.tensor(features,dtype=torch.float32),
        torch.tensor([c.before_card_ref for c in choices],dtype=torch.long),
        torch.tensor([c.after_card_ref for c in choices],dtype=torch.long))


class SharedRunChoiceModel(nn.Module):
    def __init__(self, encoder, *, width=32, seed=731, warm_start=None):
        super().__init__()
        self.schema_id=encoder.schema_id
        self.states=SharedRunStateModel(encoder.states,width=width,seed=seed,warm_start=warm_start)
        self.states.preference.requires_grad_(False)  # v1 state-only head unused here
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2)
            self.preference=nn.Sequential(nn.Linear(4*width+len(OPERATIONS)+2*len(EFFECT_SCALES)+2,width),
                                          nn.Tanh(),nn.Linear(width,1))

    def forward(self,batch):
        if not isinstance(batch,ChoiceBatch) or batch.schema_id!=self.schema_id:
            raise ValueError("versioned local choice batch required")
        left=self.states.encode_context(batch.before)
        right=self.states.encode_context(batch.after)
        cards=[]
        for b,ref in ((batch.before,batch.before_card_ref),(batch.after,batch.after_card_ref)):
            table=b.entities.tables["cards"]
            rows=self.states.backbone.rows["cards"](table,self.states.backbone.embeddings)
            card=legacy._gather(rows,ref.unsqueeze(1)).squeeze(1)
            cards.extend((card,(ref>0).float().unsqueeze(1)))
        preference=self.preference(torch.cat((left,right,*cards,batch.effects),-1)).squeeze(-1)
        if not torch.isfinite(preference).all():
            raise ValueError("non-finite local choice preference")
        return preference

    def preference_loss(self,groups,targets,encoder):
        if not groups or len(groups)!=len(targets) or any(not g for g in groups):
            raise ValueError("nonempty aligned candidate groups required")
        if any(type(t) is not int or not 0<=t<len(g) for g,t in zip(groups,targets,strict=True)):
            raise ValueError("invalid candidate target")
        scores=self(tensorize_choices([c for g in groups for c in g],encoder))
        losses,offset=[],0
        for g,t in zip(groups,targets,strict=True):
            losses.append(-scores[offset:offset+len(g)].log_softmax(0)[t])
            offset+=len(g)
        return torch.stack(losses).mean()
