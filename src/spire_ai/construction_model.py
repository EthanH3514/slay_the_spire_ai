"""One opt-in preference head for deck construction, including startup pressure.

Reward add/skip, conditional removal/upgrade, and observed shop transactions use
the SAME head. A bought card is an add with price/remaining opportunities, not a
separately calibrated card-buying score. This is still a preference model, NOT
an outcome value or an estimate of gold's HP value. Conditional demonstrations
do not label whether to buy a removal or to rest instead of upgrading.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from . import learning_model as legacy
from .deck_startup import SCHEMA as STARTUP_SCHEMA, COLUMNS, describe_startup, transition_features
from .run_afterstates import _digest
from .run_choice_model import RunChoiceEncoder, OPERATIONS as LOCAL_OPERATIONS, EFFECT_SCALES
from .run_state_model import SharedRunStateModel, tensorize_states
from .shop_choice_model import (ShopChoiceEncoder, tensorize_shop, FEATURES, PENDING, _pool)
from .shop_transactions import BUY_KINDS
from .shop_policy_scope import LEARNING_KINDS


VERSION = 'shared-deck-construction-preference-v1'
OPERATIONS = (*LOCAL_OPERATIONS, 'buy_relic')
SHOP_OPERATION = dict(leave='skip', buy_card='take_card', buy_remove='remove', buy_relic='buy_relic')


@dataclass(frozen=True, slots=True)
class ConstructionChoice:
    schema_id: str
    before: object
    after: object
    operation: int
    effects: tuple
    known: tuple
    before_card_ref: int
    after_card_ref: int
    startup_before: object
    startup_after: object
    shop: object = None


@dataclass(frozen=True, slots=True)
class ConstructionShopDecision:
    schema_id: str
    candidates: tuple
    choices: tuple  # Execution metadata, never neural features.
    complete_shelf: bool


class ConstructionEncoder:
    def __init__(self, states=None):
        self.local = RunChoiceEncoder(states)
        self.states, self.base, self.catalog = self.local.states, self.local.base, self.local.catalog
        self.shop = ShopChoiceEncoder(self.states)
        self.schema_id = _digest(dict(version=VERSION, local=self.local.schema_id,
            shop=self.shop.schema_id, startup=STARTUP_SCHEMA, operations=OPERATIONS,
            learning_scope=LEARNING_KINDS, shared_head=True))

    def _choice(self, before, after, operation, effects, known, left, right, shop=None):
        return ConstructionChoice(self.schema_id, before, after, operation, effects, known,
            left, right, describe_startup(before, self.states), describe_startup(after, self.states), shop)

    def encode(self, transition):
        c = self.local.encode(transition)
        return self._choice(c.before, c.after, c.operation, c.effects, c.known,
                            c.before_card_ref, c.after_card_ref)

    def encode_shop(self, state):
        d = self.shop.encode_decision(state)
        choices, candidates = [], []
        for choice, c in zip(d.choices, d.candidates, strict=True):
            if choice.kind not in LEARNING_KINDS:
                continue
            values = dict(zip(FEATURES, c.features, strict=True))
            # Shop net deltas are resolved. Requested heal/max-HP loss are not
            # exported by this adapter: keep unknown, never invent zero.
            effects = tuple(values.get(k, 0.) for k in EFFECT_SCALES)
            known = tuple(k in values for k in EFFECT_SCALES)
            candidates.append(self._choice(c.before, c.after,
                OPERATIONS.index(SHOP_OPERATION[choice.kind]), effects, known,
                c.before_card_ref, c.after_card_ref, c))
            choices.append(choice)
        return ConstructionShopDecision(self.schema_id, tuple(candidates), tuple(choices), d.complete_shelf)


@dataclass
class ConstructionBatch:
    schema_id: str
    before: object
    after: object
    effects: torch.Tensor
    before_card_ref: torch.Tensor
    after_card_ref: torch.Tensor
    startup: torch.Tensor
    shop: object
    shop_indices: torch.Tensor


def tensorize_construction(choices, encoder):
    if (not choices or len(choices) > 4096 or
            any(not isinstance(c, ConstructionChoice) or c.schema_id != encoder.schema_id for c in choices)):
        raise ValueError('bounded versioned construction candidates required')
    effects, startup, shop, indices = [], [], [], []
    for i,c in enumerate(choices):
        if (type(c.operation) is not int or not 0 <= c.operation < len(OPERATIONS)
                or len(c.effects) != len(EFFECT_SCALES) or len(c.known) != len(EFFECT_SCALES)
                or any(type(k) is not bool for k in c.known)
                or any(type(v) not in (int,float) or not math.isfinite(v) or (not k and v != 0)
                       for v,k in zip(c.effects,c.known,strict=True))):
            raise ValueError('invalid construction operation/effect')
        for state, ref in ((c.before,c.before_card_ref), (c.after,c.after_card_ref)):
            table = state.entities.tables['cards']
            pile = table.columns.index('pile_id')
            if (type(ref) is not int or not 0 <= ref <= len(table.values) or
                    (ref and table.values[ref-1][pile] != encoder.base.vocabularies['pile'].index('deck')+2)):
                raise ValueError('invalid construction deck reference')
        effects.append([float(j == c.operation) for j in range(len(OPERATIONS))]
                       + list(c.effects) + list(map(float,c.known)))
        startup.append(transition_features(c.startup_before, c.startup_after))
        if c.shop is not None:
            if (c.shop.before != c.before or c.shop.after != c.after or
                    (c.shop.before_card_ref,c.shop.after_card_ref) != (c.before_card_ref,c.after_card_ref)):
                raise ValueError('shop/construction context mismatch')
            shop.append(c.shop); indices.append(i)
    return ConstructionBatch(encoder.schema_id,
        tensorize_states([c.before for c in choices], encoder.states),
        tensorize_states([c.after for c in choices], encoder.states),
        torch.tensor(effects, dtype=torch.float32),
        torch.tensor([c.before_card_ref for c in choices], dtype=torch.long),
        torch.tensor([c.after_card_ref for c in choices], dtype=torch.long),
        torch.tensor(startup, dtype=torch.float32),
        tensorize_shop(shop, encoder.shop) if shop else None,
        torch.tensor(indices, dtype=torch.long))


class SharedConstructionModel(nn.Module):
    def __init__(self, encoder, *, width=32, seed=261101, warm_start=None, freeze_shared=True):
        super().__init__()
        self.schema_id, self.width = encoder.schema_id, width
        self.states = SharedRunStateModel(encoder.states, width=width, seed=seed)
        self.states.preference.requires_grad_(False)
        if warm_start is not None:
            if warm_start.schema_id != encoder.local.schema_id or warm_start.states.backbone.width != width:
                raise ValueError('incompatible construction initializer')
            self.states.load_state_dict(warm_start.states.state_dict(), strict=True)
        if freeze_shared:
            self.states.requires_grad_(False)
        common = 4*width+2+len(OPERATIONS)+2*len(EFFECT_SCALES)
        self.startup_offset = common+4*width+3+len(FEATURES)+1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.good = nn.Sequential(nn.Linear(3*12+len(BUY_KINDS)+3,width), nn.Tanh())
            self.pending = nn.Sequential(nn.Linear(width+1+len(PENDING)+2+len(BUY_KINDS),width), nn.Tanh())
            self.preference = nn.Sequential(nn.Linear(self.startup_offset+6*len(COLUMNS),width),
                                             nn.Tanh(),nn.Linear(width,1))
        if warm_start is not None:
            # Exact local-head warm start. All new feature weights start at 0;
            # there is no arbitrary size penalty or buying bonus at epoch zero.
            with torch.no_grad():
                layer, old = self.preference[0], warm_start.preference[0]
                layer.weight.zero_(); layer.bias.copy_(old.bias)
                prefix = 4*width+2
                layer.weight[:,:prefix+len(LOCAL_OPERATIONS)].copy_(old.weight[:,:prefix+len(LOCAL_OPERATIONS)])
                layer.weight[:,prefix+len(OPERATIONS):common].copy_(old.weight[:,prefix+len(LOCAL_OPERATIONS):])
                self.preference[2].load_state_dict(warm_start.preference[2].state_dict())

    def _goods(self, table):
        values = table['values']
        pieces = [self.states.backbone.embeddings[name](values[...,i+1].long())
                  for i,name in enumerate(('card','relic','potion'))]
        pieces += [torch.stack([(values[...,0] == i).float() for i in range(len(BUY_KINDS))],-1), values[...,4:]]
        return self.good(torch.cat(pieces,-1))*table['present'].unsqueeze(-1)

    def features(self, batch):
        if not isinstance(batch, ConstructionBatch) or batch.schema_id != self.schema_id:
            raise ValueError('versioned construction batch required')
        left, right = self.states.encode_context(batch.before), self.states.encode_context(batch.after)
        rows = [self.states.backbone.rows['cards'](b.entities.tables['cards'],self.states.backbone.embeddings)
                for b in (batch.before,batch.after)]
        changed = []
        for r,ref in zip(rows,(batch.before_card_ref,batch.after_card_ref),strict=True):
            changed += [legacy._gather(r,ref.unsqueeze(1)).squeeze(1),(ref>0).float().unsqueeze(1)]
        # All absent context is 0 with an explicit shop-present bit. No fake
        # shelf, price, or available alternative is inserted in human histories.
        opportunities = left.new_zeros((left.shape[0],4*self.width+3+len(FEATURES)+1))
        if batch.shop is not None:
            s = batch.shop
            goods = [self._goods(t) for t in (s.before_goods,s.after_goods)]
            pools = [_pool(g,t['present']) for g,t in zip(goods,(s.before_goods,s.after_goods),strict=True)]
            selected = legacy._gather(goods[0],s.selected_good_ref.unsqueeze(1)).squeeze(1)
            n,e,k = s.pending_refs.shape
            joined = legacy._gather(rows[1][batch.shop_indices],s.pending_refs.reshape(n,e*k)).reshape(n,e,k,-1)
            counts = (s.pending_refs>0).sum(2,keepdim=True).float()
            pending = self.pending(torch.cat((joined.sum(2)/counts.clamp_min(1),counts.log1p(),s.pending_features),-1))
            pending = pending*s.pending_present.unsqueeze(-1)
            pp,pc = _pool(pending,s.pending_present)
            # Drop the old shop-kind one-hot: common operations above unify
            # free/paid card addition and free/paid removal in the SAME head.
            context = torch.cat((pools[0][0],pools[1][0],selected,pp,
                pools[0][1],pools[1][1],pc,s.features[:,-len(FEATURES):],left.new_ones((n,1))),-1)
            opportunities = opportunities.index_copy(0,batch.shop_indices,context)
        return torch.cat((left,right,*changed,batch.effects,opportunities,batch.startup),-1)

    def forward(self,batch):
        result = self.preference(self.features(batch)).squeeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError('non-finite construction preference')
        return result

    def preference_loss(self,groups,targets,encoder):
        if not groups or len(groups) != len(targets) or any(not g for g in groups):
            raise ValueError('aligned observed candidate groups required')
        if any(type(t) is not int or not 0 <= t < len(g) for g,t in zip(groups,targets,strict=True)):
            raise ValueError('invalid construction target')
        for group in groups:
            if any(c.before != group[0].before for c in group):
                raise ValueError('one public root per observed choice menu required')
            if any(c.shop is not None for c in group):
                if (any(c.shop is None or not c.shop.features[FEATURES.index('before_shelf_complete')]
                        for c in group) or sum(c.operation == OPERATIONS.index('skip') for c in group) != 1):
                    raise ValueError('observed complete shop menu including leave required')
        scores = self(tensorize_construction([c for g in groups for c in g],encoder))
        losses,offset = [],0
        for group,target in zip(groups,targets,strict=True):
            losses.append(-scores[offset:offset+len(group)].log_softmax(0)[target]); offset += len(group)
        return torch.stack(losses).mean()
