"""Opt-in shop preference head with shared public run/item representations.

Scores are action preferences, NOT win probability, gold value or a settled
afterstate critic. Pending acquisition windows are represented explicitly and
remain candidates. No production policy imports this module. No metrics row
without an observed menu can become a listwise purchasing label here.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn

from . import learning_model as legacy
from .policy_api import PolicyInput
from .public_encoding import _key
from .run_afterstates import RunStateEncoder, capture, _digest
from .run_state_model import SharedRunStateModel, tensorize_states
from .shop_transactions import BUY_KINDS, ShopState, transact_choices


VERSION = 'shared-shop-action-preference-v1'
KINDS = ('leave', *BUY_KINDS, 'buy_remove')
PENDING = ('bottle_attack', 'bottle_skill', 'bottle_power', 'duplicate_card',
    'upgrade_skills', 'upgrade_attacks', 'card_rewards', 'potion_rewards',
    'restock', 'full_shelf_reveal', 'removal_price_reveal', 'nonstandard_relic_acquisition')
FEATURES = ('paid', 'paid_fraction', 'hp_delta', 'max_hp_delta', 'gold_delta',
    'before_removal_price', 'before_removal_price_known', 'before_removal_available',
    'before_shelf_complete', 'before_removal_services_used',
    'after_removal_price', 'after_removal_price_known', 'after_removal_available',
    'after_shelf_complete', 'after_removal_services_used', 'closed', 'pending',
    'before_card_present', 'after_card_present')
MAX_CANDIDATES = 1024


@dataclass(frozen=True, slots=True)
class EncodedShopChoice:
    schema_id: str
    before: object
    after: object
    kind: int
    features: tuple
    before_goods: tuple
    after_goods: tuple
    selected_good_ref: int
    before_card_ref: int
    after_card_ref: int
    pending: tuple


@dataclass(frozen=True, slots=True)
class EncodedShopDecision:
    schema_id: str
    candidates: tuple
    # Decoder metadata never enters tensorization or a model.
    choices: tuple
    complete_shelf: bool


def _menu_free(run):
    # The full shelf is encoded separately. Do not double-count affordable
    # goods in the root's old menu or encode old row/action indices.
    obs = run.public.as_dict()['observation']
    obs['screen'] = {}
    return replace(capture(PolicyInput(obs, ()), relic_order_known=run.relic_order_known,
                           run_public=run.run_public), phase=run.phase)


def _shop_features(state):
    price = state.removal_price
    return (0. if price is None else price / 1000.,
            float(price is not None or state.complete_shelf), float(price is not None),
            float(state.complete_shelf), state.removal_services_used / 10.)


class ShopChoiceEncoder:
    def __init__(self, states=None):
        self.states = states or RunStateEncoder()
        self.base = self.states.base
        self.schema_id = _digest(dict(version=VERSION, states=self.states.schema_id,
            kinds=KINDS, pending=PENDING, features=FEATURES,
            goods='kind/card/relic/potion/upgrade/price/affordability; unordered; no slot token',
            joins='exact deck instance and selected shelf item; zero means absent',
            output='action preference; pending is not a terminal value'))

    def token(self, category, identifier):
        names = self.base.vocabularies[category]
        value = _key(identifier)
        if value not in names:
            raise ValueError('shop item outside fixed vocabulary: ' + str(identifier))
        return names.index(value) + 2

    def goods(self, state):
        if len(state.offers) > 13:
            raise ValueError('public shop shelf exceeds slot bound')
        result = []
        money = state.run.public.observation['player']['gold']
        for offer in state.offers:
            kind = offer['kind']
            card = offer.get('card')
            result.append((BUY_KINDS.index(kind),
                self.token('card', card['id']) if card else 0,
                self.token('relic', offer['item_id']) if kind == 'buy_relic' else 0,
                self.token('potion', offer['item_id']) if kind == 'buy_potion' else 0,
                card['upgrades'] / 10. if card else 0., offer['price'] / 1000.,
                float(offer['price'] <= money)))
        return tuple(result)

    def _encode(self, transition, before, before_goods):
        t = transition
        after = self.states.encode(_menu_free(t.after.run))
        b_ref = t.choice.target_index + 1 if t.choice.kind == 'buy_remove' else 0
        a_ref = len(t.after.run.public.observation['inventory']['deck']) if t.choice.kind == 'buy_card' else 0
        selected = next((i+1 for i,o in enumerate(t.before.offers) if o['key'] == t.choice.key), 0)
        pending = []
        for event in t.after.pending:
            kind = event['kind']
            if kind not in PENDING:
                raise ValueError('unversioned pending shop effect: ' + kind)
            refs = tuple(i+1 for i in event.get('candidates', ()))
            pending.append((PENDING.index(kind), event.get('count', 0) / 10.,
                float('count' in event),
                BUY_KINDS.index(event['category']) + 1 if 'category' in event else 0, refs))
        money = t.before.run.public.observation['player']['gold']
        features = (t.paid/1000., t.paid/max(1, money), t.effects['hp_delta']/100.,
            t.effects['max_hp_delta']/100., t.effects['gold_delta']/1000.,
            *_shop_features(t.before), *_shop_features(t.after), float(t.after.closed),
            float(bool(pending)), float(b_ref > 0), float(a_ref > 0))
        return EncodedShopChoice(self.schema_id, before, after, KINDS.index(t.choice.kind),
            features, before_goods, self.goods(t.after), selected, b_ref, a_ref, tuple(pending))

    def encode_decision(self, state):
        if not isinstance(state, ShopState):
            raise TypeError('public versioned shop state required')
        transitions = transact_choices(state)
        if not 1 <= len(transitions) <= MAX_CANDIDATES:
            raise ValueError('shop candidate bound; never truncate a menu')
        legal = tuple(t.choice for t in transitions)
        before = self.states.encode(_menu_free(state.run))
        goods = self.goods(state)
        encoded = tuple(self._encode(t, before, goods) for t in transitions)
        return EncodedShopDecision(self.schema_id, encoded, legal, state.complete_shelf)


@dataclass
class ShopBatch:
    schema_id: str
    before: object
    after: object
    features: torch.Tensor
    before_goods: dict
    after_goods: dict
    selected_good_ref: torch.Tensor
    before_card_ref: torch.Tensor
    after_card_ref: torch.Tensor
    pending_features: torch.Tensor
    pending_refs: torch.Tensor
    pending_present: torch.Tensor


def _goods_batch(groups, encoder):
    width = max(1, max(map(len, groups)))
    values, present = [], []
    for group in groups:
        if len(group) > 13:
            raise ValueError('bounded shelf required')
        for row in group:
            if (len(row) != 7 or type(row[0]) is not int or not 0 <= row[0] < len(BUY_KINDS) or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in row) or
                    any(v < 0 for v in row) or row[6] not in (0., 1.)):
                raise ValueError('invalid shelf numeric row')
            for j, category in enumerate(('card', 'relic', 'potion'), 1):
                if type(row[j]) is not int or not 0 <= row[j] < len(encoder.base.vocabularies[category])+2:
                    raise ValueError('invalid shelf identity')
                if (j-1 == row[0]) != (row[j] >= 2):
                    raise ValueError('shelf kind/item identity mismatch')
        values.append(tuple(group) + ((0.,)*7,)*(width-len(group)))
        present.append([True]*len(group) + [False]*(width-len(group)))
    return dict(values=torch.tensor(values, dtype=torch.float32),
                present=torch.tensor(present, dtype=torch.bool))


def tensorize_shop(choices_, encoder):
    if (not choices_ or len(choices_) > 4096 or
            any(not isinstance(c, EncodedShopChoice) or c.schema_id != encoder.schema_id for c in choices_)):
        raise ValueError('bounded versioned shop batch required')
    features, pending, references, masks = [], [], [], []
    event_width = max(1, max(len(c.pending) for c in choices_))
    ref_width = max(1, max((len(e[4]) for c in choices_ for e in c.pending), default=0))
    if event_width > 16 or len(choices_)*event_width*ref_width > 1000000:
        raise ValueError('pending shop batch exceeds memory budget')
    for c in choices_:
        if (type(c.kind) is not int or not 0 <= c.kind < len(KINDS) or
                len(c.features) != len(FEATURES) or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in c.features)):
            raise ValueError('invalid shop choice features')
        for ref, limit in ((c.selected_good_ref, len(c.before_goods)),
                           (c.before_card_ref, len(c.before.entities.tables['cards'].values)),
                           (c.after_card_ref, len(c.after.entities.tables['cards'].values))):
            if type(ref) is not int or not 0 <= ref <= limit:
                raise ValueError('invalid shop item reference')
        features.append(tuple(float(i==c.kind) for i in range(len(KINDS)))+c.features)
        event_rows, event_refs = [], []
        for kind, count, known, category, refs in c.pending:
            if (type(kind) is not int or not 0 <= kind < len(PENDING) or
                    type(category) is not int or not 0 <= category <= len(BUY_KINDS) or
                    known not in (0., 1.) or not math.isfinite(count) or count < 0 or
                    (not known and count != 0) or
                    any(type(r) is not int or not 1 <= r <= len(c.after.entities.tables['cards'].values) for r in refs)):
                raise ValueError('invalid pending shop effect')
            event_rows.append(tuple(float(i==kind) for i in range(len(PENDING))) +
                (count, known) + tuple(float(i+1==category) for i in range(len(BUY_KINDS))))
            event_refs.append(tuple(refs)+(0,)*(ref_width-len(refs)))
        padding = event_width-len(event_rows)
        pending.append(event_rows + [(0.,)*(len(PENDING)+2+len(BUY_KINDS))]*padding)
        references.append(event_refs + [(0,)*ref_width]*padding)
        masks.append([True]*len(c.pending)+[False]*padding)
    return ShopBatch(encoder.schema_id,
        tensorize_states([c.before for c in choices_], encoder.states),
        tensorize_states([c.after for c in choices_], encoder.states),
        torch.tensor(features, dtype=torch.float32),
        _goods_batch([c.before_goods for c in choices_], encoder),
        _goods_batch([c.after_goods for c in choices_], encoder),
        torch.tensor([c.selected_good_ref for c in choices_], dtype=torch.long),
        torch.tensor([c.before_card_ref for c in choices_], dtype=torch.long),
        torch.tensor([c.after_card_ref for c in choices_], dtype=torch.long),
        torch.tensor(pending, dtype=torch.float32), torch.tensor(references, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool))


def _pool(rows, present):
    count = present.sum(1, keepdim=True).float()
    return rows.sum(1)/count.clamp_min(1), count.log1p()


class SharedShopChoiceModel(nn.Module):
    def __init__(self, encoder, *, width=32, seed=991, warm_states=None, freeze_shared=True):
        super().__init__()
        self.schema_id = encoder.schema_id
        self.states = SharedRunStateModel(encoder.states, width=width, seed=seed)
        if warm_states is not None:
            if warm_states.schema_id != encoder.states.schema_id or warm_states.backbone.width != width:
                raise ValueError('incompatible shared state warm start')
            self.states.load_state_dict(warm_states.state_dict(), strict=True)
        self.states.preference.requires_grad_(False)
        if freeze_shared:
            self.states.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            # Item embeddings are the SAME learned card/relic/potion tables.
            self.good = nn.Sequential(nn.Linear(3*12+len(BUY_KINDS)+3, width), nn.Tanh())
            self.pending = nn.Sequential(nn.Linear(width+1+len(PENDING)+2+len(BUY_KINDS), width), nn.Tanh())
            self.preference = nn.Sequential(
                nn.Linear(8*width+3+len(KINDS)+len(FEATURES), width), nn.Tanh(), nn.Linear(width, 1))

    def _goods(self, table):
        values = table['values']
        embeddings = self.states.backbone.embeddings
        pieces = [embeddings[name](values[..., i+1].long()) for i,name in enumerate(('card','relic','potion'))]
        pieces += [torch.stack([(values[...,0]==i).float() for i in range(len(BUY_KINDS))], -1), values[...,4:]]
        return self.good(torch.cat(pieces, -1)) * table['present'].unsqueeze(-1)

    def forward(self, batch):
        if not isinstance(batch, ShopBatch) or batch.schema_id != self.schema_id:
            raise ValueError('versioned shop batch required')
        left, right = self.states.encode_context(batch.before), self.states.encode_context(batch.after)
        card_rows = [self.states.backbone.rows['cards'](b.entities.tables['cards'], self.states.backbone.embeddings)
                     for b in (batch.before, batch.after)]
        card_vectors = [legacy._gather(rows, refs.unsqueeze(1)).squeeze(1)
                        for rows,refs in zip(card_rows,(batch.before_card_ref,batch.after_card_ref),strict=True)]
        goods = [self._goods(t) for t in (batch.before_goods, batch.after_goods)]
        good_vectors = [_pool(r,t['present']) for r,t in zip(goods,(batch.before_goods,batch.after_goods),strict=True)]
        selected = legacy._gather(goods[0], batch.selected_good_ref.unsqueeze(1)).squeeze(1)
        n,e,k = batch.pending_refs.shape
        joined = legacy._gather(card_rows[1], batch.pending_refs.reshape(n,e*k)).reshape(n,e,k,-1)
        counts = (batch.pending_refs>0).sum(2,keepdim=True).float()
        joined = joined.sum(2)/counts.clamp_min(1)
        pending = self.pending(torch.cat((joined, counts.log1p(), batch.pending_features),-1))
        pending = pending * batch.pending_present.unsqueeze(-1)
        pooled_pending, pending_count = _pool(pending,batch.pending_present)
        result = self.preference(torch.cat((left,right,*card_vectors,
            good_vectors[0][0],good_vectors[1][0],selected,pooled_pending,
            good_vectors[0][1],good_vectors[1][1],pending_count,batch.features),-1)).squeeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError('non-finite shop preference')
        return result

    def preference_loss(self, decisions, targets, encoder):
        if not decisions or len(decisions) != len(targets):
            raise ValueError('aligned complete shop decisions required')
        for d,t in zip(decisions,targets,strict=True):
            if (not isinstance(d,EncodedShopDecision) or d.schema_id!=encoder.schema_id or
                    d.complete_shelf is not True or not d.candidates or len(d.candidates)!=len(d.choices) or
                    tuple(c.kind for c in d.candidates)!=tuple(KINDS.index(c.kind) for c in d.choices) or
                    sum(c.kind==0 for c in d.candidates)!=1):
                raise ValueError('full observed shop menu including leave required for listwise supervision')
            if type(t) is not int or not 0<=t<len(d.candidates):
                raise ValueError('invalid observed shop target')
        scores = self(tensorize_shop([c for d in decisions for c in d.candidates],encoder))
        offset, losses = 0, []
        for d,t in zip(decisions,targets,strict=True):
            segment = scores[offset:offset+len(d.candidates)]
            losses.append(-segment.log_softmax(0)[t]); offset+=len(d.candidates)
        return torch.stack(losses).mean()
