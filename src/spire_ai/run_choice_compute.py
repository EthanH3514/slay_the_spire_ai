"""Experimental, opt-in reuse inside one local-choice forward pass.

Uses the existing modules and parameters; no learned tensor survives a forward
pass or crosses optimizer steps. Sharing differentiable work changes reduction
order, so this is NOT a bitwise-equivalent training replacement. The original
model remains the production/reference implementation while it is pinned by a
running experiment.
"""
import torch

from . import learning_model as legacy
from .run_choice_model import ChoiceBatch
from .run_state_model import StateBatch


VERSION = 'local-choice-within-forward-reuse-probe-v1'
MODES = ('reference', 'cards', 'roots_cards')


def context_and_cards(states, batch):
    """Reference context operations, retaining its already-computed card rows."""
    if not isinstance(batch, StateBatch) or batch.schema_id != states.schema_id:
        raise ValueError('versioned state batch required')
    b, model = batch.entities, states.backbone
    entity = {name: model.rows[name](b.tables[name], model.embeddings) for name in legacy.ENTITY_TABLES}
    for _ in range(3):
        neighbors = legacy._gather(entity['map_nodes'], legacy._column(b, 'map_edges', 'destination_ref'))
        messages, counts = legacy._group(neighbors, legacy._column(b, 'map_edges', 'source_ref'),
            entity['map_nodes'].shape[1], b.tables['map_edges']['present'])
        entity['map_nodes'] = model.map_message(torch.cat((entity['map_nodes'], messages, counts.log1p()), -1))
        entity['map_nodes'] = entity['map_nodes'] * b.tables['map_nodes']['present'].unsqueeze(-1)
    owners = torch.cat((entity['global'], entity['monsters']), dim=1)
    for table, target, reference in (('powers', owners, 'owner_ref'),
            ('known_draw_top', entity['cards'], 'card_ref'),
            ('action_cards', entity['cards'], 'card_ref'),
            ('event_effects', entity['event_effects'], 'parent_ref')):
        linked = legacy._gather(target, legacy._column(b, table, reference))
        entity[table] = model.relation(torch.cat((entity[table], linked), -1))
        entity[table] = entity[table] * b.tables[table]['present'].unsqueeze(-1)
    pooled = []
    for name in legacy.CONTEXT_TABLES:
        count = b.tables[name]['present'].sum(1, keepdim=True).float()
        pooled.extend((entity[name].sum(1) / count.clamp_min(1), count.log1p()))
    context = torch.tanh(model.context(torch.cat(pooled, -1)) + states.phase(batch.phase_features))
    if not torch.isfinite(context).all():
        raise ValueError('non-finite run state context')
    return context, entity['cards']


def unique_roots(before, sizes):
    """Validate exact repeated input groups before sharing their computation.

    Public instances, unknown masks, phase and all relationships must match, not
    merely card IDs or a hash. This validation is included in benchmark timings.
    """
    if not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError('positive choice group sizes required')
    n = before.phase_features.shape[0]
    if sum(sizes) != n:
        raise ValueError('choice group sizes do not cover state batch')
    if before.phase_features.device.type != 'cpu':
        raise ValueError('CPU probe only')
    counts = torch.tensor(sizes)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
    owners = torch.repeat_interleave(torch.arange(len(sizes)), counts)

    def select(tensor):
        compact = tensor.index_select(0, starts)
        if not torch.equal(tensor, compact.index_select(0, owners)):
            raise ValueError('choice candidates have different before states')
        return compact

    b = before.entities
    compact = legacy.TensorBatch(b.schema_id,
        {name: {k: select(v) for k, v in table.items()} for name, table in b.tables.items()},
        select(b.action_mask), select(b.route))
    return StateBatch(before.schema_id, compact, select(before.phase_features)), owners


def choice_scores(model, batch, *, mode='cards', sizes=None):
    if mode not in MODES:
        raise ValueError('unknown local-choice computation mode')
    if mode == 'reference':
        return model(batch)
    if not isinstance(batch, ChoiceBatch) or batch.schema_id != model.schema_id:
        raise ValueError('versioned local choice batch required')
    before, owners = batch.before, None
    if mode == 'roots_cards':
        before, owners = unique_roots(before, sizes)
    left, left_cards = context_and_cards(model.states, before)
    if owners is not None:
        left = left.index_select(0, owners)
        left_cards = left_cards.index_select(0, owners)
    right, right_cards = context_and_cards(model.states, batch.after)
    cards = []
    for rows, ref in ((left_cards, batch.before_card_ref), (right_cards, batch.after_card_ref)):
        card = legacy._gather(rows, ref.unsqueeze(1)).squeeze(1)
        cards.extend((card, (ref > 0).float().unsqueeze(1)))
    preference = model.preference(torch.cat((left, right, *cards, batch.effects), -1)).squeeze(-1)
    if not torch.isfinite(preference).all():
        raise ValueError('non-finite local choice preference')
    return preference
