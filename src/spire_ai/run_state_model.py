"""Experimental shared state encoder with separate preference and outcome heads.

No production policy imports this module. Empty legal-action sets are valid for
local afterstates: tensorization never inserts a dummy action. An untrained
outcome head is not a win probability. Preference fitting must freeze that head.
"""
from dataclasses import dataclass

import torch
from torch import nn

from . import learning_model as legacy
from .public_encoding import TABLE_COLUMNS, collate_decisions
from .run_afterstates import EncodedRunState, PHASES
from .run_learning_encoding import EXTRA_COLUMNS
from .run_learning_model import make_model


VERSION = "shared-local-run-state-model-v1"


@dataclass
class StateBatch:
    schema_id: str
    entities: legacy.TensorBatch
    phase_features: torch.Tensor


def tensorize_states(states, encoder):
    if not states or any(not isinstance(s, EncodedRunState) for s in states):
        raise TypeError("encoded run states required")
    if any(s.schema_id != encoder.schema_id for s in states):
        raise ValueError("run-state schema mismatch")
    decisions = [s.entities for s in states]
    source = collate_decisions(decisions)
    tables = {}
    extra_size = sum(len(states) * len(source["tables"][name]["values"][0]) * len(columns)
                     for name, columns in EXTRA_COLUMNS.items())
    if extra_size > 2000000:
        raise ValueError("supplemental state batch exceeds memory budget")
    for name, table in source["tables"].items():
        shape = (len(states), len(table["values"][0]), len(TABLE_COLUMNS[name]))
        values = torch.tensor(table["values"], dtype=torch.float32).reshape(shape)
        if not torch.isfinite(values).all():
            raise ValueError("non-finite state features")
        for i, column in enumerate(TABLE_COLUMNS[name]):
            if column in legacy.CATEGORIES or column.endswith("_ref"):
                raw = values[..., i]
                if ((raw < 0) | (raw != raw.round())).any():
                    raise ValueError("invalid state category/reference")
                if column in legacy.CATEGORIES and (
                        raw >= len(encoder.base.vocabularies[legacy.CATEGORIES[column]]) + 2).any():
                    raise ValueError("state category outside vocabulary")
        tables[name] = {"values": values,
            "known": torch.tensor(table["known"], dtype=torch.bool).reshape(shape),
            "present": torch.tensor(table["present"], dtype=torch.bool).reshape(shape[:2])}
        if name in EXTRA_COLUMNS:
            columns, width = EXTRA_COLUMNS[name], shape[1]
            extra_values, extra_known = [], []
            for decision in decisions:
                extra = decision.extra_tables[name]
                if extra.columns != columns or len(extra.values) != len(decision.tables[name].values):
                    raise ValueError("state supplement alignment mismatch")
                padding = width - len(extra.values)
                extra_values.append(extra.values + ((0.,) * len(columns),) * padding)
                extra_known.append(extra.known + ((False,) * len(columns),) * padding)
            shape_extra = (len(states), width, len(columns))
            tables[name]["extra_values"] = torch.tensor(extra_values, dtype=torch.float32).reshape(shape_extra)
            tables[name]["extra_known"] = torch.tensor(extra_known, dtype=torch.bool).reshape(shape_extra)
            if not torch.isfinite(tables[name]["extra_values"]).all():
                raise ValueError("non-finite state supplement")
    phases = []
    for state in states:
        if type(state.phase) is not int or not 0 <= state.phase < len(PHASES):
            raise ValueError("unknown state phase")
        if type(state.opportunities_known) is not bool:
            raise ValueError("opportunity knowledge must be boolean")
        phases.append([float(i == state.phase) for i in range(len(PHASES))] +
                      [float(state.opportunities_known)])
    mask = torch.tensor(source["action_mask"], dtype=torch.bool).reshape(len(states), -1)
    entities = legacy.TensorBatch(encoder.base.schema_id, tables, mask,
                                   torch.zeros(len(states), dtype=torch.bool))
    return StateBatch(encoder.schema_id, entities, torch.tensor(phases, dtype=torch.float32))


class SharedRunStateModel(nn.Module):
    def __init__(self, encoder, *, width=32, seed=731, warm_start=None):
        super().__init__()
        self.schema_id = encoder.schema_id
        self.backbone = make_model(encoder.base, width=width, seed=seed)
        if warm_start is not None:
            if warm_start.schema_id != encoder.base.schema_id or warm_start.width != width:
                raise ValueError("incompatible state encoder warm start")
            self.backbone.load_state_dict(warm_start.state_dict(), strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            self.phase = nn.Linear(len(PHASES) + 1, width, bias=False)
            self.preference = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        # Keep inherited policy heads only for explicit, reversible warm starts.
        self.backbone.run_head.requires_grad_(False)
        self.backbone.route_head.requires_grad_(False)
        self.backbone.value_head.requires_grad_(False)

    def encode_context(self, batch):
        if not isinstance(batch, StateBatch) or batch.schema_id != self.schema_id:
            raise ValueError("versioned state batch required")
        b, model = batch.entities, self.backbone
        entity = {name: model.rows[name](b.tables[name], model.embeddings) for name in legacy.ENTITY_TABLES}
        for _ in range(3):
            neighbors = legacy._gather(entity["map_nodes"], legacy._column(b, "map_edges", "destination_ref"))
            messages, counts = legacy._group(neighbors, legacy._column(b, "map_edges", "source_ref"),
                entity["map_nodes"].shape[1], b.tables["map_edges"]["present"])
            entity["map_nodes"] = model.map_message(torch.cat((entity["map_nodes"], messages, counts.log1p()), -1))
            entity["map_nodes"] = entity["map_nodes"] * b.tables["map_nodes"]["present"].unsqueeze(-1)
        owners = torch.cat((entity["global"], entity["monsters"]), dim=1)
        for table, target, reference in (("powers", owners, "owner_ref"),
                ("known_draw_top", entity["cards"], "card_ref"),
                ("action_cards", entity["cards"], "card_ref"),
                ("event_effects", entity["event_effects"], "parent_ref")):
            linked = legacy._gather(target, legacy._column(b, table, reference))
            entity[table] = model.relation(torch.cat((entity[table], linked), -1))
            entity[table] = entity[table] * b.tables[table]["present"].unsqueeze(-1)
        pooled = []
        for name in legacy.CONTEXT_TABLES:
            count = b.tables[name]["present"].sum(1, keepdim=True).float()
            pooled.extend((entity[name].sum(1) / count.clamp_min(1), count.log1p()))
        context = torch.tanh(model.context(torch.cat(pooled, -1)) + self.phase(batch.phase_features))
        if not torch.isfinite(context).all():
            raise ValueError("non-finite run state context")
        return context

    def forward(self, batch):
        context = self.encode_context(batch)
        preference = self.preference(context).squeeze(-1)
        value = self.backbone.value_head(context).squeeze(-1)
        if not torch.isfinite(preference).all() or not torch.isfinite(value).all():
            raise ValueError("non-finite run state output")
        return preference, value

    def preference_loss(self, groups, targets, encoder):
        if len(groups) != len(targets) or not groups or any(not group for group in groups):
            raise ValueError("nonempty candidate groups and aligned targets required")
        for group, target in zip(groups, targets, strict=True):
            if type(target) is not int or not 0 <= target < len(group):
                raise ValueError("invalid candidate preference target")
        scores, _ = self(tensorize_states([s for group in groups for s in group], encoder))
        losses, offset = [], 0
        for group, target in zip(groups, targets, strict=True):
            segment = scores[offset:offset+len(group)]
            losses.append(-segment.log_softmax(0)[target])
            offset += len(group)
        return torch.stack(losses).mean()
