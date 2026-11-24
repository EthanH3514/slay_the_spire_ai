"""Optional CPU PyTorch candidate actor/critic; public numeric inputs only.

The pilot learns run/route heads, while combat actions come from a fixed policy.
The critic can learn on every public transition, including fixed combat.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .public_encoding import EncodedDecision, PublicEncoder, TABLE_COLUMNS, collate_decisions


MODEL_VERSION = "public-entity-run-route-v1"
CATEGORIES = {"screen_id": "screen", "boss_id": "encounter", "room_id": "room",
    "input_id": "input", "task_id": "task", "event_id": "event", "card_id": "card",
    "pile_id": "pile", "type_id": "card_type", "monster_id": "monster", "move_id": "move",
    "power_id": "power", "relic_id": "relic", "potion_id": "potion", "action_id": "action",
    "option_id": "option", "key_id": "key", "operation_id": "operation", "rarity_id": "rarity",
    "pool_id": "pool", "selector_id": "selector"}
ENTITY_TABLES = tuple(name for name in TABLE_COLUMNS if name != "map_edges")
CONTEXT_TABLES = ("global", "cards", "monsters", "powers", "relics", "potions",
                  "map_nodes", "known_draw_top", "actions", "action_cards", "event_effects")


@dataclass
class TensorBatch:
    schema_id: str
    tables: dict
    action_mask: torch.Tensor
    route: torch.Tensor

    def payload(self):
        # Tensor-only input plus schema metadata: no Actions, seed or environment.
        return {"schema_id": self.schema_id, "tables": self.tables,
                "action_mask": self.action_mask, "route": self.route}


def tensorize(decisions, encoder: PublicEncoder) -> TensorBatch:
    if not decisions or any(not isinstance(d, EncodedDecision) for d in decisions):
        raise TypeError("tensor adapter accepts only EncodedDecision sequences")
    if any(d.schema_id != encoder.schema_id for d in decisions):
        raise ValueError("tensor encoder schema mismatch")
    source = collate_decisions(decisions)
    tables = {}
    for name, table in source["tables"].items():
        shape = (len(decisions), len(table["values"][0]), len(TABLE_COLUMNS[name]))
        values = torch.tensor(table["values"], dtype=torch.float32).reshape(shape)
        known = torch.tensor(table["known"], dtype=torch.bool).reshape(shape)
        present = torch.tensor(table["present"], dtype=torch.bool)
        if not torch.isfinite(values).all():
            raise ValueError("non-finite model features")
        for i, column in enumerate(TABLE_COLUMNS[name]):
            if column in CATEGORIES or column.endswith("_ref"):
                raw = values[..., i]
                if ((raw < 0) | (raw != raw.round())).any():
                    raise ValueError("invalid categorical/reference column")
                if column in CATEGORIES and (raw >= len(encoder.vocabularies[CATEGORIES[column]]) + 2).any():
                    raise ValueError("categorical token outside schema")
        tables[name] = {"values": values, "known": known, "present": present}
    mask = torch.tensor(source["action_mask"], dtype=torch.bool)
    if mask.ndim != 2 or not mask.any(dim=1).all():
        raise ValueError("every model row requires a scope-eligible action")
    screen_index = TABLE_COLUMNS["global"].index("screen_id")
    map_token = encoder.vocabularies["screen"].index("map") + 2
    route = tables["global"]["values"][:, 0, screen_index].long() == map_token
    return TensorBatch(encoder.schema_id, tables, mask, route)


def _column(batch, table, column):
    return batch.tables[table]["values"][..., TABLE_COLUMNS[table].index(column)]


def _gather(rows, references):
    """One-based entity joins, never treating row indexes as continuous features."""
    if ((references < 0) | (references != references.round()) | (references > rows.shape[1])).any():
        raise ValueError("model entity reference out of range")
    padded = torch.cat((rows.new_zeros((rows.shape[0], 1, rows.shape[2])), rows), dim=1)
    return padded.gather(1, references.long().unsqueeze(-1).expand(-1, -1, rows.shape[2]))


def _group(rows, owners, count, present):
    if ((owners < 0) | (owners != owners.round()) | (owners > count)).any():
        raise ValueError("model relationship owner out of range")
    active = present & (owners > 0)
    output = rows.new_zeros((rows.shape[0], count + 1, rows.shape[2]))
    counts = rows.new_zeros((rows.shape[0], count + 1, 1))
    output.scatter_add_(1, owners.long().unsqueeze(-1).expand_as(rows), rows * active.unsqueeze(-1))
    counts.scatter_add_(1, owners.long().unsqueeze(-1), active.unsqueeze(-1).to(rows.dtype))
    return output[:, 1:] / counts[:, 1:].clamp_min(1), counts[:, 1:]


class RowEncoder(nn.Module):
    def __init__(self, name, width, embedding_width=12):
        super().__init__()
        columns = TABLE_COLUMNS[name]
        self.categorical = [(i, CATEGORIES[c]) for i, c in enumerate(columns) if c in CATEGORIES]
        self.numeric = [i for i, c in enumerate(columns) if c not in CATEGORIES and not c.endswith("_ref")]
        self.network = nn.Sequential(nn.Linear(len(self.numeric) * 2 + len(self.categorical) * (embedding_width + 1), width),
                                     nn.Tanh(), nn.LayerNorm(width))

    def forward(self, table, embeddings):
        values, known = table["values"], table["known"]
        numbers = values[..., self.numeric]
        # Fixed signed-log compression; no trajectory-fitted statistics.
        parts = [numbers.sign() * numbers.abs().log1p(), known[..., self.numeric].float()]
        for i, kind in self.categorical:
            parts.extend((embeddings[kind](values[..., i].long()), known[..., i:i + 1].float()))
        output = self.network(torch.cat(parts, dim=-1))
        return output * table["present"].unsqueeze(-1)


class CandidateActorCritic(nn.Module):
    def __init__(self, encoder: PublicEncoder, width=32):
        super().__init__()
        if type(width) is not int or not 8 <= width <= 128:
            raise ValueError("pilot model width outside bounds")
        self.schema_id, self.width = encoder.schema_id, width
        self.embeddings = nn.ModuleDict({kind: nn.Embedding(len(names) + 2, 12, padding_idx=0)
                                         for kind, names in encoder.vocabularies.items()})
        self.rows = nn.ModuleDict({name: RowEncoder(name, width) for name in ENTITY_TABLES})
        self.map_message = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.Tanh())
        self.relation = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(len(CONTEXT_TABLES) * (width + 1), width), nn.Tanh())
        self.run_head = nn.Sequential(nn.Linear(width * 8 + 2, width), nn.Tanh(), nn.Linear(width, 1))
        self.route_head = nn.Sequential(nn.Linear(width * 8 + 2, width), nn.Tanh(), nn.Linear(width, 1))
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, batch: TensorBatch):
        if not isinstance(batch, TensorBatch) or batch.schema_id != self.schema_id:
            raise ValueError("model input schema mismatch")
        if not batch.action_mask.any(dim=1).all():
            raise ValueError("all actions masked")
        entity = {name: self.rows[name](batch.tables[name], self.embeddings) for name in ENTITY_TABLES}
        # Public topology: three outgoing-neighbor message rounds, not a full planner.
        edge_present = batch.tables["map_edges"]["present"]
        for _ in range(3):
            neighbors = _gather(entity["map_nodes"], _column(batch, "map_edges", "destination_ref"))
            messages, counts = _group(neighbors, _column(batch, "map_edges", "source_ref"),
                                      entity["map_nodes"].shape[1], edge_present)
            entity["map_nodes"] = self.map_message(torch.cat((entity["map_nodes"], messages, counts.log1p()), -1))
            entity["map_nodes"] = entity["map_nodes"] * batch.tables["map_nodes"]["present"].unsqueeze(-1)
        # Link public owner/order information instead of pooling unrelated IDs.
        owners = torch.cat((entity["global"], entity["monsters"]), dim=1)
        for table, target, reference in (("powers", owners, "owner_ref"),
            ("known_draw_top", entity["cards"], "card_ref"),
            ("action_cards", entity["cards"], "card_ref"),
            ("event_effects", entity["event_effects"], "parent_ref")):
            linked = _gather(target, _column(batch, table, reference))
            entity[table] = self.relation(torch.cat((entity[table], linked), -1))
            entity[table] = entity[table] * batch.tables[table]["present"].unsqueeze(-1)
        pooled = []
        for name in CONTEXT_TABLES:
            count = batch.tables[name]["present"].sum(1, keepdim=True).to(torch.float32)
            pooled.extend((entity[name].sum(1) / count.clamp_min(1), count.log1p()))
        context = self.context(torch.cat(pooled, dim=-1))
        actions = entity["actions"]
        linked_cards, n_cards = _group(entity["action_cards"], _column(batch, "action_cards", "action_ref"),
                                       actions.shape[1], batch.tables["action_cards"]["present"])
        effects, n_effects = _group(entity["event_effects"], _column(batch, "event_effects", "action_ref"),
                                    actions.shape[1], batch.tables["event_effects"]["present"])
        parts = [actions, context[:, None, :].expand_as(actions)]
        for table, column in (("cards", "card_ref"), ("potions", "potion_ref"),
                              ("monsters", "monster_ref"), ("map_nodes", "map_ref")):
            parts.append(_gather(entity[table], _column(batch, "actions", column)))
        parts.extend((linked_cards, effects, n_cards.log1p(), n_effects.log1p()))
        candidates = torch.cat(parts, dim=-1)
        logits = torch.where(batch.route[:, None], self.route_head(candidates).squeeze(-1),
                             self.run_head(candidates).squeeze(-1))
        values = self.value_head(context).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError("non-finite model outputs")
        return logits.masked_fill(~batch.action_mask, -torch.inf), values

    def distribution(self, batch):
        logits, values = self(batch)
        return torch.distributions.Categorical(logits=logits), values


def make_model(encoder, *, width=32, seed=731):
    # Model RNG is independent of simulator seeds and doesn't alter caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CandidateActorCritic(encoder, width)
