"""Same run/route architecture, expanded numeric card/context inputs only."""
import torch
from torch import nn

from . import learning_model as legacy
from .run_learning_encoding import EXTRA_COLUMNS, RunEncodedDecision, legacy_view


MODEL_VERSION = "public-entity-run-route-inputs-v2"


def tensorize(decisions, encoder):
    if not decisions or any(not isinstance(d, RunEncodedDecision) for d in decisions):
        raise TypeError("run model requires versioned supplemental inputs")
    batch = legacy.tensorize([legacy_view(d) for d in decisions], encoder)
    total = sum(len(decisions) * batch.tables[k]["values"].shape[1] * len(columns)
                for k, columns in EXTRA_COLUMNS.items())
    if total > 2_000_000:
        raise ValueError("supplemental batch exceeds memory budget; split, never truncate")
    for name, columns in EXTRA_COLUMNS.items():
        width = batch.tables[name]["values"].shape[1]
        values, known = [], []
        for d in decisions:
            t = d.extra_tables[name]
            if t.columns != columns or len(t.values) != len(d.tables[name].values):
                raise ValueError("supplemental entity alignment mismatch")
            padding = width - len(t.values)
            values.append(t.values + ((0.0,) * len(columns),) * padding)
            known.append(t.known + ((False,) * len(columns),) * padding)
        shape = (len(decisions), width, len(columns))
        table = batch.tables[name]
        table["extra_values"] = torch.tensor(values, dtype=torch.float32).reshape(shape)
        table["extra_known"] = torch.tensor(known, dtype=torch.bool).reshape(shape)
        if not torch.isfinite(table["extra_values"]).all():
            raise ValueError("non-finite supplemental features")
    return batch


class ExpandedRow(legacy.RowEncoder):
    def __init__(self, name, width):
        super().__init__(name, width)
        # Algebraically equivalent to concatenating inputs before the original
        # Linear. Zero initialization supports an explicit old-weight warm start.
        self.extra = nn.Linear(len(EXTRA_COLUMNS[name]) * 2, width, bias=False)
        nn.init.zeros_(self.extra.weight)

    def forward(self, table, embeddings):
        values, known = table["values"], table["known"]
        numbers = values[..., self.numeric]
        parts = [numbers.sign() * numbers.abs().log1p(), known[..., self.numeric].float()]
        for i, kind in self.categorical:
            parts.extend((embeddings[kind](values[..., i].long()), known[..., i:i + 1].float()))
        extra = table["extra_values"]
        extra = torch.cat((extra.sign() * extra.abs().log1p(), table["extra_known"].float()), -1)
        result = self.network[0](torch.cat(parts, -1)) + self.extra(extra)
        return self.network[2](self.network[1](result)) * table["present"].unsqueeze(-1)


class RunActorCritic(legacy.CandidateActorCritic):
    def __init__(self, encoder, width=32):
        super().__init__(encoder, width)
        for name in EXTRA_COLUMNS:
            previous = self.rows[name]
            replacement = ExpandedRow(name, width)
            replacement.network.load_state_dict(previous.network.state_dict())
            self.rows[name] = replacement

    def warm_start(self, old_model, encoder):
        if old_model.schema_id != encoder.legacy_schema_id or old_model.width != self.width:
            raise ValueError("incompatible legacy warm start")
        result = self.load_state_dict(old_model.state_dict(), strict=False)
        expected = {"rows." + name + ".extra.weight" for name in EXTRA_COLUMNS}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError("unexpected warm-start parameter mismatch")
        for name in EXTRA_COLUMNS:
            nn.init.zeros_(self.rows[name].extra.weight)
        return {"source_schema": old_model.schema_id, "target_schema": self.schema_id,
                "new_weights_zero": sorted(expected), "optimizer_resumed": False,
                "input_semantics_changed": True}


def make_model(encoder, *, width=32, seed=731):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RunActorCritic(encoder, width)
