"""Training support for the released combat model (leaf_actor)."""

from .leaf_numeric import ACTION_GROUPS
from .leaf_numeric import NUMBERS
from .leaf_numeric import STATE_GROUPS
from .leaf_numeric import schema
from .learned_continuation_contract import canonical
from .learned_continuation_contract import digest
from torch import nn
import math
import torch


ARTIFACT = "public-leaf-actor-weights-v1"


class LeafActor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.embedding = nn.Embedding(len(spec["vocab"]), 8, padding_idx=0)
        self.row = nn.Linear(4 * 8 + 2 * NUMBERS, 32)
        self.context = nn.Linear(len(STATE_GROUPS) * 33, 32)
        self.head = nn.Linear(32 + len(ACTION_GROUPS) * 33, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, encoded):
        if encoded["schema_id"] != self.spec["id"]:
            raise ValueError("actor numeric schema mismatch")

        def pool(rows):
            if not rows:
                return self.row.weight.new_zeros(33)
            cats = torch.tensor([r["categories"] for r in rows], dtype=torch.long)
            numeric = torch.tensor(
                [r["values"] + r["known"] for r in rows], dtype=torch.float32
            )
            x = torch.cat((self.embedding(cats).flatten(1), numeric), 1)
            values = self.row(x).tanh().mean(0)
            return torch.cat((values, values.new_tensor([math.log1p(len(rows))])))

        context = self.context(
            torch.cat([pool(encoded["state"][g]) for g in STATE_GROUPS])
        ).tanh()
        # BLAS tail kernels can round identical rows differently. Compute each
        # identical candidate once, then gather: semantic ties stay exact and
        # gradients for duplicate choices still accumulate through the gather.
        unique = []
        positions = {}
        inverse = []
        for c in encoded["candidates"]:
            identity = canonical(c)
            if identity not in positions:
                positions[identity] = len(unique)
                unique.append(c)
            inverse.append(positions[identity])
        x = torch.stack(
            [torch.cat([context, *[pool(c[g]) for g in ACTION_GROUPS]]) for c in unique]
        )
        scores = self.output(self.head(x).tanh()).squeeze(-1)
        logits = scores[torch.tensor(inverse, dtype=torch.long)]
        mask = torch.tensor(encoded["mask"], dtype=torch.bool)
        if not torch.isfinite(logits).all() or not mask.any():
            raise ValueError("invalid actor logits/mask")
        return logits.masked_fill(~mask, -torch.inf)


def make_actor(spec=None, seed=7341):
    spec = spec or schema()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LeafActor(spec)
    return model


def validate_artifact(value, spec):
    if (
        set(value)
        != {"version", "schema", "tensors", "dtype", "architecture", "critic"}
        or value["version"] != ARTIFACT
    ):
        raise ValueError("actor artifact envelope")
    if value["schema"] != spec or spec["id"] != digest(
        {k: v for k, v in spec.items() if k != "id"}
    ):
        raise ValueError("actor artifact schema")
    if (
        value["dtype"] != "float32"
        or value["architecture"] != "combat_entity_candidate_actor_v1"
        or value["critic"] is not False
    ):
        raise ValueError("actor artifact architecture")
    shapes = {
        "embedding.weight": [len(spec["vocab"]), 8],
        "row.weight": [32, 96],
        "row.bias": [32],
        "context.weight": [32, 396],
        "context.bias": [32],
        "head.weight": [32, 164],
        "head.bias": [32],
        "output.weight": [1, 32],
        "output.bias": [1],
    }
    if set(value["tensors"]) != set(shapes):
        raise ValueError("actor tensor names")
    for name, shape in shapes.items():
        tensor = value["tensors"][name]
        if (
            set(tensor) != {"shape", "values"}
            or tensor["shape"] != shape
            or len(tensor["values"]) != math.prod(shape)
        ):
            raise ValueError("actor tensor shape")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4028234e38
            for x in tensor["values"]
        ):
            raise ValueError("actor tensor values")
    if any(x != 0 for x in value["tensors"]["embedding.weight"]["values"][:8]):
        raise ValueError("nonzero padding embedding")
    return value


def load_actor(value, spec=None):
    spec = spec or schema()
    validate_artifact(value, spec)
    model = make_actor(spec)
    model.load_state_dict(
        {
            k: torch.tensor(t["values"], dtype=torch.float32).reshape(t["shape"])
            for k, t in value["tensors"].items()
        }
    )
    model.eval()
    return model
