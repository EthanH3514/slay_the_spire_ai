"""Training support for the released combat model (combat_value)."""

from .leaf_actor import load_actor
from .leaf_actor import validate_artifact as validate_actor
from .leaf_numeric import schema
from .learned_continuation_contract import digest
from torch import nn
from torch.nn import functional as F
import math
import torch


VERSION = "public-combat-policy-value-v1"


TARGET = "terminal-outcome-resource-quantiles-v1"


QUANTILES = 21


SCALE = 100.0


LEVELS = [(i + 0.5) / QUANTILES for i in range(QUANTILES)]


def features(policy, batch):
    """Same operation order as leaf_batch; no new public or hidden features."""
    x = torch.cat(
        (policy.embedding(batch["categories"]).flatten(1), batch["numbers"]), 1
    )
    rows = policy.row(x).tanh()
    pooled = rows.new_zeros((len(batch["counts"]), rows.shape[1]))
    pooled.index_add_(0, batch["assignment"], rows)
    pooled = pooled / batch["counts"].clamp_min(1).unsqueeze(1)
    pooled = torch.cat((pooled, batch["counts"].log1p().unsqueeze(1)), 1)
    contexts = policy.context(pooled[batch["context"]].flatten(1)).tanh()
    candidates = pooled[batch["candidate_groups"]].flatten(1)
    return contexts, candidates


class CombatPolicyValue(nn.Module):
    def __init__(self, actor, seed=93103):
        super().__init__()
        self.policy = load_actor(actor)
        self.policy.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.value_hidden = nn.Linear(32, 64)
            self.outcome = nn.Linear(64, 2)
            self.resource = nn.Linear(64, 2 * QUANTILES)
            # Small positive initial increments; no resource upper bound.
            nn.init.normal_(self.resource.weight, std=0.01)
            nn.init.constant_(self.resource.bias, math.log(math.expm1(0.02)))

    @property
    def spec(self):
        return self.policy.spec

    def value(self, contexts):
        hidden = self.value_hidden(contexts.detach()).tanh()
        outcomes = self.outcome(hidden)
        quantiles = F.softplus(self.resource(hidden).reshape(-1, 2, QUANTILES)).cumsum(
            -1
        )
        if not torch.isfinite(outcomes).all() or not torch.isfinite(quantiles).all():
            raise ValueError("nonfinite combat value")
        return outcomes, quantiles

    def forward(self, batch):
        contexts, candidates = features(self.policy, batch)
        x = torch.cat((contexts[batch["candidate_states"]], candidates), 1)
        logits = self.policy.output(self.policy.head(x).tanh()).squeeze(-1)[
            batch["inverse"]
        ]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite combat policy")
        outcomes, quantiles = self.value(contexts)
        return logits.split(batch["lengths"]), outcomes, quantiles

    def value_parameters(self):
        return [
            p for name, p in self.named_parameters() if not name.startswith("policy.")
        ]


def validate_artifact(value, spec=None):
    expected = {
        "version",
        "actor",
        "value_tensors",
        "dtype",
        "target",
        "outcome_order",
        "quantile_levels",
        "resource_scale",
        "value_gradient_to_encoder",
        "continuation",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["version"] != VERSION
    ):
        raise ValueError("combat value artifact envelope")
    validate_actor(value["actor"], spec or schema())
    if (
        value["dtype"] != "float32"
        or value["target"] != TARGET
        or value["outcome_order"] != ["loss", "win"]
        or value["quantile_levels"] != LEVELS
        or value["resource_scale"] != SCALE
        or value["value_gradient_to_encoder"] is not False
        or value["continuation"]
        != dict(
            kind="actor_softmax", temperature=1.0, policy_digest=digest(value["actor"])
        )
    ):
        raise ValueError("combat value artifact semantics")
    shapes = {
        "value_hidden.weight": [64, 32],
        "value_hidden.bias": [64],
        "outcome.weight": [2, 64],
        "outcome.bias": [2],
        "resource.weight": [2 * QUANTILES, 64],
        "resource.bias": [2 * QUANTILES],
    }
    if not isinstance(value["value_tensors"], dict) or set(
        value["value_tensors"]
    ) != set(shapes):
        raise ValueError("combat value tensor names")
    for name, shape in shapes.items():
        t = value["value_tensors"][name]
        if (
            set(t) != {"shape", "values"}
            or t["shape"] != shape
            or len(t["values"]) != math.prod(shape)
        ):
            raise ValueError("combat value tensor shape")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4028234e38
            for x in t["values"]
        ):
            raise ValueError("combat value tensor values")
    return value


def load_model(value, spec=None):
    validate_artifact(value, spec)
    model = CombatPolicyValue(value["actor"])
    tensors = {
        **{"policy." + n: t for n, t in value["actor"]["tensors"].items()},
        **value["value_tensors"],
    }
    model.load_state_dict(
        {
            n: torch.tensor(t["values"], dtype=torch.float32).reshape(t["shape"])
            for n, t in tensors.items()
        }
    )
    model.eval()
    return model
