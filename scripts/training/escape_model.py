"""Training support for the released combat model (escape_model)."""

from .combat_value import QUANTILES
from .combat_value import SCALE
from .learned_continuation_contract import require_keys
from .rich_combat import Model as BinaryModel
from .rich_combat import export as binary_export
from .rich_combat import schema
from torch import nn
from torch.nn import functional as F
import math
import torch


VERSION = "public-combat-policy-value-v3-escape"


TARGET = "terminal-outcome-resource-quantiles-v2-escape"


ORDER = ["loss", "win", "escape"]


class Model(BinaryModel):
    def __init__(self, parent, seed=94103, spec=None):
        super().__init__(parent, seed, spec)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.outcome = nn.Linear(64, 3)
            self.resource = nn.Linear(64, 3 * QUANTILES)

    def forward(self, b):
        context, candidates, base_logits = self.base_features(b)
        ac, target = self.actor_extra(b)
        vc, _ = self.critic_extra(b)
        x = torch.cat((context[b["states"]], candidates, ac[b["states"]], target), 1)
        logits = base_logits + self.actor_output(self.actor_hidden(x).tanh()).squeeze(
            -1
        )
        hidden = (self.value_hidden(context) + self.value_extra(vc)).tanh()
        outcomes = self.outcome(hidden)
        quantiles = F.softplus(self.resource(hidden).reshape(-1, 3, QUANTILES)).cumsum(
            -1
        )
        if not all(torch.isfinite(v).all() for v in (logits, outcomes, quantiles)):
            raise ValueError("nonfinite escape output")
        return logits.split(b["lengths"]), outcomes, quantiles


def export(model):
    value = binary_export(model)
    value["version"] = VERSION
    value["semantics"].update(target=TARGET, outcome_order=ORDER)
    return value


def load(value):
    fields = {"version", "schema", "parent", "tensors", "semantics"}
    require_keys(value, fields, fields)
    if value["version"] != VERSION or value["schema"] != schema():
        raise ValueError("escape artifact version/schema")
    model = Model(value["parent"], spec=value["schema"])
    expected = export(model)
    if value["semantics"] != expected["semantics"] or set(value["tensors"]) != set(
        expected["tensors"]
    ):
        raise ValueError("escape artifact semantics")
    state = model.state_dict()
    for name, t in value["tensors"].items():
        if (
            set(t) != {"shape", "values"}
            or t["shape"] != list(state[name].shape)
            or len(t["values"]) != state[name].numel()
        ):
            raise ValueError("escape tensor shape")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3e38
            for v in t["values"]
        ):
            raise ValueError("escape tensor numeric")
        state[name] = torch.tensor(t["values"], dtype=torch.float32).reshape(t["shape"])
    model.load_state_dict(state)
    model.eval()
    return model


def terminal_target(result):
    if result is None:
        raise ValueError("censored battle has no value target")
    if (
        result.get("version") != "public-battle-result-v1"
        or type(result.get("outcome")) is not int
        or result["outcome"] not in (1, 2, 3)
    ):
        raise ValueError("unsupported terminal value outcome")
    hp, reserve = result.get("hp"), result.get("potion_reserve")
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0
        for x in (hp, reserve)
    ):
        raise ValueError("invalid terminal resources")
    return dict(
        outcome={2: 0, 1: 1, 3: 2}[result["outcome"]],
        resource=((0.0 if result["outcome"] == 2 else hp) + reserve) / SCALE,
    )
