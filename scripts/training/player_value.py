"""Training support for the released combat model (player_value)."""

from . import escape_model
from .combat_value import QUANTILES
from torch import nn
from torch.nn import functional as F
import copy
import math
import torch


VERSION = "public-combat-policy-value-v4-player"


FIELDS = [["current_hp", 100], ["max_hp", 100], ["block", 100], ["energy", 10]]


PATH = "direct-public-player-v1"


def numbers(view):
    result = []
    for name, scale in FIELDS:
        x = view["player"][name]
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1e9:
            raise ValueError("invalid public player number: " + name)
        result.append(x / scale)
    return result


class Model(escape_model.Model):
    def __init__(self, legacy):
        checked = escape_model.load(legacy)
        super().__init__(checked.parent, spec=checked.spec)
        self.load_state_dict(checked.state_dict())
        # Exact initial parity; only 256 additional value parameters.
        with torch.random.fork_rng(devices=[]):
            self.player_value = nn.Linear(len(FIELDS), 64, bias=False)
            nn.init.zeros_(self.player_value.weight)

    def value_parameters(self):
        return super().value_parameters() + list(self.player_value.parameters())

    def forward(self, b):
        context, candidates, base_logits = self.base_features(b)
        ac, target = self.actor_extra(b)
        vc, _ = self.critic_extra(b)
        x = torch.cat((context[b["states"]], candidates, ac[b["states"]], target), 1)
        logits = base_logits + self.actor_output(self.actor_hidden(x).tanh()).squeeze(
            -1
        )
        player = b["player_numbers"]
        if (
            player.shape != (len(context), len(FIELDS))
            or not torch.isfinite(player).all()
        ):
            raise ValueError("player input shape/finite")
        hidden = (
            self.value_hidden(context)
            + self.value_extra(vc)
            + self.player_value(player)
        ).tanh()
        outcomes = self.outcome(hidden)
        quantiles = F.softplus(self.resource(hidden).reshape(-1, 3, QUANTILES)).cumsum(
            -1
        )
        if not all(torch.isfinite(v).all() for v in (logits, outcomes, quantiles)):
            raise ValueError("nonfinite player-value output")
        return logits.split(b["lengths"]), outcomes, quantiles


def export(model):
    result = escape_model.export(model)
    result["version"] = VERSION
    result["semantics"].update(
        value_player_input=PATH, player_fields=copy.deepcopy(FIELDS)
    )
    return result


def load(value):
    legacy = copy.deepcopy(value)
    if legacy["version"] != VERSION:
        raise ValueError("player value version")
    legacy["version"] = escape_model.VERSION
    if (
        legacy["semantics"].pop("value_player_input") != PATH
        or legacy["semantics"].pop("player_fields") != FIELDS
    ):
        raise ValueError("player value semantics")
    tensor = legacy["tensors"].pop("player_value.weight")
    if (
        set(tensor) != {"shape", "values"}
        or tensor["shape"] != [64, len(FIELDS)]
        or len(tensor["values"]) != 64 * len(FIELDS)
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3e38
            for x in tensor["values"]
        )
    ):
        raise ValueError("player value tensor")
    model = Model(legacy)
    with torch.no_grad():
        model.player_value.weight.copy_(
            torch.tensor(tensor["values"]).reshape(64, len(FIELDS))
        )
    return model.eval()
