"""Training support for the released combat model (critic_state_value)."""

from . import escape_model
from . import player_value
from .combat_value import QUANTILES
from .leaf_numeric import STATE_GROUPS
from .rich_combat import collate
from torch import nn
from torch.nn import functional as F
import copy
import math
import torch


VERSION = "public-combat-policy-value-v5-critic-state"


ENCODER = "independent-value-state-encoder-v1"


RAW = "public-value-state-rows-v1"


class StateEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.embedding = copy.deepcopy(policy.embedding)
        self.row = copy.deepcopy(policy.row)
        self.context = copy.deepcopy(policy.context)

    def forward(self, b):
        if b.get("version") != RAW:
            raise ValueError(
                "raw critic state rows required; frozen context is not a value cache"
            )
        x = torch.cat((self.embedding(b["categories"]).flatten(1), b["numbers"]), 1)
        rows = self.row(x).tanh()
        pooled = rows.new_zeros((len(b["counts"]), 32)).index_add(
            0, b["assignment"], rows
        )
        pooled = pooled / b["counts"].clamp_min(1)[:, None]
        pooled = torch.cat((pooled, b["counts"].log1p()[:, None]), 1)
        return self.context(pooled.reshape(-1, len(STATE_GROUPS) * 33)).tanh()


def prepare_state(old):
    """Only immutable public rows are reusable; no learned features are cached."""
    ns = len(STATE_GROUPS)
    counts = old["counts"][:ns]
    n = int(counts.sum())
    if (
        len(counts) != ns
        or n < 1
        or int((old["assignment"] < ns).sum()) != n
        or not torch.equal(
            old["assignment"][:n],
            torch.repeat_interleave(torch.arange(ns), counts.long()),
        )
    ):
        raise ValueError("state row grouping")
    return dict(
        version=RAW,
        categories=old["categories"][:n],
        numbers=old["numbers"][:n],
        assignment=old["assignment"][:n],
        counts=counts,
    )


def state_batch(parts):
    if not parts or any(p.get("version") != RAW for p in parts):
        raise ValueError("raw critic state cache version")
    return dict(
        version=RAW,
        categories=torch.cat([p["categories"] for p in parts]),
        numbers=torch.cat([p["numbers"] for p in parts]),
        assignment=torch.cat(
            [p["assignment"] + i * len(STATE_GROUPS) for i, p in enumerate(parts)]
        ),
        counts=torch.cat([p["counts"] for p in parts]),
    )


def batch(rows):
    b = collate([r["packed"] for r in rows])
    b["player_numbers"] = torch.tensor([r["player"] for r in rows], dtype=torch.float32)
    b["critic_state"] = state_batch([r["critic_state"] for r in rows])
    return b


class Model(player_value.Model):
    def __init__(self, original):
        checked = player_value.load(original)
        legacy = copy.deepcopy(original)
        legacy["version"] = escape_model.VERSION
        legacy["semantics"].pop("value_player_input")
        legacy["semantics"].pop("player_fields")
        legacy["tensors"].pop("player_value.weight")
        super().__init__(legacy)
        self.load_state_dict(checked.state_dict())
        self.critic_state = StateEncoder(self.base.policy)

    def value_parameters(self):
        return super().value_parameters() + list(self.critic_state.parameters())

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
            player.shape != (len(context), len(player_value.FIELDS))
            or not torch.isfinite(player).all()
        ):
            raise ValueError("player input shape/finite")
        state = self.critic_state(b["critic_state"])
        hidden = (
            self.value_hidden(state) + self.value_extra(vc) + self.player_value(player)
        ).tanh()
        outcomes = self.outcome(hidden)
        quantiles = F.softplus(self.resource(hidden).reshape(-1, 3, QUANTILES)).cumsum(
            -1
        )
        if not all(torch.isfinite(v).all() for v in (logits, outcomes, quantiles)):
            raise ValueError("nonfinite critic-state output")
        return logits.split(b["lengths"]), outcomes, quantiles


def export(model):
    a = player_value.export(model)
    a["version"] = VERSION
    a["semantics"].update(
        value_state_encoder=ENCODER, value_state_schema_id=model.base.policy.spec["id"]
    )
    return a


def load(artifact):
    a = copy.deepcopy(artifact)
    if a["version"] != VERSION:
        raise ValueError("critic-state model version")
    a["version"] = player_value.VERSION
    if (
        a["semantics"].pop("value_state_encoder", None) != ENCODER
        or a["semantics"].pop("value_state_schema_id", None)
        != a["parent"]["actor"]["schema"]["id"]
    ):
        raise ValueError("critic-state schema/semantics")
    extra = {
        k: a["tensors"].pop(k)
        for k in list(a["tensors"])
        if k.startswith("critic_state.")
    }
    m = Model(a)
    expected = {
        k: v for k, v in m.state_dict().items() if k.startswith("critic_state.")
    }
    if extra.keys() != expected.keys():
        raise ValueError("critic-state tensor names")
    with torch.no_grad():
        for k, t in extra.items():
            if (
                set(t) != {"shape", "values"}
                or t["shape"] != list(expected[k].shape)
                or len(t["values"]) != expected[k].numel()
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 3e38
                    for v in t["values"]
                )
            ):
                raise ValueError("critic-state tensor shape/values")
            expected[k].copy_(torch.tensor(t["values"]).reshape(expected[k].shape))
    if torch.count_nonzero(m.critic_state.embedding.weight[0]):
        raise ValueError("nonzero critic padding embedding")
    return m.eval()
