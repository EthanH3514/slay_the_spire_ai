"""Minibatch losses for root visits and terminal value labels."""

import torch
from torch.nn import functional as F

from . import critic_state_value
from .combat_value import LEVELS, QUANTILES
from .rich_combat import collate


def value_loss(model, rows):
    batch = critic_state_value.batch(rows)
    context = model.critic_state(batch["critic_state"])
    extra, _ = model.critic_extra(batch)
    hidden = (
        model.value_hidden(context)
        + model.value_extra(extra)
        + model.player_value(batch["player_numbers"])
    ).tanh()
    logits = model.outcome(hidden)
    quantiles = F.softplus(model.resource(hidden).reshape(-1, 3, QUANTILES)).cumsum(-1)
    classes = torch.tensor([row["label"]["outcome"] for row in rows])
    resources = torch.tensor([row["label"]["resource"] for row in rows])
    selected = quantiles[torch.arange(len(rows)), classes]
    error = resources[:, None] - selected
    levels = quantiles.new_tensor(LEVELS)
    pinball = torch.maximum(levels * error, (levels - 1) * error).mean(dim=1)
    return (F.cross_entropy(logits, classes, reduction="none") + pinball).mean()


def prior_loss(model, rows):
    batch = collate([row["packed"] for row in rows])
    context, candidates, base_logits = model.base_features(batch)
    extra, target = model.actor_extra(batch)
    indices = batch["states"]
    features = torch.cat((context[indices], candidates, extra[indices], target), dim=1)
    logits = base_logits + model.actor_output(
        model.actor_hidden(features).tanh()
    ).squeeze(-1)
    losses = []
    for row, menu in zip(rows, logits.split(batch["lengths"]), strict=True):
        visits = menu.new_tensor(row["visit_target"])
        if (
            visits.shape != menu.shape
            or len(visits) < 2
            or not torch.isfinite(visits).all()
            or (visits < 0).any()
            or not torch.isclose(
                visits.sum(), visits.new_tensor(1.0), atol=1e-6, rtol=0
            )
        ):
            raise ValueError(
                "root visits must form a normalized legal-action distribution"
            )
        losses.append(-(visits * menu.log_softmax(0)).sum())
    return torch.stack(losses).mean()


def update(model, optimizer, rows, kind):
    optimizer.zero_grad(set_to_none=True)
    if kind == "value":
        loss = value_loss(model, rows)
        parameters = model.value_parameters()
    else:
        loss = prior_loss(model, rows)
        parameters = model.actor_parameters()
    if not torch.isfinite(loss):
        raise ValueError(f"non-finite combat {kind} loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
