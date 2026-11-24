"""Public-only bridge to the existing construction PPO critic, not preferences.

The critic predicts its recorded stage-return objective. Its accuracy at event
exits still needs outcome validation; this adapter does not calibrate it or
silently change its training target to whole-run victory probability.
"""
from .run_afterstates import capture
from .run_state_model import tensorize_states


class ConstructionEventValue:
    def __init__(self, encoder, model, *, objective, checkpoint_sha256):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("an independent frozen evaluation model is required")
        if not objective or len(checkpoint_sha256) != 64:
            raise ValueError("explicit objective and checkpoint digest required")
        self.encoder, self.model = encoder, model
        self.version = f"event-exit-construction-critic-v1/{objective}/{checkpoint_sha256}"

    def __call__(self, outcome):
        import torch
        if outcome.dead:
            return 0.0
        state = capture(outcome.public, relic_order_known=True, run_public=outcome.supplement)
        encoded = self.encoder.states.encode(state)
        with torch.inference_mode():
            context = self.model.common.states.encode_context(
                tensorize_states([encoded], self.encoder.states))
            value = self.model.common.states.backbone.value_head(context).reshape(())
        return float(value)
