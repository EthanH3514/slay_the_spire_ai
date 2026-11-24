from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .actions import Action, ActionKind
from .combat_search import StarterRulePolicy
from .environment import InvalidActionError
from .policy_api import PolicyInput, PolicySpecialist
from .policy_scope import ScopeDecision, assess_policy_scope


CORPUS_POLICY_VERSION = "public-corpus-policy-v2"


@dataclass(frozen=True, slots=True)
class CorpusDecision:
    action: Action
    specialist: PolicySpecialist
    mode: str


class PublicCorpusPolicy:
    """Deterministic public-only behavior policy for discovering later combat roots.

    This is a corpus generator, not a gameplay baseline. Combat uses the reviewed
    rule policy as a generic heuristic; every other context takes the first stable
    non-control action. Decision modes make weak fallbacks visible in the report.
    """

    version = CORPUS_POLICY_VERSION

    def __init__(self, combat: StarterRulePolicy | None = None) -> None:
        self.combat = combat or StarterRulePolicy()
        self.last_decision: CorpusDecision | None = None
        self.last_scope_decision: ScopeDecision | None = None

    def choose(self, public: PolicyInput) -> Action:
        self.last_decision = None
        self.last_scope_decision = None
        if not public.legal_actions:
            raise InvalidActionError("corpus policy received no legal actions")
        self.last_scope_decision = assess_policy_scope(public.observation, public.legal_actions)
        public = PolicyInput(public.observation, self.last_scope_decision.eligible_actions)
        observation = public.observation
        if isinstance(observation.get("combat"), Mapping):
            action = self.combat.choose(public)
            specialist = PolicySpecialist.COMBAT
            combat = observation["combat"]
            normal = str(combat.get("input_state", "")).upper() == "PLAYER_NORMAL"
            mode = "combat_rule" if normal else "combat_selection_native_rule"
        else:
            screen = observation.get("screen")
            screen_type = (
                str(screen.get("type", "")).upper() if isinstance(screen, Mapping) else ""
            )
            specialist = (
                PolicySpecialist.ROUTE
                if screen_type in {"MAP", "MAP_SCREEN"}
                else PolicySpecialist.RUN
            )
            action = self._first_stable_action(public.legal_actions)
            mode = f"{specialist.value}_first_legal:{screen_type or 'UNKNOWN'}"
        if action not in public.legal_actions:
            raise InvalidActionError(
                f"corpus policy returned an illegal action: {action.as_dict()}"
            )
        self.last_decision = CorpusDecision(action, specialist, mode)
        return action

    @staticmethod
    def _first_stable_action(actions: tuple[Action, ...]) -> Action:
        control_kinds = {ActionKind.WAIT, ActionKind.REQUEST_STATE, ActionKind.CANCEL}
        return next((action for action in actions if action.kind not in control_kinds), actions[0])
