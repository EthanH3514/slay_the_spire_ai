from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol

from .actions import Action, ActionKind
from .environment import EnvironmentError, EnvironmentState, EpisodeExhaustedError
from .mechanism_evidence import (
    DEFAULT_MECHANISM_EVIDENCE, EvidenceCapability,
)
from .policy_api import PolicyInput, policy_input_from_state
from .hand_resident_curses import hand_resident_curse_reason
from .combat_selections import SELECTION_TASKS

if TYPE_CHECKING:
    from .simulator_environment import StsLightSpeedEnvironment


RESAMPLER_VERSION = "combat-public-resample-v12-opening-order-approx"
CARD_HISTORY_VERSION = "combat-public-card-history-v64"
CARD_SAMPLING_STATUS = "conditional_v60_recipes"
ENCOUNTER_SAMPLING_STATUS = "conditional_v11_recipes"
STREAM_NAMES = ("ai", "card_random", "misc", "monster_hp", "potion", "shuffle", "draw_order")
# Compatibility export derived from the evidence registry, not a second allowlist.
AUDITED_CARD_IDS = DEFAULT_MECHANISM_EVIDENCE.registered_ids(
    "card", EvidenceCapability.SEARCH,
)


class UnsupportedCombatSamplingError(EnvironmentError):
    """The prototype cannot preserve the public constraints of this root."""


def public_fingerprint(policy_input: PolicyInput) -> str:
    return policy_input.fingerprint()


def sampling_seeds(policy_input: PolicyInput, sample_seed: int) -> tuple[int, ...]:
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise ValueError("sample_seed must be an integer")
    if not 0 <= sample_seed < 2**64:
        raise ValueError("sample_seed must fit an unsigned 64-bit integer")
    fingerprint = public_fingerprint(policy_input)
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{RESAMPLER_VERSION}:{sample_seed}:{fingerprint}:{stream}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        for stream in STREAM_NAMES
    )


@dataclass(frozen=True, slots=True)
class CombatHistoryEntry:
    action: Action
    before_fingerprint: str
    after_fingerprint: str


@dataclass(frozen=True, slots=True)
class CombatPublicHistory:
    """Seed-free action ledger for one observed combat; not a full run ledger."""

    entries: tuple[CombatHistoryEntry, ...] = ()
    blocked_reason: str | None = None

    def record(
        self, before: EnvironmentState, action: Action, after: EnvironmentState
    ) -> CombatPublicHistory:
        before_input = policy_input_from_state(before)
        after_input = policy_input_from_state(after)
        return self._record_public(before_input, action, after_input)

    def _record_public(
        self, before_input: PolicyInput, action: Action, after_input: PolicyInput,
    ) -> CombatPublicHistory:
        """Internal adapter path: reuse snapshots already sanitized at read time.

        All transition/scope/history checks below are shared with ``record``;
        ordinary caller-owned EnvironmentStates still go through that factory.
        """
        reason = self.blocked_reason
        if reason is None:
            from .dynamic_damage import dynamic_state_reason
            from .play_triggers import play_trigger_state_reason
            from .forethought import free_play_state_reason
            from .top_card_autoplay import deferred_autoplay_reason
            for state in (before_input, after_input):
                combat = state.observation.get("combat", {})
                if isinstance(combat, Mapping):
                    from .encounter_public_state import encounter_public_state_reason, with_recoverable_cards
                    reason = encounter_public_state_reason(combat)
                    if reason is not None:
                        break
                    combat = with_recoverable_cards(combat)
                    reason = (
                        # Order uncertainty is a CURRENT-root sampling gate, not
                        # permanent history damage. A real shuffle can clear it.
                        hand_resident_curse_reason(combat, require_order_certainty=False)
                        or dynamic_state_reason(combat) or play_trigger_state_reason(combat)
                        or free_play_state_reason(combat, require_order_certainty=False)
                        or deferred_autoplay_reason(combat)
                    )
                    if reason is not None:
                        break
        if reason is None and action.kind is ActionKind.PLAY_CARD:
            combat = before_input.observation.get("combat", {})
            hand = combat.get("hand", ()) if isinstance(combat, Mapping) else ()
            index = action.card_index
            card = hand[index] if index is not None and 0 <= index < len(hand) else {}
            card_id = card.get("id", "") if isinstance(card, Mapping) else ""
            from .top_card_autoplay import autoplay_support_reason
            reason = autoplay_support_reason(combat, played=card)
            if reason is None:
                from .random_generation import generation_support_reason
                reason = generation_support_reason(before_input.observation, played=card)
            if reason is None and not DEFAULT_MECHANISM_EVIDENCE.supports(
                "card", card_id, EvidenceCapability.PUBLIC_HISTORY, imported=True,
            ):
                reason = (
                    f"public-history evidence missing for played card {card_id!r}"
                )
        elif reason is None and action.kind is ActionKind.CHOOSE:
            combat = before_input.observation.get("combat", {})
            is_audited_selection = (
                isinstance(combat, Mapping)
                and str(combat.get("input_state", "")).upper() == "CARD_SELECT"
                and str(combat.get("card_select_task", "")).upper()
                in SELECTION_TASKS
            )
            if not is_audited_selection:
                reason = "draw-order effects of choose are not audited for this selection"
            elif action not in before_input.legal_actions:
                reason = "selection action is not publicly legal"
            elif str(combat.get("card_select_task", "")).upper() == "LIQUID_MEMORIES_POTION":
                from .combat_selections import retrieval_choices
                try:
                    choices = retrieval_choices(combat)
                    if tuple(tuple(g) for g in combat.get("selection_options", ())) != choices:
                        raise ValueError("Liquid Memories requires complete public ordered choices")
                except ValueError as error:
                    reason = str(error)
            elif str(combat.get("card_select_task", "")).upper() == "GAMBLE":
                from .redraw_selection import pending_redraw_options
                try:
                    options = pending_redraw_options(combat)
                    if (action not in before_input.legal_actions or type(action.choice_index) is not int
                            or not 0 <= action.choice_index < len(options)):
                        raise ValueError("Gamble choice references no public subset")
                except ValueError as error:
                    reason = str(error)
            elif str(combat.get("card_select_task", "")).upper() == "CODEX":
                from .random_generation import generation_support_reason
                reason = generation_support_reason(before_input.observation)
                if reason is None and (action not in before_input.legal_actions
                        or type(action.choice_index) is not int or not 0 <= action.choice_index <= 3):
                    reason = "Codex choice references no public candidate or skip"
                # Insertion is NOT a shuffle. Native public top/bottom uncertainty
                # is retained and checked at each root, not cleared by this ledger.
            elif str(combat.get("card_select_task", "")).upper() == "FORETHOUGHT":
                from .forethought import pending_bottom_choices
                try:
                    choices = pending_bottom_choices(combat)
                    if type(action.choice_index) is not int or not 0 <= action.choice_index < len(choices):
                        raise ValueError("Forethought choice has no public ordered option")
                except ValueError as error:
                    reason = str(error)
            elif str(combat.get("card_select_task", "")).upper() == "EXHAUST_MANY":
                from .multi_exhaust import pending_exhaust_options
                try:
                    options = pending_exhaust_options(combat)
                    if type(action.choice_index) is not int or not 0 <= action.choice_index < len(options):
                        raise ValueError("multi-exhaust choice references no public subset")
                except ValueError as error:
                    reason = str(error)
            elif str(combat.get("card_select_task", "")).upper() == "DISCOVERY":
                from .random_generation import generation_support_reason
                reason = generation_support_reason(before_input.observation)
                if reason is None and (type(action.choice_index) is not int or not 0 <= action.choice_index < 3):
                    reason = "Discovery choice references no public candidate"
            elif str(combat.get("card_select_task", "")).upper() == "DUAL_WIELD":
                from .selected_copy import pending_copy_count
                try:
                    pending_copy_count(combat)
                except ValueError as error:
                    reason = str(error)
        elif reason is None and action.kind is ActionKind.END_TURN:
            # Persistent generation can execute after its source power card has
            # left all piles; apply the same scope checks at the turn boundary.
            from .random_generation import generation_support_reason
            reason = generation_support_reason(before_input.observation)
        elif reason is None and action.kind is ActionKind.USE_POTION:
            from .potion_contracts import potion_history_reason
            reason = potion_history_reason(before_input, action, after_input)
        elif reason is None and action.kind not in {
            ActionKind.END_TURN, ActionKind.DISCARD_POTION, ActionKind.PLAY_CARD
        }:
            reason = f"draw-order effects of {action.kind.value} are not audited"
        entry = CombatHistoryEntry(
            action, public_fingerprint(before_input),
            public_fingerprint(after_input),
        )
        return CombatPublicHistory(self.entries + (entry,), reason)


class CombatOutcome(str, Enum):
    ONGOING = "UNDECIDED"
    VICTORY = "PLAYER_VICTORY"
    LOSS = "PLAYER_LOSS"
    ESCAPED = "PLAYER_ESCAPED"


@dataclass(frozen=True, slots=True)
class CombatStepResult:
    policy_input: PolicyInput
    outcome: CombatOutcome
    reward: Mapping[str, int | float | bool]

    @property
    def done(self) -> bool:
        return self.outcome is not CombatOutcome.ONGOING


class CombatBranch:
    """Policy-facing single-combat branch: no reset, audit state, or exact clone."""

    resampler_version = RESAMPLER_VERSION

    def __init__(self, environment: StsLightSpeedEnvironment) -> None:
        self._environment = environment
        if environment._state is None:
            raise EnvironmentError("sampled combat branch has no state")
        self.policy_input = environment._current_public_input()
        self.outcome = CombatOutcome.ONGOING

    def step(self, action: Action) -> CombatStepResult:
        if self.outcome is not CombatOutcome.ONGOING:
            raise EpisodeExhaustedError("the sampled combat has ended")
        result = self._environment.step(action)
        try:
            self.outcome = CombatOutcome(result.state.info["combat_outcome"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvironmentError("native combat branch omitted its local outcome") from exc
        self.policy_input = self._environment._current_public_input()
        reward = MappingProxyType({
            key: value for key, value in result.reward.items()
            if key in {"hp_delta", "max_hp_delta", "gold_delta"}
        })
        return CombatStepResult(self.policy_input, self.outcome, reward)


class CombatSampler(Protocol):
    def sample(self, policy_input: PolicyInput, *, sample_seed: int) -> CombatBranch: ...


class BoundCombatSampler:
    """Give search only the public-sampling capability for its current live root."""

    def __init__(self, environment: StsLightSpeedEnvironment) -> None:
        self._environment = environment

    @property
    def conditional_roots(self) -> bool:
        return self._environment.session_search

    def sample(self, policy_input: PolicyInput, *, sample_seed: int) -> CombatBranch:
        # resample_combat still does a FRESH native read and checks this input.
        # Do not perform two identical root reads for the same sampling request.
        return self._environment.resample_combat(
            sample_seed=sample_seed, expected_public=policy_input,
        )

    def native_search(self, public: PolicyInput, seeds, budget, *, trace=False):
        return self._environment._native_combat_search(public, seeds, budget, trace=trace)

    def native_tree_search(self, public: PolicyInput, seeds, budget, *, trace=False):
        return self._environment._native_combat_search(public, seeds, budget, trace=trace, history_tree=True)
