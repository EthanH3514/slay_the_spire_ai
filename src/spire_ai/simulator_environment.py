from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .actions import Action, ActionKind
from .card_recipes import (
    DEFAULT_CARD_RECIPES,
    EffectPrimitive,
    SelectedCardDestination,
    SelectedCardSource,
)
from .combat_sampling import (
    CombatBranch,
    CombatPublicHistory,
    UnsupportedCombatSamplingError,
    public_fingerprint,
    sampling_seeds,
)
from .policy_api import PolicyInput, hidden_enemy_intents, policy_input_from_state
from .pile_retrieval import RETRIEVAL_TASKS, eligible, retrieval_step
from .selected_copy import copy_eligible
from .rule_catalog import RuleCatalog
from .environment import (
    EnvironmentError,
    EnvironmentState,
    EpisodeExhaustedError,
    InvalidActionError,
    StepResult,
)
from .simulator import load_sts_lightspeed


_UNCHANGED = object()


def _selected_unique_id(
    pile: object, index: int | None, description: str,
) -> int:
    if not isinstance(pile, list) or index is None or not 0 <= index < len(pile):
        raise EnvironmentError(f"{description} references an invalid pile index")
    card = pile[index]
    unique_id = card.get("unique_id") if isinstance(card, Mapping) else None
    if isinstance(unique_id, bool) or not isinstance(unique_id, int):
        raise EnvironmentError(f"{description} card omitted its integer unique ID")
    return unique_id


@dataclass(frozen=True, slots=True)
class _SimulatorOption:
    action: Action
    command: str | None
    description: str
    virtual_action: str | None = None
    reward_key: tuple[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _RewardGroup:
    key: tuple[str, int]
    description: str
    options: tuple[_SimulatorOption, ...]


class StsLightSpeedEnvironment:
    """Live ``sts_lightspeed`` adapter for the backend-neutral environment API.

    The pinned simulator needs the project's small ``SimulatorSession`` binding
    patch. Keeping command strings inside this adapter prevents console grammar from
    entering traces, policies or model vocabularies.
    """

    backend_name = "sts_lightspeed"

    def __init__(
        self,
        module_dir: Path,
        *,
        compiler_runtime_dir: Path | None = None,
        default_seed: int = 123456789,
        default_character: str = "IRONCLAD",
        default_ascension: int = 0,
        module: ModuleType | None = None,
        session_search: bool = False,
    ) -> None:
        self.module = module or load_sts_lightspeed(
            module_dir, compiler_runtime_dir=compiler_runtime_dir
        )
        session_type = getattr(self.module, "SimulatorSession", None)
        if session_type is None:
            raise EnvironmentError(
                "sts_lightspeed does not expose SimulatorSession; rebuild the pinned fork "
                "using scripts/bootstrap_sts_lightspeed.ps1 and simulator-lock.json"
            )
        self.session = session_type()
        self.session_search = session_search
        if session_search:
            from .native_search import compatible_session
            if not compatible_session(self.module) or not callable(getattr(self.session, "enable_public_session", None)):
                raise EnvironmentError("conditional session search contract/version mismatch; no legacy fallback")
            self.session.enable_public_session(True)
        self.default_seed = default_seed
        self.default_character = default_character.upper()
        self.default_ascension = default_ascension
        self._options: tuple[_SimulatorOption, ...] = ()
        self._state: EnvironmentState | None = None
        self._public_input: PolicyInput | None = None
        self._step = 0
        self._active_card_reward: tuple[str, int] | None = None
        self._skipped_card_rewards: set[tuple[str, int]] = set()
        self._combat_history: CombatPublicHistory | None = None
        self._known_draw_top_unique_id: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        character: str | None = None,
        ascension: int | None = None,
    ) -> EnvironmentState:
        resolved_seed = self.default_seed if seed is None else seed
        resolved_character = self.default_character if character is None else character.upper()
        resolved_ascension = self.default_ascension if ascension is None else ascension
        if not 0 <= resolved_ascension <= 20:
            raise EnvironmentError("ascension must be between 0 and 20")
        try:
            character_value = getattr(self.module.CharacterClass, resolved_character)
        except AttributeError as exc:
            raise EnvironmentError(
                f"unsupported simulator character: {resolved_character}"
            ) from exc

        self._public_input = None
        try:
            self.session.reset(character_value, resolved_seed, resolved_ascension)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise EnvironmentError(f"failed to reset sts_lightspeed: {exc}") from exc
        self._step = 0
        self._active_card_reward = None
        self._skipped_card_rewards.clear()
        self._known_draw_top_unique_id = None
        self._state = self._read_state()
        self._combat_history = (
            CombatPublicHistory() if "combat" in self._state.observation else None
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise EpisodeExhaustedError("reset must be called before step")
        if not bool(self._state.observation.get("in_game", False)):
            raise EpisodeExhaustedError("the simulator episode has ended")

        option = next((option for option in self._options if option.action == action), None)
        if option is None:
            raise InvalidActionError(
                f"action is not legal at simulator step {self._step}: {action.as_dict()}"
            )

        before_state = self._state
        before_input = self._current_public_input()
        self._public_input = None
        before = before_state.observation
        pending_known_top: int | None | object = _UNCHANGED
        if option.virtual_action == "open_card_reward":
            self._active_card_reward = option.reward_key
        elif option.virtual_action == "skip_card_reward":
            if option.reward_key is not None:
                self._skipped_card_rewards.add(option.reward_key)
            self._active_card_reward = None
        else:
            if option.command is None:
                raise EnvironmentError(
                    f"simulator option has no executable command: {action.as_dict()}"
                )
            self._active_card_reward = None
            pending_known_top = self._known_top_after_action(before_state, action)
            try:
                if self.session_search:
                    self.session.take_public_action(option.command, json.dumps(action.as_dict()))
                else:
                    self.session.take_action(option.command)
            except (AttributeError, TypeError, RuntimeError) as exc:
                raise EnvironmentError(
                    f"sts_lightspeed rejected command for {action.as_dict()}: {exc}"
                ) from exc
            if pending_known_top is not _UNCHANGED:
                self._known_draw_top_unique_id = pending_known_top

        current_step = self._step
        self._step += 1
        self._state = self._read_state()
        after = self._state.observation
        if "combat" not in after:
            self._combat_history = None
        elif "combat" not in before:
            self._combat_history = CombatPublicHistory()
        elif self._combat_history is not None:
            self._combat_history = self._combat_history._record_public(
                before_input, action, self._current_public_input(),
            )
        terminated = not bool(after.get("in_game", False))
        reward = _reward_components(before, after, terminated)
        return StepResult(
            state=self._state,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "backend": self.backend_name,
                "step": current_step,
                "action_description": option.description,
            },
        )

    def clone_exact(self) -> StsLightSpeedEnvironment:
        """Return an independent, bit-exact branch for oracle/search code.

        This capability is deliberately absent from ``SpireEnvironment``. Normal
        policies receive only ``PolicyInput`` and cannot use exact cloning to read
        or search over hidden RNG state.
        """

        if self._state is None:
            raise EpisodeExhaustedError("reset must be called before clone_exact")
        clone_method = getattr(self.session, "clone_exact", None)
        if clone_method is None:
            raise EnvironmentError(
                "sts_lightspeed does not expose clone_exact; rebuild the pinned fork "
                "using scripts/bootstrap_sts_lightspeed.ps1 and simulator-lock.json"
            )

        try:
            session = clone_method()
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise EnvironmentError(f"failed to clone sts_lightspeed state: {exc}") from exc
        return self._copy_with_session(session)

    def _prepare_combat_sampling(
        self, expected_public: PolicyInput | None = None, *, conditioned: bool = False,
    ) -> PolicyInput:
        """Shared fresh-root checks for reference and native public sampling."""

        if self._state is None:
            raise EpisodeExhaustedError("reset must be called before resample_combat")
        current = self._read_state()
        current_input = self._current_public_input()
        if (expected_public is not None
                and public_fingerprint(current_input) != public_fingerprint(expected_public)):
            raise EnvironmentError("combat sampler input does not match the current root")
        if "combat" not in current.observation or self._combat_history is None:
            raise UnsupportedCombatSamplingError("an observed active combat is required")
        if self._combat_history.blocked_reason is not None:
            raise UnsupportedCombatSamplingError(self._combat_history.blocked_reason)
        method = getattr(self.session, "resample_combat", None)
        if method is None:
            raise EnvironmentError(
                "native resample_combat is missing; reapply the patch and rebuild"
            )
        from .dynamic_damage import dynamic_state_reason
        from .play_triggers import play_trigger_state_reason
        from .forethought import free_play_state_reason
        combat = current_input.observation.get("combat", {})
        from .public_piles import draw_order_sampling_reason
        from .encounter_public_state import encounter_public_state_reason, with_recoverable_cards
        encounter_reason = encounter_public_state_reason(combat)
        combat = with_recoverable_cards(combat)
        from .hand_resident_curses import hand_resident_curse_reason
        reason = (
            (None if conditioned else draw_order_sampling_reason(combat))
            or encounter_reason or hand_resident_curse_reason(combat, require_order_certainty=not conditioned) or dynamic_state_reason(combat)
            or play_trigger_state_reason(combat) or free_play_state_reason(combat, require_order_certainty=not conditioned)
        )
        if reason is None:
            from .top_card_autoplay import autoplay_support_reason
            reason = autoplay_support_reason(combat)
        if reason is None:
            from .random_generation import generation_support_reason
            reason = generation_support_reason(current_input.observation)
        if reason is not None:
            raise UnsupportedCombatSamplingError(reason)
        if hidden_enemy_intents(current_input.observation):
            raise UnsupportedCombatSamplingError(
                "Runic Dome hides the current enemy intent from combat sampling"
            )
        return current_input

    def resample_combat(
        self, *, sample_seed: int, expected_public: PolicyInput | None = None,
    ) -> CombatBranch:
        """Approximate, combat-only public sampling; never a run-level oracle."""
        if self.session_search:
            raise EnvironmentError("conditional session must use native_search, not legacy independent resampling")
        current_input = self._prepare_combat_sampling(expected_public)
        method = self.session.resample_combat
        seeds = sampling_seeds(current_input, sample_seed)
        try:
            known_top = (
                -1
                if self._known_draw_top_unique_id is None
                else self._known_draw_top_unique_id
            )
            session = method(seeds, known_top)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise UnsupportedCombatSamplingError(
                f"native combat sampling rejected root: {exc}"
            ) from exc
        clone = self._copy_with_session(session)
        assert clone._state is not None
        if clone._current_public_input().as_dict() != current_input.as_dict():
            raise EnvironmentError("combat resampling changed public information")
        return CombatBranch(clone)

    def _native_combat_search(self, public: PolicyInput, seeds, budget, *, trace=False, history_tree=False):
        from .native_search import compatible_versions, metadata_json, projection_json
        from .event_sampling import STRICT_ROOT_MODEL, EVENT_ROOT_MODEL
        model = budget.get("root_model", STRICT_ROOT_MODEL)
        if model not in {STRICT_ROOT_MODEL, EVENT_ROOT_MODEL}:
            raise EnvironmentError("unknown root probability model")
        if model == EVENT_ROOT_MODEL and (
            not self.session_search or getattr(self.module, "public_event_offer_abi", None) != EVENT_ROOT_MODEL
        ):
            raise EnvironmentError("explicit event probability module required; no strict-model substitution")
        from .history_tree_search import COMBAT_OBJECTIVE_VERSION, OBJECTIVE_TREE_VERSION, EFFICIENT_TREE_VERSION
        objective = budget.get("combat_objective")
        if objective is not None and (objective != COMBAT_OBJECTIVE_VERSION or not self.session_search
                or getattr(self.module, "public_combat_objective_abi", None) != objective):
            raise EnvironmentError("explicit combat-objective module required")
        if not callable(getattr(self.session, "search_public", None)):
            return None
        if not compatible_versions(self.module):
            return {"unavailable": "native_contract_version_mismatch", "steps": 0, "samples": 0}
        if history_tree:
            from .history_tree_search import TREE_SEARCH_VERSION, GUARDED_TREE_VERSION, PAIRED_TREE_VERSION
            version = budget.get("tree_algorithm", TREE_SEARCH_VERSION)
            entries = {TREE_SEARCH_VERSION: ("public_tree_search_abi", "search_session_tree"),
                       GUARDED_TREE_VERSION: ("public_guarded_tree_search_abi", "search_session_guarded_tree"),
                       PAIRED_TREE_VERSION: ("public_paired_tree_search_abi", "search_session_paired_tree"),
                       OBJECTIVE_TREE_VERSION: ("public_objective_tree_search_abi", "search_session_paired_tree"),
                       EFFICIENT_TREE_VERSION: ("public_efficient_tree_search_abi", "search_session_paired_tree")}
            if version not in entries:
                raise EnvironmentError("unknown history-tree algorithm")
            abi, entry = entries[version]
            if (not self.session_search or getattr(self.module, abi, None) != version
                    or not callable(getattr(self.session, entry, None))):
                raise EnvironmentError("explicit history-tree session contract required; no flat fallback")
        import time
        started = time.perf_counter()
        current = self._prepare_combat_sampling(public, conditioned=self.session_search)
        import json
        candidates = [a.as_dict() for a in current.legal_actions]
        expected, encoded_actions, metadata = projection_json(current), json.dumps(candidates), metadata_json()
        if self.session_search:
            from .native_search import session_projection_json, compatible_session
            if not compatible_session(self.module):
                raise EnvironmentError("conditional session contract changed; no legacy fallback")
            expected = session_projection_json(current)
        budget = dict(budget)
        if budget["max_seconds"] is not None:
            budget["max_seconds"] = max(0., budget["max_seconds"] - (time.perf_counter() - started))
        if self.session_search:
            method = getattr(self.session, entry) if history_tree else self.session.search_session_public
            result = method(seeds, expected, encoded_actions, json.dumps(budget), metadata, trace)
        else:
            result = self.session.search_public(
                seeds, -1 if self._known_draw_top_unique_id is None else self._known_draw_top_unique_id,
                expected, encoded_actions, json.dumps(budget), metadata, trace,
            )
        return json.loads(result)

    def _copy_with_session(self, session: Any) -> StsLightSpeedEnvironment:
        clone = object.__new__(type(self))
        clone.module = self.module
        clone.session = session
        clone.session_search = self.session_search
        clone.default_seed = self.default_seed
        clone.default_character = self.default_character
        clone.default_ascension = self.default_ascension
        clone._options = ()
        clone._state = None
        clone._public_input = None
        clone._step = self._step
        clone._active_card_reward = self._active_card_reward
        clone._skipped_card_rewards = set(self._skipped_card_rewards)
        clone._combat_history = self._combat_history
        clone._known_draw_top_unique_id = self._known_draw_top_unique_id
        clone._state = clone._read_state()
        return clone

    def _read_state(self) -> EnvironmentState:
        # Explicit reads are refresh boundaries, including audit/out-of-band
        # native changes. Only successful reads may publish a new owned snapshot.
        self._public_input = None
        try:
            observation = copy.deepcopy(dict(self.session.get_state()))
            raw_audit_state = observation.pop("rng_state", {})
            combat_outcome = observation.pop("combat_outcome", None)
            moved_top = observation.pop("public_moved_draw_top_unique_id", -1)
            features = tuple(self.session.get_observation())
            action_text = str(self.session.get_actions_text()) if observation.get("in_game") else ""
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise EnvironmentError(f"failed to read sts_lightspeed state: {exc}") from exc

        combat = observation.get("combat")
        if isinstance(combat, Mapping) and combat.get("public_known_top_ids"):
            # The suffix is exported and sampled natively. Do not copy its last
            # ID into the legacy single-top cache: it could survive a shuffle.
            self._known_draw_top_unique_id = None
            moved_top = -1
        # A public movement event, including automatic selection after draws.
        # Never infer its identity from the hidden final draw-pile order.
        if type(moved_top) is int and moved_top >= 0:
            self._known_draw_top_unique_id = moved_top
        if not isinstance(combat, Mapping):
            self._known_draw_top_unique_id = None
        elif self._known_draw_top_unique_id is not None:
            draw_pile = combat.get("draw_pile")
            actual_top = (
                draw_pile[-1]
                if isinstance(draw_pile, list) and draw_pile
                else None
            )
            if (
                not isinstance(actual_top, Mapping)
                or actual_top.get("unique_id") != self._known_draw_top_unique_id
            ):
                self._known_draw_top_unique_id = None

        screen = observation.setdefault("screen", {})
        screen_type = str(_mapping(screen).get("type", "")).upper()
        if not bool(observation.get("in_game", False)):
            # Native terminal battles can retain an earlier CARD_SELECT task.
            # A terminal state has no decisions, regardless of stale screen data.
            self._active_card_reward = None
            self._skipped_card_rewards.clear()
            self._options = ()
        elif screen_type == "REWARDS":
            self._options = tuple(self._parse_reward_options(action_text))
            if isinstance(screen, dict):
                virtual_type = (
                    "CARD_REWARD" if self._active_card_reward is not None else "COMBAT_REWARD"
                )
                screen["type"] = virtual_type
                screen["name"] = virtual_type
                screen["is_up"] = True
        else:
            self._active_card_reward = None
            self._skipped_card_rewards.clear()
            self._options = tuple(_parse_options(observation, action_text, public_occurrences=self.session_search))
        descriptions = [option.description for option in self._options]
        if isinstance(screen, dict):
            screen["choices"] = descriptions
            if not isinstance(combat, Mapping) and screen.get("type") != "CARD_REWARD":
                from .run_options import native_run_options

                screen.update(native_run_options(observation, self._options, action_text))
            if screen.get("type") == "CARD_REWARD":
                from .reward_policy import native_reward_card

                screen["card_reward_options"] = [
                    {"action": option.action.as_dict(),
                     "card": native_reward_card(option.description)
                     if option.command and option.command.startswith("card ") else None}
                    for option in self._options
                ]
        state = EnvironmentState(
            observation=observation,
            legal_actions=tuple(option.action for option in self._options),
            backend_features=features,
            audit_state={"rng": copy.deepcopy(dict(raw_audit_state))}
            if isinstance(raw_audit_state, Mapping)
            else {},
            info={
                "backend": self.backend_name,
                "step": self._step,
                "action_descriptions": descriptions,
                "observation_fidelity": "structured+sts_lightspeed-412",
                **(
                    {
                        "public_draw_pile_known_top_unique_id":
                            self._known_draw_top_unique_id,
                    }
                    if self._known_draw_top_unique_id is not None
                    else {}
                ),
                **({"combat_outcome": combat_outcome} if combat_outcome is not None else {}),
            },
        )
        self._public_input = policy_input_from_state(state)
        return state

    def _current_public_input(self) -> PolicyInput:
        """Owned, sanitized view of the latest read, with no backend reference.

        _read_state always rebuilds this; steps, reset and both kinds of branch
        creation read anew. No cache is attached to caller-mutable EnvironmentState.
        """
        if self._public_input is None:
            raise EnvironmentError("no validated simulator public snapshot")
        return self._public_input

    @staticmethod
    def _known_top_after_action(
        before: EnvironmentState, action: Action,
    ) -> int | None | object:
        """Return a newly public draw-pile top ID, or ``_UNCHANGED``.

        Headbutt and hand-to-top selections make only the selected identity
        public. Automatic post-draw selection is supplied by a native movement
        event instead; it cannot safely be inferred from the pre-action hand.
        """

        combat = before.observation.get("combat")
        if not isinstance(combat, Mapping):
            return _UNCHANGED
        if action.kind is ActionKind.END_TURN:
            from .mayhem import mayhem_stacks
            try:
                active = mayhem_stacks(combat)
            except ValueError:
                active = True  # Invalid state is refused separately; never keep stale knowledge.
            if active:
                return None
        if action.kind is ActionKind.CHOOSE:
            task = str(combat.get("card_select_task", "")).upper()
            is_topdeck = (
                str(combat.get("input_state", "")).upper() == "CARD_SELECT"
                and task in {"HEADBUTT", "WARCRY"}
            )
            if not is_topdeck:
                return _UNCHANGED
            return _selected_unique_id(
                combat.get("discard_pile" if task == "HEADBUTT" else "hand"),
                action.choice_index, "public topdeck selection",
            )
        if action.kind is not ActionKind.PLAY_CARD:
            return _UNCHANGED
        hand = combat.get("hand")
        index = action.card_index
        if not isinstance(hand, list) or index is None or not 0 <= index < len(hand):
            return _UNCHANGED
        card = hand[index]
        recipe = DEFAULT_CARD_RECIPES.lookup(
            card.get("id") if isinstance(card, Mapping) else None
        )
        if recipe and any(step.primitive is EffectPrimitive.AUTOPLAY_TOP_CARD for step in recipe.effects):
            # The old top is consumed. Even if the same UUID happens to return
            # after a hidden shuffle, that coincidence is not public knowledge.
            # A new explicit native topdeck event can re-establish knowledge.
            return None
        if recipe and combat.get("discard_pile") and any(
            step.primitive is EffectPrimitive.SHUFFLE_DISCARD_INTO_DRAW for step in recipe.effects
        ):
            # Discard is checked at use time. No Draw/full hand does not prevent
            # this shuffle, and a coincidentally unchanged top is not knowledge.
            return None
        upgraded = (
            int(card.get("upgrades", card.get("upgraded", 0))) > 0
            if isinstance(card, Mapping)
            else False
        )
        if recipe is None or not any(
            step.primitive is EffectPrimitive.MOVE_SELECTED_CARD
            and step.selected_card_source is SelectedCardSource.DISCARD_PILE
            and step.selected_card_destination is SelectedCardDestination.DRAW_PILE_TOP
            and step.amount(upgraded) > 0
            for step in recipe.effects
        ):
            return _UNCHANGED
        discard = combat.get("discard_pile")
        if not isinstance(discard, list) or len(discard) != 1:
            return _UNCHANGED
        return _selected_unique_id(discard, 0, "automatic selected-card movement")

    def _parse_reward_options(self, action_text: str) -> list[_SimulatorOption]:
        groups, proceed = _parse_reward_groups(action_text)
        visible_groups = [
            group for group in groups if group.key not in self._skipped_card_rewards
        ]

        if self._active_card_reward is not None:
            group = next(
                (group for group in visible_groups if group.key == self._active_card_reward),
                None,
            )
            if group is None or group.key[0] != "card":
                self._active_card_reward = None
            else:
                options = [
                    _SimulatorOption(
                        Action(ActionKind.CHOOSE, choice_index=choice_index),
                        candidate.command,
                        candidate.description,
                    )
                    for choice_index, candidate in enumerate(group.options)
                ]
                options.append(
                    _SimulatorOption(
                        Action(ActionKind.PROCEED),
                        None,
                        "skip card reward",
                        virtual_action="skip_card_reward",
                        reward_key=group.key,
                    )
                )
                return options

        options: list[_SimulatorOption] = []
        for choice_index, group in enumerate(visible_groups):
            if group.key[0] == "card":
                options.append(
                    _SimulatorOption(
                        Action(ActionKind.CHOOSE, choice_index=choice_index),
                        None,
                        group.description,
                        virtual_action="open_card_reward",
                        reward_key=group.key,
                    )
                )
            else:
                direct = group.options[0]
                options.append(
                    _SimulatorOption(
                        Action(ActionKind.CHOOSE, choice_index=choice_index),
                        direct.command,
                        direct.description,
                    )
                )
        if proceed is not None:
            options.append(
                _SimulatorOption(Action(ActionKind.PROCEED), proceed[0], proceed[1])
            )
        return options


def _parse_options(
    observation: Mapping[str, Any], action_text: str, *, public_occurrences: bool = False
) -> list[_SimulatorOption]:
    def selection_key(card):
        # New session actions use public profile order, with stable occurrence
        # indices for indistinguishable copies. Native IDs remain adapter-only.
        value = {**card, "unique_id": -1} if public_occurrences else card
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    screen = _mapping(observation.get("screen"))
    combat = _mapping(observation.get("combat"))
    is_normal_combat = (
        str(screen.get("type", "")).upper() == "BATTLE"
        and str(combat.get("input_state", "")).upper() == "PLAYER_NORMAL"
    )
    if is_normal_combat:
        return _parse_normal_combat_options(action_text)
    task = str(combat.get("card_select_task", "")).upper()
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "GAMBLE":
        from .redraw_selection import redraw_subsets
        hand = combat.get("hand", ())
        try:
            groups = [list(group) for group in redraw_subsets(combat)]
        except ValueError as error:
            raise EnvironmentError(str(error)) from error
        if {o.command for o in _parse_choice_options(action_text)} != {str(i) for i in range(len(hand))}:
            raise EnvironmentError("Gamble commands disagree with public hand")
        combat["selection_options"] = groups
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i),
            " ".join(map(str, reversed(group))) if group else "none", "Discard and redraw hand indices " + str(group))
            for i, group in enumerate(groups)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "CODEX":
        candidates = combat.get("selection_candidates")
        if (not isinstance(candidates, list) or len(candidates) != 3
                or any(not isinstance(c, Mapping) or not c.get("id") for c in candidates)
                or {o.command for o in _parse_choice_options(action_text)} != {"0", "1", "2", "3"}):
            raise EnvironmentError("Codex requires three public candidates and explicit skip")
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i), str(i),
            "Skip Codex" if i == 3 else "Shuffle " + candidates[i]["id"]) for i in range(4)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "LIQUID_MEMORIES_POTION":
        source = combat.get("discard_pile", ())
        if (not isinstance(source, list) or not source or any(not isinstance(c, Mapping) or not c.get("id") for c in source)
                or {o.command for o in _parse_choice_options(action_text)} != {str(i) for i in range(len(source))}):
            raise EnvironmentError("Liquid Memories commands disagree with public discard pile")
        valid = sorted(enumerate(source), key=lambda item: selection_key(item[1]))
        combat["selection_candidates"] = [copy.deepcopy(c) for _, c in valid]
        from .combat_selections import retrieval_choices
        try:
            groups = retrieval_choices(combat)
        except ValueError as error:
            raise EnvironmentError(str(error)) from error
        combat["selection_options"] = [list(g) for g in groups]
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i),
                " ".join(str(valid[j][0]) for j in group),
                "Retrieve " + ", ".join(valid[j][1]["id"] for j in group))
                for i, group in enumerate(groups)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "ARMAMENTS":
        # Commands index the full hand, while public choices are densely numbered.
        # Use the SAME mapping for scoring, model encoding and actual dispatch.
        options = _parse_choice_options(action_text)
        hand = combat.get("hand", ())
        candidates, seen = [], set()
        for option in options:
            if option.action.kind is not ActionKind.CHOOSE or not re.fullmatch(r"\d+", option.command):
                raise EnvironmentError("Armaments selection has an invalid native command")
            index = int(option.command)
            if index in seen or not 0 <= index < len(hand):
                raise EnvironmentError("Armaments selection references an invalid or duplicate hand index")
            selected = hand[index]
            if not isinstance(selected, Mapping) or not selected.get("id"):
                raise EnvironmentError("Armaments selection has no structured public card")
            seen.add(index)
            candidates.append(copy.deepcopy(selected))
        if not candidates:
            raise EnvironmentError("Armaments selection has no public candidates")
        combat["selection_candidates"] = candidates
        return options
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "FORETHOUGHT":
        from .forethought import bottom_choices
        try:
            groups = bottom_choices(combat)
        except ValueError as error:
            raise EnvironmentError(str(error)) from error
        combat["selection_options"] = [list(group) for group in groups]
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i),
                                 " ".join(map(str, group)) if group else "none",
                                 "Move hand indices to bottom in order " + str(list(group)))
                for i, group in enumerate(groups)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "EXHAUST_MANY":
        from .multi_exhaust import exhaust_subsets
        try:
            groups = exhaust_subsets(combat)
        except ValueError as error:
            raise EnvironmentError(str(error)) from error
        commands = {option.command for option in _parse_choice_options(action_text)}
        if commands != {str(i) for i in range(len(combat["hand"]))}:
            raise EnvironmentError("multi-exhaust console candidates disagree with public hand")
        combat["selection_options"] = [list(group) for group in groups]
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i),
                                 " ".join(map(str, reversed(group))) if group else "none",
                                 "Exhaust hand indices " + str(list(group)))
                for i, group in enumerate(groups)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task == "DISCOVERY":
        candidates = combat.get("selection_candidates", ())
        if not isinstance(candidates, list) or len(candidates) != 3 or not all(isinstance(c, Mapping) and c.get("id") for c in candidates):
            raise EnvironmentError("Discovery selection has no three structured public candidates")
        if {option.command for option in _parse_choice_options(action_text)} != {"0", "1", "2"}:
            raise EnvironmentError("Discovery commands disagree with public candidate indices")
        # Retain actual screen order. Never synthesize candidates from console text.
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i), str(i),
                                 f"Discover {card['id']}") for i, card in enumerate(candidates)]
    if str(combat.get("input_state", "")).upper() == "CARD_SELECT" and task in RETRIEVAL_TASKS | {"DUAL_WIELD"}:
        # Native commands index the source pile. Keep them adapter-private and
        # expose dense indices into a canonical public candidate list instead.
        is_copy = task == "DUAL_WIELD"
        step = None if is_copy else retrieval_step(task)
        source = combat.get("hand" if is_copy else "exhaust_pile" if task == "EXHUME" else "draw_pile", ())
        catalog = RuleCatalog.bundled()
        valid = [(i, card) for i, card in enumerate(source)
                 if isinstance(card, Mapping) and (copy_eligible(card, catalog) if is_copy else eligible(card, step, catalog))]
        if is_copy:
            count = re.search(r"Dual Wield: Choose a card to create ([12]) copies of in your hand\.", action_text)
            if count is None:
                raise EnvironmentError("Dual Wield selection has no reviewed public copy count")
            combat["selection_copy_count"] = int(count.group(1))
        commands = {option.command for option in _parse_choice_options(action_text)}
        expected = {str(i) for i, _ in valid}
        # Upstream console prints Exhume entries even though its search legality
        # excludes them. Never expose those invalid choices to a policy.
        allowed = {str(i) for i in range(len(source))} if task == "EXHUME" else expected
        if not expected <= commands or not commands <= allowed:
            raise EnvironmentError(f"{task} native commands disagree with public eligibility")
        valid.sort(key=lambda item: selection_key(item[1]))
        combat["selection_candidates"] = [copy.deepcopy(card) for _, card in valid]
        return [_SimulatorOption(Action(ActionKind.CHOOSE, choice_index=i), str(native_index),
                                 f"{'Copy' if is_copy else 'Retrieve'} {card.get('id')} (+{card.get('upgrades', 0)})")
                for i, (native_index, card) in enumerate(valid)]
    return _parse_choice_options(action_text)


def _parse_normal_combat_options(action_text: str) -> list[_SimulatorOption]:
    options: list[_SimulatorOption] = []
    for raw_line in action_text.splitlines():
        line = raw_line.strip()
        discard_match = re.match(r"^discard\s+(\d+)\s+\(([^)]*)\)$", line)
        if discard_match:
            index = int(discard_match.group(1))
            options.append(_SimulatorOption(Action(ActionKind.DISCARD_POTION, potion_index=index),
                            f"discard {index}", "Discard " + discard_match.group(2)))
            continue
        potion_match = re.match(r"^drink/discard\s+(\d+)\s+\(([^)]*)\)(.*)$", line)
        if potion_match:
            potion_index = int(potion_match.group(1))
            description = potion_match.group(2).strip()
            targets = [int(value) for value in re.findall(r"\d+", potion_match.group(3))]
            if targets:
                for target in targets:
                    options.append(
                        _SimulatorOption(
                            Action(
                                ActionKind.USE_POTION,
                                potion_index=potion_index,
                                target_index=target,
                            ),
                            f"drink {potion_index} {target}",
                            f"Use {description} on target {target}",
                        )
                    )
            else:
                options.append(
                    _SimulatorOption(
                        Action(ActionKind.USE_POTION, potion_index=potion_index),
                        f"drink {potion_index}",
                        f"Use {description}",
                    )
                )
            options.append(
                _SimulatorOption(
                    Action(ActionKind.DISCARD_POTION, potion_index=potion_index),
                    f"discard {potion_index}",
                    f"Discard {description}",
                )
            )
            continue

        if re.match(r"^end\s*:", line, re.IGNORECASE):
            options.append(_SimulatorOption(Action(ActionKind.END_TURN), "end", "End turn"))
            continue

        card_match = re.match(r"^(\d+)\s*:\s*(.*)$", line)
        if not card_match:
            continue
        card_index = int(card_match.group(1))
        description = card_match.group(2).strip()
        target_match = re.search(r",\s*((?:\d+[\s,]*)+)$", description)
        targets = (
            [int(value) for value in re.findall(r"\d+", target_match.group(1))]
            if target_match
            else []
        )
        if targets:
            for target in targets:
                options.append(
                    _SimulatorOption(
                        Action(
                            ActionKind.PLAY_CARD,
                            card_index=card_index,
                            target_index=target,
                        ),
                        f"{card_index} {target}",
                        f"{description} -> target {target}",
                    )
                )
        else:
            options.append(
                _SimulatorOption(
                    Action(ActionKind.PLAY_CARD, card_index=card_index),
                    str(card_index),
                    description,
                )
            )
    return _deduplicate_options(options)


def _parse_choice_options(action_text: str) -> list[_SimulatorOption]:
    parsed: list[tuple[str, str]] = []
    for raw_line in action_text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^([^:]+)\s*:\s*(.*)$", line)
        if not match:
            continue
        command = match.group(1).strip()
        description = match.group(2).strip() or command
        if re.fullmatch(r"\d+(?:\s+\d+)?", command):
            parsed.append((command, description))
            continue
        first_word = command.split()[0]
        if first_word in {
            "card",
            "relic",
            "potion",
            "gold",
            "maxhp",
            "emeraldKey",
            "sapphireKey",
            "remove",
            "skip",
            "proceed",
        }:
            parsed.append((command, description))

    options: list[_SimulatorOption] = []
    for choice_index, (command, description) in enumerate(parsed):
        kind = ActionKind.PROCEED if command in {"skip", "proceed"} else ActionKind.CHOOSE
        action = (
            Action(kind)
            if kind is ActionKind.PROCEED
            else Action(kind, choice_index=choice_index)
        )
        options.append(_SimulatorOption(action, command, description))
    return _deduplicate_options(options)


def _parse_reward_groups(
    action_text: str,
) -> tuple[list[_RewardGroup], tuple[str, str] | None]:
    grouped: dict[tuple[str, int], list[_SimulatorOption]] = {}
    descriptions: dict[tuple[str, int], str] = {}
    order: list[tuple[str, int]] = []
    proceed: tuple[str, str] | None = None

    for raw_line in action_text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^([^:]+)\s*:\s*(.*)$", line)
        if not match:
            continue
        command = match.group(1).strip()
        description = match.group(2).strip() or command
        if command in {"skip", "proceed"}:
            proceed = (command, description)
            continue

        parts = command.split()
        kind = parts[0]
        if kind == "card" and len(parts) == 3 and parts[1].isdigit():
            key = (kind, int(parts[1]))
            group_description = "card"
        elif kind == "maxhp" and len(parts) == 2 and parts[1].isdigit():
            key = ("card", int(parts[1]))
            group_description = "card"
        elif kind in {"gold", "relic", "potion"} and len(parts) == 2 and parts[1].isdigit():
            key = (kind, int(parts[1]))
            group_description = description
        elif kind in {"emeraldKey", "sapphireKey"} and len(parts) == 1:
            key = (kind, 0)
            group_description = description
        else:
            continue

        if key not in grouped:
            grouped[key] = []
            descriptions[key] = group_description
            order.append(key)
        grouped[key].append(
            _SimulatorOption(Action(ActionKind.CHOOSE, choice_index=0), command, description)
        )

    return (
        [
            _RewardGroup(key, descriptions[key], tuple(grouped[key]))
            for key in order
        ],
        proceed,
    )


def _deduplicate_options(options: list[_SimulatorOption]) -> list[_SimulatorOption]:
    result: list[_SimulatorOption] = []
    seen: set[Action] = set()
    for option in options:
        if option.action in seen:
            continue
        seen.add(option.action)
        result.append(option)
    return result


def _reward_components(
    before: Mapping[str, Any], after: Mapping[str, Any], terminated: bool
) -> dict[str, int | bool]:
    before_player = _mapping(before.get("player"))
    after_player = _mapping(after.get("player"))
    before_run = _mapping(before.get("run"))
    after_run = _mapping(after.get("run"))
    outcome = str(after_run.get("outcome", "UNDECIDED")).upper()
    return {
        "hp_delta": _delta(before_player, after_player, "hp"),
        "max_hp_delta": _delta(before_player, after_player, "max_hp"),
        "gold_delta": _delta(before_player, after_player, "gold"),
        "floor_delta": _delta(before_run, after_run, "floor"),
        "combat_ended": "combat" in before and "combat" not in after,
        "run_ended": terminated,
        "terminal": 1 if outcome == "PLAYER_VICTORY" else -1 if outcome == "PLAYER_LOSS" else 0,
    }


def _delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int:
    left = before.get(key)
    right = after.get(key)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return int(right - left)
    return 0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
