from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .identifiers import canonical_identifier


CARD_RECIPE_VERSION = "combat-card-recipes-v48"


class EffectPrimitive(str, Enum):
    GAIN_INTANGIBLE = "gain_per_packet_intangible_with_round_decay"
    REGISTER_DELAYED_DAMAGE = "register_independent_delayed_area_damage"
    SELECT_BOTTOM_FREE = "ordered_hand_to_bottom_positive_cost_one_use_free"
    GAIN_ARTIFACT = "gain_debuff_prevention_charges"
    APPLY_NO_BLOCK = "apply_artifact_blockable_card_block_restriction"
    EXHAUST_UP_TO_HAND = "exhaust_zero_to_limit_selected_hand_cards"
    RETRIEVE_RANDOM_ATTACKS = "move_random_draw_attacks_to_hand_overflow_discard"
    SHUFFLE_DISCARD_INTO_DRAW = "shuffle_discard_into_draw_if_nonempty"
    GENERATE_X_HAND = "generate_energy_count_random_cards_into_hand"
    GENERATE_CHOICE_HAND = "choose_random_candidate_into_hand"
    GENERATE_RANDOM_DRAW = "generate_random_cards_into_draw_pile"
    GENERATE_RANDOM_HAND = "generate_random_cards_into_hand"
    AUTOPLAY_TOP_CARD = "autoplay_top_draw_card"
    DYNAMIC_DAMAGE = "attack_from_public_card_state"
    GROW_PLAYED_DAMAGE = "grow_same_combat_card_damage_after_attack"
    MODIFY_HAND_COST = "modify_visible_hand_costs"
    REGISTER_CORRUPTION = "register_skill_free_play_and_exhaust"
    REGISTER_ATTACK_REPEAT = "register_queued_attack_repeat_until_turn_end"
    APPLY_NO_DRAW = "apply_artifact_blockable_no_draw_until_turn_end"
    UPGRADE_CARDS = "upgrade_combat_cards"
    ENEMY_STRENGTH_LOSS = "apply_enemy_strength_loss"
    GAIN_CURRENT_STRENGTH = "gain_current_signed_strength"
    REGISTER_EVENT_TRIGGER = "register_event_trigger"
    ON_HP_LOSS_COST_REDUCTION = "intrinsic_hp_loss_cost_reduction_active_piles"
    EXHAUST_HAND_GROUP = "exhaust_hand_group_with_payoff"
    ON_EXHAUST_ENERGY = "intrinsic_on_exhaust_gain_energy"
    REGISTER_TURN_TRIGGER = "register_persistent_turn_trigger"
    SELF_VULNERABLE = "apply_self_vulnerable"
    RETAIN_BLOCK = "retain_block_across_turns"
    CURRENT_BLOCK_GAIN = "gain_current_block_without_card_modifiers"
    END_TURN_BLOCK = "register_end_turn_non_card_block"
    TEMP_RETALIATION = "register_attack_retaliation_until_next_turn"
    HEAL = "heal_player"
    AREA_LIFESTEAL = "area_attack_heal_actual_hp_damage"
    FATAL_ATTACK = "attack_with_eligible_fatal_reward"
    ATTACK_SEQUENCE = "attack_sequence"
    DIRECT_DAMAGE = "direct_damage"
    AREA_DAMAGE = "area_damage"
    CURRENT_BLOCK_DAMAGE = "public_block_scaled_damage"
    GAIN_BLOCK = "gain_block"
    LOSE_HP = "lose_hp"
    GAIN_ENERGY = "gain_energy"
    GAIN_STRENGTH = "gain_strength"
    APPLY_LOSE_STRENGTH = "apply_lose_strength"
    DRAW_CARDS = "draw_cards"
    REGISTER_DRAW_TRIGGER = "register_persistent_draw_trigger"
    REGISTER_EXHAUST_TRIGGER = "register_persistent_exhaust_trigger"
    EXHAUST_RANDOM_HAND = "exhaust_random_hand"
    EXHAUST_CHOSEN_HAND = "exhaust_chosen_hand"
    GENERATE_CARD = "generate_card"
    MOVE_SELECTED_CARD = "move_selected_card"
    COPY_SELECTED_CARD = "copy_selected_card"
    APPLY_WEAK = "apply_weak"
    APPLY_VULNERABLE = "apply_vulnerable"


class CardTrait(str, Enum):
    PLAYABLE_CURSE = "naturally_playable_curse_with_candle_hp_loss"
    UNPLAYABLE_STATUS = "unplayable_status_except_medical_kit"
    UNPLAYABLE_CURSE = "unplayable_curse_except_blue_candle"
    INNATE = "innate"
    BASE_EXHAUST = "exhaust_unupgraded_only"
    UPGRADE_INNATE = "upgrade_innate"
    ETHEREAL = "ethereal"
    BASE_ETHEREAL = "ethereal_unupgraded_only"
    EXHAUST = "exhaust"
    PUBLIC_HAND_PLAYABILITY = "public_hand_playability"
    UPGRADE_COST_CHANGE = "upgrade_cost_change"
    TARGETED_ENEMY = "targeted_enemy"


class EffectTarget(str, Enum):
    TARGET_ENEMY = "target_enemy"
    RANDOM_ENEMY = "random_living_enemy_each_hit"
    ALL_ENEMIES = "all_enemies"
    PLAYER = "player"


class AmountSource(str, Enum):
    FIXED = "fixed"
    CURRENT_BLOCK = "current_block"
    CURRENT_STRENGTH = "current_signed_strength"


class DamageScaling(str, Enum):
    PERSISTENT_CARD_BASE_DAMAGE = "public_persistent_card_base_damage"
    DRAW_PILE_COUNT = "current_public_draw_pile_count"
    STRIKE_COUNT = "strike_tags_in_hand_draw_discard_including_self"
    CARD_BASE_DAMAGE = "current_combat_card_base_damage"
    UPGRADE_COUNT = "repeatable_upgrade_count"


@dataclass(frozen=True, slots=True)
class DynamicDamage:
    source: DamageScaling

    def __post_init__(self) -> None:
        if not isinstance(self.source, DamageScaling):
            raise ValueError("dynamic damage requires a declared source")

    @property
    def signature(self) -> str:
        return self.source.value + ":before_strength_weak_vulnerable"


class UpgradeScope(str, Enum):
    CHOSEN_HAND = "one_eligible_hand_card_auto_if_single"
    ALL_HAND = "all_eligible_hand_cards"
    ALL_COMBAT_PILES = "eligible_hand_draw_discard_exhaust"


class CostSelection(str, Enum):
    RANDOM_POSITIVE = "random_positive_turn_cost_else_positive_combat_cost"
    ALL_HAND = "all_hand_independent_cost_caps"


class CostDuration(str, Enum):
    TURN = "turn"
    COMBAT = "combat"


@dataclass(frozen=True, slots=True)
class HandCostChange:
    selection: CostSelection
    base_duration: CostDuration
    upgraded_duration: CostDuration

    def __post_init__(self) -> None:
        if not isinstance(self.selection, CostSelection) or any(
            not isinstance(d, CostDuration) for d in (self.base_duration, self.upgraded_duration)
        ):
            raise ValueError("hand cost change requires declared selection/duration enums")

    def duration(self, upgraded: bool) -> CostDuration:
        return self.upgraded_duration if upgraded else self.base_duration

    @property
    def signature(self) -> str:
        return f"{self.selection.value}:{self.base_duration.value}/{self.upgraded_duration.value}:exclude_played"


@dataclass(frozen=True, slots=True)
class CombatUpgrade:
    """One upgrade per eligible identity; never the master deck or card in play.

    Preserve mutable turn-cost differences through the card's upgrade semantics.
    Repeatable upgrades use their count, not a generic boolean toggle.
    """
    base_scope: UpgradeScope
    upgraded_scope: UpgradeScope

    def __post_init__(self) -> None:
        if not isinstance(self.base_scope, UpgradeScope) or not isinstance(self.upgraded_scope, UpgradeScope):
            raise ValueError("combat upgrade requires declared scope enums")

    def scope(self, upgraded: bool) -> UpgradeScope:
        return self.upgraded_scope if upgraded else self.base_scope

    @property
    def signature(self) -> str:
        return f"{self.base_scope.value}/{self.upgraded_scope.value}:once_per_eligible_identity:combat_only"


class StrengthRestoration(str, Enum):
    NONE = "none"
    ENEMY_TURN_END = "enemy_turn_end_if_no_artifact_at_play"


@dataclass(frozen=True, slots=True)
class StrengthReduction:
    """Positive loss amount is applied as a negative, Artifact-blockable delta.

    Optional restoration eligibility is snapshotted before queued reduction, not
    after Artifact consumption. Restore the full scheduled amount at enemy turn
    end, even if the original reduction hit the -999 Strength floor.
    """
    restoration: StrengthRestoration = StrengthRestoration.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.restoration, StrengthRestoration):
            raise ValueError("Strength restoration must be declared")

    @property
    def signature(self) -> str:
        return f"artifact_blockable:clamp=-999/999:restore={self.restoration.value}"


class PublicCondition(str, Enum):
    HAND_HAS_NO_ATTACK = "hand_has_no_attack_at_queued_resolution"
    TARGET_HAS_VULNERABLE = "target_has_vulnerable"
    TARGET_ATTACK_INTENT = "target_attack_intent"


class GeneratedCardSource(str, Enum):
    PLAYED_COPY = "played_copy"
    FIXED_BASE = "fixed_base"


class GeneratedCardDestination(str, Enum):
    HAND = "hand_with_discard_overflow"
    DISCARD_PILE = "discard_pile"
    DRAW_PILE_RANDOM = "draw_pile_random_position"


class SelectedCardSource(str, Enum):
    HAND = "hand_after_preceding_draws"
    DISCARD_PILE = "discard_pile"
    DRAW_PILE = "public_draw_pile_multiset"
    EXHAUST_PILE = "exhaust_pile"


class SelectedCardDestination(str, Enum):
    DRAW_PILE_TOP = "draw_pile_top"
    HAND_OVERFLOW_DISCARD = "hand_without_draw_callbacks_overflow_discard"
    HAND_UNLESS_FULL = "hand_without_draw_callbacks_no_op_when_full"


class SelectedCardFilter(str, Enum):
    NON_EXHUME = "exclude_all_exhume_instances"
    SKILL = "skill_only"
    ATTACK = "attack_only"


class CopyIdentityRule(str, Enum):
    ORIGINAL_ONLY_IF_AUTOMATIC = "keep_original_if_singleton_replace_selected_if_manual"


class AttackRepeatContract(str, Enum):
    SAME_ID_PURGED_AUTOPLAY = "one_extra_attack_play:same_identity:autoplay:no_recursive_charge:original_energy_on_use:purge_copy:end_turn_expiry"


class TopCardAutoplayContract(str, Enum):
    HAVOC = "draw_top:shuffle_discard_if_empty:random_live_target:free:queued_energy_on_use:not_draw:exhaust_once_except_power:rejected_play_still_settles"
    MAYHEM = "per_stack_wrapper:target_at_wrapper:append_after_queued_draws:draw_top:shuffle_discard_if_empty:fifo_card_queue:free:queued_energy_on_use:not_draw:no_forced_exhaust:rejected_play_still_settles"


class DiscoveryGeneration(str, Enum):
    IRONCLAD_THREE = "ironclad_combat_pool_no_healing:three_distinct:sample_at_action:public_choice_one:base_copy:blood_for_blood_prior_damage:turn_zero_except_x:fresh_identity:hand_overflow_discard:not_draw"


class RandomDrawGeneration(str, Enum):
    IRONCLAD_SKILL = "ironclad_skill_pool_no_healing"
    IRONCLAD_ATTACK = "ironclad_attack_pool_no_healing"

    @property
    def signature(self) -> str:
        return self.value + ":base_copy:positive_cost_to_combat_zero:x_unchanged:sample_at_use:with_replacement:fresh_identity:ordered_random_insert_below_existing_top:not_draw"


class RandomHandGeneration(str, Enum):
    IRONCLAD_ATTACK_TURN_ZERO = "ironclad_attack_pool_no_healing:base_copy:blood_for_blood_prior_damage:turn_zero_except_x"
    COLORLESS_BASE = "colorless_pool_no_healing:base_copy:normal_cost"
    COLORLESS_X_TURN_ZERO = "colorless_pool_no_healing:upgrade_copy_if_generator_upgraded:turn_zero_except_x:energy_on_use+chemical_x(2):spend_all_unless_free"
    COLORLESS_TURN_START = "colorless_pool_no_healing:base_copy:normal_cost:per_power_stack:before_normal_draw"

    @property
    def signature(self) -> str:
        timing = ("action" if self is self.COLORLESS_X_TURN_ZERO else
                  "turn_trigger" if self is self.COLORLESS_TURN_START else "use")
        return self.value + f":sample_at_{timing}:with_replacement:fresh_identity:ordered_hand_overflow_discard:not_draw"

    def energy_count(self, energy: int, *, chemical_x: bool = False) -> int:
        if self is not self.COLORLESS_X_TURN_ZERO:
            raise ValueError("only X generation takes an energy count")
        if type(energy) is not int or energy < 0:
            raise ValueError("public energy must be a non-negative integer")
        return energy + (2 if chemical_x else 0)


@dataclass(frozen=True, slots=True)
class SelectedCardCopy:
    """Stat-equivalent hand copies; overflow discards, not draw/play callbacks."""
    card_types: tuple[str, ...] = ("ATTACK", "POWER")
    identity: CopyIdentityRule = CopyIdentityRule.ORIGINAL_ONLY_IF_AUTOMATIC

    def __post_init__(self) -> None:
        if tuple(self.card_types) != ("ATTACK", "POWER") or self.identity is not CopyIdentityRule.ORIGINAL_ONLY_IF_AUTOMATIC:
            raise ValueError("unreviewed selected-copy filter/identity contract")
        object.__setattr__(self, "card_types", tuple(self.card_types))

    @property
    def signature(self) -> str:
        return f"hand:{','.join(self.card_types)}:stat_equivalent:{self.identity.value}:hand_overflow_discard"


class DrawTriggerEffect(str, Enum):
    DRAW = "draw_cards"
    NON_ATTACK_AREA_DAMAGE = "non_attack_area_damage"


class CombatEvent(str, Enum):
    FIFTH_CARD_PLAYED = "every_fifth_card_play_reset_each_turn"
    ENEMY_DEBUFF_APPLIED = "successful_enemy_debuff_packet_excluding_shackled"
    SELF_HP_LOST = "positive_self_sourced_hp_loss"
    ATTACK_PLAYED = "attack_card_played_once"
    BLOCK_GAINED = "positive_block_gain_including_at_cap"


class EventEffect(str, Enum):
    AREA_DAMAGE = "all_living_enemies_non_attack_damage"
    ENEMY_DAMAGE = "debuff_target_non_attack_damage"
    STRENGTH = "strength"
    NON_CARD_BLOCK = "non_card_block"
    RANDOM_DAMAGE = "random_living_enemy_non_attack_damage"


class TriggerDuration(str, Enum):
    COMBAT = "combat"
    TURN = "until_end_of_turn"


@dataclass(frozen=True, slots=True)
class EventTrigger:
    """Registration only; each event queues one callback, not one per HP/hit.

    Event amounts stack. Positive block includes an attempted gain at the 999
    cap; prevented HP loss is not an event. Native queues own callback ordering.
    """
    power_id: str
    event: CombatEvent
    effect: EventEffect
    duration: TriggerDuration = TriggerDuration.COMBAT

    def __post_init__(self) -> None:
        if not self.power_id or canonical_identifier(self.power_id, "card") != self.power_id:
            raise ValueError("event trigger needs a canonical power identifier")
        if not isinstance(self.event, CombatEvent) or not isinstance(self.effect, EventEffect):
            raise ValueError("event trigger needs declared event/effect enums")
        if not isinstance(self.duration, TriggerDuration):
            raise ValueError("event trigger needs a declared duration")

    @property
    def signature(self) -> str:
        return f"{self.power_id}:{self.event.value}:{self.effect.value}:{self.duration.value}"


class TurnTiming(str, Enum):
    START = "start_of_turn"
    POST_DRAW = "start_of_turn_post_draw"
    END = "end_of_player_turn"


@dataclass(frozen=True, slots=True)
class DelayedDamage:
    """Independent one-shot non-attack area packet, not a persistent stack.

    The playing turn counts; source-card removal does not cancel the instance.
    Concrete handlers own queue timing, block and per-enemy damage callbacks.
    """
    turns: int = 3
    timing: TurnTiming = TurnTiming.END

    def __post_init__(self) -> None:
        if type(self.turns) is not int or self.turns < 1 or not isinstance(self.timing, TurnTiming):
            raise ValueError("delayed damage requires a positive integer countdown and declared phase")

    @property
    def signature(self) -> str:
        return (f"{self.turns}:{self.timing.value}:independent:one_shot_non_attack_area:"
                "counts_played_turn:survives_source_removal")


class TurnEffectKind(str, Enum):
    AUTOPLAY_TOP = "deferred_top_card_autoplay"
    GENERATE_RANDOM_HAND = "generate_random_cards_into_hand"
    DRAW = "draw_cards"
    SELF_HP_LOSS = "self_hp_loss"
    NON_ATTACK_AREA_DAMAGE = "non_attack_area_damage"
    STRENGTH = "gain_strength"
    ENERGY = "gain_energy"


@dataclass(frozen=True, slots=True)
class TurnEffect:
    kind: TurnEffectKind
    base_amount: int
    upgraded_amount: int
    random_hand_generation: RandomHandGeneration | None = None
    top_card_autoplay: TopCardAutoplayContract | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TurnEffectKind):
            raise ValueError("turn effect kind must be declared")
        if self.kind is TurnEffectKind.AUTOPLAY_TOP:
            if self.top_card_autoplay is not TopCardAutoplayContract.MAYHEM or (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("turn autoplay requires the reviewed one-card Mayhem contract")
        elif self.top_card_autoplay is not None:
            raise ValueError("only turn autoplay may carry an autoplay contract")
        if any(type(n) is not int or n < 1 for n in (self.base_amount, self.upgraded_amount)):
            raise ValueError("turn effect amounts must be positive integers")
        if self.kind is TurnEffectKind.GENERATE_RANDOM_HAND:
            if self.random_hand_generation is not RandomHandGeneration.COLORLESS_TURN_START or (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("turn generation requires the reviewed colorless one-card contract")
        elif self.random_hand_generation is not None:
            raise ValueError("only turn generation may carry a generation contract")

    def amount(self, upgraded: bool) -> int:
        return self.upgraded_amount if upgraded else self.base_amount

    @property
    def signature(self) -> str:
        result = f"{self.kind.value}:{self.base_amount}/{self.upgraded_amount}"
        return (result + (f":{self.random_hand_generation.signature}" if self.random_hand_generation else "")
                + (f":{self.top_card_autoplay.value}" if self.top_card_autoplay else ""))


@dataclass(frozen=True, slots=True)
class TurnTrigger:
    """Ordered callback payload per application, additive across power stacks.

    Entries queue in declared order; they are not immediate play effects. Each
    stacked callback combines its matching amounts (Combust loses HP once, not
    once per copy). Admission of a timing/payload combination requires evidence.
    """
    power_id: str
    timing: TurnTiming
    effects: tuple[TurnEffect, ...]

    def __post_init__(self) -> None:
        if not self.power_id or canonical_identifier(self.power_id, "card") != self.power_id:
            raise ValueError("turn trigger needs a canonical power identifier")
        if not isinstance(self.timing, TurnTiming):
            raise ValueError("turn timing must be declared")
        if not self.effects or any(not isinstance(e, TurnEffect) for e in self.effects):
            raise ValueError("turn trigger needs an ordered effect payload")
        object.__setattr__(self, "effects", tuple(self.effects))

    @property
    def signature(self) -> str:
        return f"{self.power_id}:{self.timing.value}:" + ">".join(e.signature for e in self.effects)


@dataclass(frozen=True, slots=True)
class DrawTrigger:
    power_id: str
    card_types: tuple[str, ...]
    effect: DrawTriggerEffect

    def __post_init__(self) -> None:
        if not self.power_id or canonical_identifier(self.power_id, "card") != self.power_id:
            raise ValueError("draw trigger needs a canonical power identifier")
        if (not self.card_types or len(set(self.card_types)) != len(self.card_types)
                or any(t not in {"STATUS", "CURSE"} for t in self.card_types)):
            raise ValueError("draw trigger needs distinct Status/Curse filters")
        if not isinstance(self.effect, DrawTriggerEffect):
            raise ValueError("draw trigger effect must be declared")
        object.__setattr__(self, "card_types", tuple(self.card_types))

    @property
    def signature(self) -> str:
        return f"{self.power_id}:{','.join(self.card_types)}:{self.effect.value}"


class ExhaustTriggerEffect(str, Enum):
    DRAW = "draw_cards"
    NON_CARD_BLOCK = "gain_non_card_block"


class HandExhaustFilter(str, Enum):
    NON_ATTACK = "non_attack_identity_snapshot"
    ALL = "all_hand_count_snapshot_random_exhausts"


class HandExhaustPayoff(str, Enum):
    NONE = "none"
    BLOCK = "card_block_per_selected_card"
    ATTACK = "snapshot_attack_per_selected_card"


@dataclass(frozen=True, slots=True)
class HandExhaust:
    """Select/count after the played card leaves hand, before queued callbacks.

    NON_ATTACK schedules fixed identities in reverse hand order. ALL schedules
    a fixed number of one-card random exhaust actions (not an until-empty loop).
    Payoff amounts are snapshotted at play and resolve after those exhausts.
    On-exhaust callbacks retain their own queue priority; this is not an atomic
    promise that all callback effects resolve before the payoff.
    """
    filter: HandExhaustFilter
    payoff: HandExhaustPayoff

    def __post_init__(self) -> None:
        if not isinstance(self.filter, HandExhaustFilter) or not isinstance(self.payoff, HandExhaustPayoff):
            raise ValueError("hand exhaust filter and payoff must be declared")

    @property
    def signature(self) -> str:
        return f"{self.filter.value}:{self.payoff.value}:exclude_played_card"


@dataclass(frozen=True, slots=True)
class ExhaustTrigger:
    power_id: str
    effect: ExhaustTriggerEffect

    def __post_init__(self) -> None:
        if not self.power_id or canonical_identifier(self.power_id, "card") != self.power_id:
            raise ValueError("exhaust trigger needs a canonical power identifier")
        if not isinstance(self.effect, ExhaustTriggerEffect):
            raise ValueError("exhaust trigger effect must be declared")

    @property
    def signature(self) -> str:
        return f"{self.power_id}:{self.effect.value}"


class FatalRewardKind(str, Enum):
    CARD_DAMAGE = "same_identity_combat_and_master_card_damage"
    MAX_HP = "max_hp"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class FatalReward:
    kind: FatalRewardKind
    base_amount: int
    upgraded_amount: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FatalRewardKind):
            raise ValueError("fatal reward kind must be declared")
        if any(type(n) is not int or n < 1 for n in (self.base_amount, self.upgraded_amount)):
            raise ValueError("fatal reward amounts must be positive integers")

    def amount(self, upgraded: bool) -> int:
        return self.upgraded_amount if upgraded else self.base_amount

    @property
    def signature(self) -> str:
        return f"{self.kind.value}:{self.base_amount}/{self.upgraded_amount}:eligible_non_minion_kill"


class AttackCountSource(str, Enum):
    FIXED = "fixed"
    ENERGY_ON_USE = "energy_on_use_with_chemical_x"


class AttackDamageTiming(str, Enum):
    SNAPSHOT = "snapshot"
    EACH_HIT = "recalculate_each_hit"


@dataclass(frozen=True, slots=True)
class AttackSequence:
    """Per-hit damage with explicit count source and damage timing.

    ENERGY_ON_USE represents ordinary manual X-cost play: current public energy
    plus Chemical X's two effects; spend all energy unless free-to-play. Deferred
    autoplay/duplicate energy overrides are not modeled by the public scorer.
    """

    base_hits: int = 1
    upgraded_hits: int = 1
    base_strength_multiplier: int = 1
    upgraded_strength_multiplier: int = 1
    count_source: AttackCountSource = AttackCountSource.FIXED
    damage_timing: AttackDamageTiming = AttackDamageTiming.SNAPSHOT

    def __post_init__(self) -> None:
        if not isinstance(self.count_source, AttackCountSource) or not isinstance(
            self.damage_timing, AttackDamageTiming
        ):
            raise ValueError("attack count source and damage timing must be declared enums")
        if any(type(value) is not int or value < 1 for value in (
            self.base_hits, self.upgraded_hits,
            self.base_strength_multiplier, self.upgraded_strength_multiplier,
        )):
            raise ValueError("attack sequence counts and multipliers must be positive integers")
        if self.count_source is AttackCountSource.ENERGY_ON_USE and (
            self.base_hits != 1 or self.upgraded_hits != 1
        ):
            raise ValueError("energy-count attacks cannot also declare fixed hit counts")

    def hits(self, upgraded: bool) -> int:
        if self.count_source is not AttackCountSource.FIXED:
            raise ValueError("energy-count attack needs public energy")
        return self.upgraded_hits if upgraded else self.base_hits

    def resolved_hits(self, upgraded: bool, *, energy: int = 0, chemical_x: bool = False) -> int:
        if self.count_source is AttackCountSource.FIXED:
            return self.hits(upgraded)
        if type(energy) is not int or energy < 0:
            raise ValueError("public energy must be a non-negative integer")
        return energy + (2 if chemical_x else 0)

    def strength_multiplier(self, upgraded: bool) -> int:
        return self.upgraded_strength_multiplier if upgraded else self.base_strength_multiplier

    @property
    def signature(self) -> str:
        count = (f"{self.base_hits}/{self.upgraded_hits}" if self.count_source is AttackCountSource.FIXED
                 else "energy_on_use+chemical_x(2);spend_all_unless_free")
        return (f"hits={count}:strength={self.base_strength_multiplier}/"
                f"{self.upgraded_strength_multiplier}:{self.damage_timing.value}")


@dataclass(frozen=True, slots=True)
class EffectStep:
    primitive: EffectPrimitive
    target: EffectTarget
    base_amount: int | None = None
    upgraded_amount: int | None = None
    amount_source: AmountSource = AmountSource.FIXED
    condition: PublicCondition | None = None
    generated_card_source: GeneratedCardSource | None = None
    generated_card_id: str | None = None
    generated_card_destination: GeneratedCardDestination | None = None
    selected_card_source: SelectedCardSource | None = None
    selected_card_destination: SelectedCardDestination | None = None
    selected_card_filter: SelectedCardFilter | None = None
    selected_card_copy: SelectedCardCopy | None = None
    attack_repeat: AttackRepeatContract | None = None
    top_card_autoplay: TopCardAutoplayContract | None = None
    random_hand_generation: RandomHandGeneration | None = None
    random_draw_generation: RandomDrawGeneration | None = None
    discovery_generation: DiscoveryGeneration | None = None
    draw_trigger: DrawTrigger | None = None
    exhaust_trigger: ExhaustTrigger | None = None
    attack_sequence: AttackSequence | None = None
    fatal_reward: FatalReward | None = None
    turn_trigger: TurnTrigger | None = None
    delayed_damage: DelayedDamage | None = None
    hand_exhaust: HandExhaust | None = None
    event_trigger: EventTrigger | None = None
    strength_reduction: StrengthReduction | None = None
    combat_upgrade: CombatUpgrade | None = None
    hand_cost: HandCostChange | None = None
    dynamic_damage: DynamicDamage | None = None
    upgraded_target: EffectTarget | None = None

    def __post_init__(self) -> None:
        if self.upgraded_target is not None and not (
            self.primitive in {EffectPrimitive.APPLY_WEAK, EffectPrimitive.APPLY_VULNERABLE}
            and self.target is EffectTarget.TARGET_ENEMY
            and self.upgraded_target is EffectTarget.ALL_ENEMIES
            and self.condition is None
        ):
            raise ValueError("unreviewed upgrade-dependent target scope")
        if self.primitive is EffectPrimitive.SHUFFLE_DISCARD_INTO_DRAW and (
            (self.base_amount, self.upgraded_amount) != (1, 1) or self.condition is not None
        ):
            raise ValueError("shuffle step requires one unconditional discard merge and draw-pile shuffle")
        if self.condition is PublicCondition.HAND_HAS_NO_ATTACK and self.primitive is not EffectPrimitive.DRAW_CARDS:
            raise ValueError("no-attack hand condition is reviewed only for queued draw")
        expected_targets = {
            EffectPrimitive.GAIN_INTANGIBLE: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_DELAYED_DAMAGE: {EffectTarget.PLAYER},
            EffectPrimitive.SELECT_BOTTOM_FREE: {EffectTarget.PLAYER},
            EffectPrimitive.EXHAUST_UP_TO_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.RETRIEVE_RANDOM_ATTACKS: {EffectTarget.PLAYER},
            EffectPrimitive.SHUFFLE_DISCARD_INTO_DRAW: {EffectTarget.PLAYER},
            EffectPrimitive.GENERATE_X_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.AUTOPLAY_TOP_CARD: {EffectTarget.PLAYER},
            EffectPrimitive.DYNAMIC_DAMAGE: {EffectTarget.TARGET_ENEMY},
            EffectPrimitive.GROW_PLAYED_DAMAGE: {EffectTarget.PLAYER},
            EffectPrimitive.MODIFY_HAND_COST: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_CORRUPTION: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_ATTACK_REPEAT: {EffectTarget.PLAYER},
            EffectPrimitive.APPLY_NO_DRAW: {EffectTarget.PLAYER},
            EffectPrimitive.GAIN_ARTIFACT: {EffectTarget.PLAYER},
            EffectPrimitive.APPLY_NO_BLOCK: {EffectTarget.PLAYER},
            EffectPrimitive.UPGRADE_CARDS: {EffectTarget.PLAYER},
            EffectPrimitive.ENEMY_STRENGTH_LOSS: {EffectTarget.TARGET_ENEMY},
            EffectPrimitive.GAIN_CURRENT_STRENGTH: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_EVENT_TRIGGER: {EffectTarget.PLAYER},
            EffectPrimitive.ON_HP_LOSS_COST_REDUCTION: {EffectTarget.PLAYER},
            EffectPrimitive.EXHAUST_HAND_GROUP: {EffectTarget.PLAYER, EffectTarget.TARGET_ENEMY},
            EffectPrimitive.ON_EXHAUST_ENERGY: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_TURN_TRIGGER: {EffectTarget.PLAYER},
            EffectPrimitive.SELF_VULNERABLE: {EffectTarget.PLAYER},
            EffectPrimitive.RETAIN_BLOCK: {EffectTarget.PLAYER},
            EffectPrimitive.CURRENT_BLOCK_GAIN: {EffectTarget.PLAYER},
            EffectPrimitive.END_TURN_BLOCK: {EffectTarget.PLAYER},
            EffectPrimitive.TEMP_RETALIATION: {EffectTarget.PLAYER},
            EffectPrimitive.HEAL: {EffectTarget.PLAYER},
            EffectPrimitive.AREA_LIFESTEAL: {EffectTarget.ALL_ENEMIES},
            EffectPrimitive.FATAL_ATTACK: {EffectTarget.TARGET_ENEMY},
            EffectPrimitive.ATTACK_SEQUENCE: {
                EffectTarget.TARGET_ENEMY, EffectTarget.RANDOM_ENEMY, EffectTarget.ALL_ENEMIES,
            },
            EffectPrimitive.DIRECT_DAMAGE: {EffectTarget.TARGET_ENEMY},
            EffectPrimitive.AREA_DAMAGE: {EffectTarget.ALL_ENEMIES},
            EffectPrimitive.CURRENT_BLOCK_DAMAGE: {EffectTarget.TARGET_ENEMY},
            EffectPrimitive.GAIN_BLOCK: {EffectTarget.PLAYER},
            EffectPrimitive.LOSE_HP: {EffectTarget.PLAYER},
            EffectPrimitive.GAIN_ENERGY: {EffectTarget.PLAYER},
            EffectPrimitive.GAIN_STRENGTH: {EffectTarget.PLAYER},
            EffectPrimitive.APPLY_LOSE_STRENGTH: {EffectTarget.PLAYER},
            EffectPrimitive.DRAW_CARDS: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_DRAW_TRIGGER: {EffectTarget.PLAYER},
            EffectPrimitive.REGISTER_EXHAUST_TRIGGER: {EffectTarget.PLAYER},
            EffectPrimitive.EXHAUST_RANDOM_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.EXHAUST_CHOSEN_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.GENERATE_CARD: {EffectTarget.PLAYER},
            EffectPrimitive.GENERATE_RANDOM_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.GENERATE_RANDOM_DRAW: {EffectTarget.PLAYER},
            EffectPrimitive.GENERATE_CHOICE_HAND: {EffectTarget.PLAYER},
            EffectPrimitive.MOVE_SELECTED_CARD: {EffectTarget.PLAYER},
            EffectPrimitive.COPY_SELECTED_CARD: {EffectTarget.PLAYER},
            EffectPrimitive.APPLY_WEAK: {
                EffectTarget.TARGET_ENEMY, EffectTarget.ALL_ENEMIES,
            },
            EffectPrimitive.APPLY_VULNERABLE: {
                EffectTarget.TARGET_ENEMY, EffectTarget.ALL_ENEMIES,
            },
        }
        if self.target not in expected_targets[self.primitive]:
            raise ValueError(
                f"invalid target {self.target.value} for {self.primitive.value}"
            )
        if self.primitive is EffectPrimitive.DYNAMIC_DAMAGE:
            if not isinstance(self.dynamic_damage, DynamicDamage):
                raise ValueError("dynamic damage must declare a payload")
            if (self.dynamic_damage.source, self.base_amount, self.upgraded_amount) not in {
                (DamageScaling.STRIKE_COUNT, 6, 6),
                (DamageScaling.CARD_BASE_DAMAGE, 8, 8),
                (DamageScaling.UPGRADE_COUNT, 12, 16),
                (DamageScaling.DRAW_PILE_COUNT, 0, 0),
            }:
                raise ValueError("unreviewed dynamic damage combination")
        elif self.primitive is EffectPrimitive.FATAL_ATTACK and self.dynamic_damage is not None:
            if (self.dynamic_damage.source is not DamageScaling.PERSISTENT_CARD_BASE_DAMAGE
                    or self.fatal_reward is None or self.fatal_reward.kind is not FatalRewardKind.CARD_DAMAGE
                    or (self.base_amount, self.upgraded_amount) != (15, 15)):
                raise ValueError("unreviewed persistent fatal damage contract")
        elif self.dynamic_damage is not None:
            raise ValueError("only dynamic damage may carry a scaling payload")
        if self.primitive in {EffectPrimitive.EXHAUST_UP_TO_HAND, EffectPrimitive.RETRIEVE_RANDOM_ATTACKS}:
            expected = (3, 5) if self.primitive is EffectPrimitive.EXHAUST_UP_TO_HAND else (3, 4)
            if (self.base_amount, self.upgraded_amount) != expected or self.condition is not None:
                raise ValueError("unreviewed bounded exhaust/retrieval contract")
        if self.primitive is EffectPrimitive.GROW_PLAYED_DAMAGE and (self.base_amount, self.upgraded_amount) != (5, 8):
            raise ValueError("unreviewed per-card damage growth")
        if self.primitive is EffectPrimitive.MODIFY_HAND_COST:
            if not isinstance(self.hand_cost, HandCostChange):
                raise ValueError("hand cost change must declare a payload")
            if (self.hand_cost.selection, self.hand_cost.base_duration, self.hand_cost.upgraded_duration,
                self.base_amount, self.upgraded_amount) not in {
                (CostSelection.RANDOM_POSITIVE, CostDuration.COMBAT, CostDuration.COMBAT, 0, 0),
                (CostSelection.ALL_HAND, CostDuration.TURN, CostDuration.COMBAT, 1, 1),
            }:
                raise ValueError("unreviewed hand cost combination")
        elif self.hand_cost is not None:
            raise ValueError("only hand cost steps may carry cost payloads")
        if self.primitive is EffectPrimitive.AUTOPLAY_TOP_CARD:
            if self.top_card_autoplay is not TopCardAutoplayContract.HAVOC or (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("top-card autoplay needs the reviewed single-card lifecycle contract")
        elif self.top_card_autoplay is not None:
            raise ValueError("only top-card autoplay may declare its queue contract")
        if self.primitive is EffectPrimitive.GENERATE_CHOICE_HAND:
            if self.discovery_generation is not DiscoveryGeneration.IRONCLAD_THREE or (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("Discovery requires three distinct Ironclad candidates and one selected copy")
        elif self.discovery_generation is not None:
            raise ValueError("only generated choice may declare a Discovery contract")
        if self.primitive is EffectPrimitive.GENERATE_RANDOM_DRAW:
            if not isinstance(self.random_draw_generation, RandomDrawGeneration) or (self.base_amount, self.upgraded_amount) != (3, 5):
                raise ValueError("random draw generation requires a reviewed pool/count/cost contract")
        elif self.random_draw_generation is not None:
            raise ValueError("only random draw generation may declare its pool contract")
        if self.primitive is EffectPrimitive.GENERATE_X_HAND:
            if self.random_hand_generation is not RandomHandGeneration.COLORLESS_X_TURN_ZERO or (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("X generation requires the reviewed energy/count/upgrade contract")
        elif self.primitive is EffectPrimitive.GENERATE_RANDOM_HAND:
            if not isinstance(self.random_hand_generation, RandomHandGeneration) or (
                self.random_hand_generation, self.base_amount, self.upgraded_amount
            ) not in {
                (RandomHandGeneration.IRONCLAD_ATTACK_TURN_ZERO, 1, 1),
                (RandomHandGeneration.COLORLESS_BASE, 1, 2),
            }:
                raise ValueError("random generation requires a reviewed pool/count/cost contract")
        elif self.random_hand_generation is not None:
            raise ValueError("only random generation may declare a random pool contract")
        if self.primitive is EffectPrimitive.REGISTER_ATTACK_REPEAT:
            if self.attack_repeat is not AttackRepeatContract.SAME_ID_PURGED_AUTOPLAY or (self.base_amount, self.upgraded_amount) != (1, 2):
                raise ValueError("attack-repeat registration needs reviewed one/two charges and queued contract")
        elif self.attack_repeat is not None:
            raise ValueError("only attack-repeat registration may declare repeat contract")
        if self.primitive is EffectPrimitive.REGISTER_CORRUPTION and (self.base_amount, self.upgraded_amount) != (1, 1):
            raise ValueError("Corruption is a non-stacking registration")
        if self.primitive is EffectPrimitive.APPLY_NO_DRAW:
            if (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("No Draw applies one Artifact-blockable turn restriction")
        if self.primitive is EffectPrimitive.GAIN_ARTIFACT and (self.base_amount, self.upgraded_amount) != (1, 2):
            raise ValueError("unreviewed Artifact charge contract")
        if self.primitive is EffectPrimitive.SELECT_BOTTOM_FREE and (
            (self.base_amount, self.upgraded_amount) != (1, 10) or self.target is not EffectTarget.PLAYER
        ):
            raise ValueError("bottom/free selection requires base one or upgraded zero-to-ten ordered cards")
        if self.primitive is EffectPrimitive.APPLY_NO_BLOCK and (self.base_amount, self.upgraded_amount) != (2, 2):
            raise ValueError("unreviewed No Block duration; two end-of-round decrements")
        if self.primitive is EffectPrimitive.UPGRADE_CARDS:
            if not isinstance(self.combat_upgrade, CombatUpgrade):
                raise ValueError("combat upgrade must declare scope")
            if (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("combat upgrade applies once per eligible card")
        elif self.combat_upgrade is not None:
            raise ValueError("only combat upgrade steps may declare upgrade scope")
        if self.primitive is EffectPrimitive.ENEMY_STRENGTH_LOSS:
            if not isinstance(self.strength_reduction, StrengthReduction):
                raise ValueError("Strength loss must declare restoration semantics")
        elif self.strength_reduction is not None:
            raise ValueError("only enemy Strength loss may declare restoration")
        if self.primitive is EffectPrimitive.REGISTER_EVENT_TRIGGER:
            if not isinstance(self.event_trigger, EventTrigger):
                raise ValueError("event registration must declare a trigger")
        elif self.event_trigger is not None:
            raise ValueError("only event registration may declare an event trigger")
        if self.primitive is EffectPrimitive.ON_HP_LOSS_COST_REDUCTION:
            if (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("reviewed HP loss reduction is one per event, floored at zero")
        if self.primitive is EffectPrimitive.EXHAUST_HAND_GROUP:
            if not isinstance(self.hand_exhaust, HandExhaust):
                raise ValueError("hand exhaust must declare filter/payoff")
            expected_target = (EffectTarget.TARGET_ENEMY if self.hand_exhaust.payoff is HandExhaustPayoff.ATTACK
                               else EffectTarget.PLAYER)
            if self.target is not expected_target:
                raise ValueError("hand exhaust target must match payoff")
            if self.hand_exhaust.payoff is HandExhaustPayoff.NONE and (self.base_amount != 0 or self.upgraded_amount != 0):
                raise ValueError("exhaust without payoff must have zero amount")
        elif self.hand_exhaust is not None:
            raise ValueError("only hand-group exhaustion may declare filter/payoff")
        if self.primitive is EffectPrimitive.REGISTER_DELAYED_DAMAGE:
            if not isinstance(self.delayed_damage, DelayedDamage):
                raise ValueError("delayed registration requires a typed countdown contract")
        elif self.delayed_damage is not None:
            raise ValueError("only delayed registration may declare delayed damage")
        if self.primitive is EffectPrimitive.REGISTER_TURN_TRIGGER:
            if not isinstance(self.turn_trigger, TurnTrigger):
                raise ValueError("turn trigger must declare its ordered payload")
            if self.base_amount != 1 or self.upgraded_amount != 1:
                raise ValueError("turn trigger registers one application per play")
        elif self.turn_trigger is not None:
            raise ValueError("only register-turn-trigger steps may declare a turn trigger")
        if self.primitive is EffectPrimitive.FATAL_ATTACK:
            if not isinstance(self.fatal_reward, FatalReward):
                raise ValueError("fatal attack must declare its reward")
            if self.fatal_reward.kind is FatalRewardKind.CARD_DAMAGE and self.dynamic_damage is None:
                raise ValueError("persistent fatal reward requires public dynamic damage")
        elif self.fatal_reward is not None:
            raise ValueError("only fatal attacks may declare a reward")
        if self.primitive is EffectPrimitive.ATTACK_SEQUENCE:
            if not isinstance(self.attack_sequence, AttackSequence):
                raise ValueError("attack sequence primitive must declare its hit profile")
            profile = self.attack_sequence
            expected = {
                EffectTarget.TARGET_ENEMY: (AttackCountSource.FIXED, AttackDamageTiming.SNAPSHOT),
                EffectTarget.RANDOM_ENEMY: (AttackCountSource.FIXED, AttackDamageTiming.EACH_HIT),
                EffectTarget.ALL_ENEMIES: (AttackCountSource.ENERGY_ON_USE, AttackDamageTiming.SNAPSHOT),
            }
            if (profile.count_source, profile.damage_timing) != expected[self.target]:
                raise ValueError("unreviewed attack target/count/timing combination")
        elif self.attack_sequence is not None:
            raise ValueError("only attack sequence steps may declare a hit profile")
        if self.primitive is EffectPrimitive.REGISTER_DRAW_TRIGGER:
            if not isinstance(self.draw_trigger, DrawTrigger):
                raise ValueError("persistent draw trigger must declare its event payload")
        elif self.draw_trigger is not None:
            raise ValueError("only register-draw-trigger steps may declare a trigger")
        if self.primitive is EffectPrimitive.REGISTER_EXHAUST_TRIGGER:
            if not isinstance(self.exhaust_trigger, ExhaustTrigger):
                raise ValueError("persistent exhaust trigger must declare its event payload")
        elif self.exhaust_trigger is not None:
            raise ValueError("only register-exhaust-trigger steps may declare an exhaust trigger")
        expected_source = (
            AmountSource.CURRENT_BLOCK
            if self.primitive in {EffectPrimitive.CURRENT_BLOCK_DAMAGE, EffectPrimitive.CURRENT_BLOCK_GAIN}
            else AmountSource.CURRENT_STRENGTH if self.primitive is EffectPrimitive.GAIN_CURRENT_STRENGTH
            else AmountSource.FIXED
        )
        if self.amount_source is not expected_source:
            raise ValueError(
                f"invalid amount source {self.amount_source.value} "
                f"for {self.primitive.value}"
            )
        if self.condition is not None and not isinstance(
            self.condition, PublicCondition
        ):
            raise ValueError("recipe conditions must be declared public conditions")
        if self.amount_source is AmountSource.FIXED:
            if (
                isinstance(self.base_amount, bool)
                or isinstance(self.upgraded_amount, bool)
                or not isinstance(self.base_amount, int)
                or not isinstance(self.upgraded_amount, int)
                or self.base_amount < 0
                or self.upgraded_amount < 0
            ):
                raise ValueError("fixed recipe steps need non-negative base/upgraded amounts")
        elif self.base_amount is not None or self.upgraded_amount is not None:
            raise ValueError("dynamic recipe steps cannot also declare fixed amounts")
        generated_fields = (
            self.generated_card_source,
            self.generated_card_id,
            self.generated_card_destination,
        )
        if self.primitive is EffectPrimitive.GENERATE_CARD:
            if (
                not isinstance(self.generated_card_source, GeneratedCardSource)
                or not isinstance(
                    self.generated_card_destination, GeneratedCardDestination
                )
            ):
                raise ValueError("generate-card steps need a source and destination")
            if self.generated_card_source is GeneratedCardSource.PLAYED_COPY:
                if self.generated_card_id is not None:
                    raise ValueError("played-copy generation cannot declare a fixed card")
            elif self.generated_card_source is GeneratedCardSource.FIXED_BASE:
                canonical = canonical_identifier(self.generated_card_id, "card")
                if not canonical or canonical != self.generated_card_id:
                    raise ValueError("fixed generated-card identifiers must be canonical")
        elif any(value is not None for value in generated_fields):
            raise ValueError("only generate-card steps may declare card generation")

        if self.primitive is EffectPrimitive.COPY_SELECTED_CARD:
            if not isinstance(self.selected_card_copy, SelectedCardCopy) or (self.base_amount, self.upgraded_amount) != (1, 2):
                raise ValueError("selected-copy payload requires reviewed one/two-copy amounts")
        elif self.selected_card_copy is not None:
            raise ValueError("only selected-copy steps may declare a copy payload")

        selected_fields = (self.selected_card_source, self.selected_card_destination, self.selected_card_filter)
        if self.primitive is EffectPrimitive.MOVE_SELECTED_CARD:
            if (
                not isinstance(self.selected_card_source, SelectedCardSource)
                or not isinstance(
                    self.selected_card_destination, SelectedCardDestination
                )
            ):
                raise ValueError("selected-card steps need a source and destination")
            if (self.selected_card_source, self.selected_card_destination, self.selected_card_filter) not in {
                (SelectedCardSource.HAND, SelectedCardDestination.DRAW_PILE_TOP, None),
                (SelectedCardSource.DISCARD_PILE, SelectedCardDestination.DRAW_PILE_TOP, None),
                (SelectedCardSource.EXHAUST_PILE, SelectedCardDestination.HAND_UNLESS_FULL, SelectedCardFilter.NON_EXHUME),
                (SelectedCardSource.DRAW_PILE, SelectedCardDestination.HAND_OVERFLOW_DISCARD, SelectedCardFilter.SKILL),
                (SelectedCardSource.DRAW_PILE, SelectedCardDestination.HAND_OVERFLOW_DISCARD, SelectedCardFilter.ATTACK),
            }:
                raise ValueError("unreviewed selected-card source/filter/destination")
            if (self.base_amount, self.upgraded_amount) != (1, 1):
                raise ValueError("selected-card movement retrieves exactly one identity")
        elif any(value is not None for value in selected_fields):
            raise ValueError("only selected-card steps may declare card movement")

    def target_for(self, upgraded: bool) -> EffectTarget:
        return self.upgraded_target if upgraded and self.upgraded_target is not None else self.target

    def amount(self, upgraded: bool, *, current_block: int = 0, current_strength: int = 0) -> int:
        if self.amount_source is AmountSource.CURRENT_BLOCK:
            return max(0, int(current_block))
        if self.amount_source is AmountSource.CURRENT_STRENGTH:
            return int(current_strength)
        value = self.upgraded_amount if upgraded else self.base_amount
        assert value is not None
        return value

    @property
    def signature(self) -> str:
        amount = (
            self.amount_source.value
            if self.amount_source is not AmountSource.FIXED
            else f"{self.base_amount}/{self.upgraded_amount}"
        )
        result = f"{self.primitive.value}:{self.target.value}:{amount}"
        if self.upgraded_target is not None:
            result += f":upgraded_target={self.upgraded_target.value}"
        if self.random_hand_generation is not None:
            result += f":{self.random_hand_generation.signature}"
        if self.random_draw_generation is not None:
            result += f":{self.random_draw_generation.signature}"
        if self.discovery_generation is not None:
            result += f":{self.discovery_generation.value}"
        if self.dynamic_damage is not None:
            result += f":scaling={self.dynamic_damage.signature}"
        if self.hand_cost is not None:
            result += f":cost={self.hand_cost.signature}"
        if self.combat_upgrade is not None:
            result += f":upgrade={self.combat_upgrade.signature}"
        if self.strength_reduction is not None:
            result += f":{self.strength_reduction.signature}"
        if self.event_trigger is not None:
            result += f":on_event={self.event_trigger.signature}"
        if self.hand_exhaust is not None:
            result += f":hand_group={self.hand_exhaust.signature}"
        if self.turn_trigger is not None:
            result += f":on_turn={self.turn_trigger.signature}"
        if self.delayed_damage is not None:
            result += f":delayed={self.delayed_damage.signature}"
        if self.fatal_reward is not None:
            result += f":reward={self.fatal_reward.signature}"
        if self.attack_sequence is not None:
            result += f":{self.attack_sequence.signature}"
        if self.condition is not None:
            result += f":when={self.condition.value}"
        if self.draw_trigger is not None:
            result += f":on_draw={self.draw_trigger.signature}"
        if self.exhaust_trigger is not None:
            result += f":on_exhaust={self.exhaust_trigger.signature}"
        if self.primitive is EffectPrimitive.GENERATE_CARD:
            assert self.generated_card_source is not None
            assert self.generated_card_destination is not None
            card = self.generated_card_id or "<played>"
            result += (
                f":{self.generated_card_source.value}:{card}:"
                f"{self.generated_card_destination.value}"
            )
        elif self.primitive is EffectPrimitive.MOVE_SELECTED_CARD:
            assert self.selected_card_source is not None
            assert self.selected_card_destination is not None
            result += (
                f":{self.selected_card_source.value}:"
                f"{self.selected_card_destination.value}"
            )
            if self.selected_card_filter is not None:
                result += f":{self.selected_card_filter.value}"
        if self.selected_card_copy is not None:
            result += f":{self.selected_card_copy.signature}"
        if self.attack_repeat is not None:
            result += f":{self.attack_repeat.value}"
        if self.top_card_autoplay is not None:
            result += f":{self.top_card_autoplay.value}"
        return result


class IntrinsicTiming(str, Enum):
    DRAW = "on_actual_draw_queue_bottom"
    END_TURN_HAND = "end_turn_in_hand_no_use_triggers"


class IntrinsicEffect(str, Enum):
    LOSE_ENERGY = "lose_energy_floor_zero"
    SELF_DAMAGE = "self_sourced_blockable_non_attack_damage"
    SELF_HP_LOSS = "self_sourced_block_bypassing_hp_loss"
    SELF_WEAK = "self_weak_artifact_blockable_fresh_skip_first_round_decay"
    SELF_FRAIL = "self_frail_artifact_blockable_fresh_skip_first_round_decay"


class IntrinsicAmountSource(str, Enum):
    FIXED = "fixed"
    END_HAND_SNAPSHOT = "hand_count_including_self_at_end_turn_callback"


@dataclass(frozen=True, slots=True)
class IntrinsicCardEffect:
    """Passive card callback, never an effect of manually playing the card."""
    timing: IntrinsicTiming
    effect: IntrinsicEffect
    base_amount: int
    upgraded_amount: int
    amount_source: IntrinsicAmountSource = IntrinsicAmountSource.FIXED

    def __post_init__(self) -> None:
        if not isinstance(self.timing, IntrinsicTiming) or not isinstance(self.effect, IntrinsicEffect):
            raise ValueError("intrinsic card effects require declared timing/effect")
        if not isinstance(self.amount_source, IntrinsicAmountSource):
            raise ValueError("intrinsic amount source must be declared")
        if any(type(n) is not int or n <= 0 for n in (self.base_amount, self.upgraded_amount)):
            raise ValueError("intrinsic card amounts must be positive integers")

    def amount(self, upgraded: bool, *, end_hand_count: int | None = None) -> int:
        coefficient = self.upgraded_amount if upgraded else self.base_amount
        if self.amount_source is IntrinsicAmountSource.END_HAND_SNAPSHOT:
            if type(end_hand_count) is not int or end_hand_count < 1:
                raise ValueError("end-hand callback requires a positive snapshot including self")
            return coefficient * end_hand_count
        return coefficient

    @property
    def signature(self) -> str:
        result = f"{self.timing.value}:{self.effect.value}:{self.base_amount}/{self.upgraded_amount}"
        if self.amount_source is not IntrinsicAmountSource.FIXED:
            result += f":amount_source={self.amount_source.value}"
        return result


class ResidentRuleKind(str, Enum):
    OTHER_PLAY_HP_LOSS = "other_card_use_queue_top_self_hp_loss_per_resident_identity"
    TURN_PLAY_LIMIT = "hand_resident_limit_all_successful_turn_plays_including_auto_repeat"


@dataclass(frozen=True, slots=True)
class ResidentCardRule:
    kind: ResidentRuleKind
    amount: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResidentRuleKind) or type(self.amount) is not int or self.amount <= 0:
            raise ValueError("resident rule requires declared kind and positive integer amount")

    @property
    def signature(self) -> str:
        return f"{self.kind.value}:{self.amount}"


class RemovalEligibility(str, Enum):
    STANDARD = "standard_purge_and_transform_selection"
    EXCLUDED = "excluded_from_purge_and_transform_selection"


class SelfCopyTiming(str, Enum):
    EXHAUST = "after_exhaust_callbacks"
    END_HAND = "end_turn_hand_snapshot_no_use"


class SelfCopyMode(str, Enum):
    BASE_HAND = "fresh_base_copy_to_hand_overflow_discard"
    STAT_DRAW_TOP = "fresh_stat_equivalent_copy_to_draw_top"


@dataclass(frozen=True, slots=True)
class SelfCopyCallback:
    timing: SelfCopyTiming
    mode: SelfCopyMode
    count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.timing, SelfCopyTiming) or not isinstance(self.mode, SelfCopyMode):
            raise ValueError("self-copy callbacks require typed timing/mode")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("self-copy count must be a positive integer")

    @property
    def signature(self) -> str:
        return f"{self.timing.value}:{self.mode.value}:{self.count}"


@dataclass(frozen=True, slots=True)
class MasterDeckLifecycle:
    """Reviewed selection/removal only, not combat exhaust or full-run policy.

    The loss amount is nominal, not an implementation of max-HP clamping.
    """
    removal: RemovalEligibility = RemovalEligibility.STANDARD
    max_hp_loss_on_remove: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.removal, RemovalEligibility):
            raise ValueError("master-deck removal requires declared eligibility")
        if type(self.max_hp_loss_on_remove) is not int or self.max_hp_loss_on_remove < 0:
            raise ValueError("master-deck HP loss must be a nonnegative integer")

    @property
    def signature(self) -> str:
        return f"{self.removal.value}:on_master_remove_max_hp_loss={self.max_hp_loss_on_remove}"

    def as_dict(self) -> dict[str, object]:
        return {"removal_selection": self.removal.value,
                "max_hp_loss_on_master_remove": self.max_hp_loss_on_remove,
                "scope": "selection_and_master_removal_only_not_combat_exhaust_or_full_run_policy"}


@dataclass(frozen=True, slots=True)
class CardEffectRecipe:
    identifier: str
    effects: tuple[EffectStep, ...]
    traits: tuple[CardTrait, ...] = ()
    intrinsic_effects: tuple[IntrinsicCardEffect, ...] = ()
    master_deck: MasterDeckLifecycle | None = None
    resident_rule: ResidentCardRule | None = None
    self_copy: SelfCopyCallback | None = None

    def __post_init__(self) -> None:
        canonical = canonical_identifier(self.identifier, "card")
        if not canonical or canonical != self.identifier:
            raise ValueError("card recipe identifiers must be canonical")
        passive_traits = {CardTrait.UNPLAYABLE_STATUS, CardTrait.UNPLAYABLE_CURSE}
        empty_exhaust = set(self.traits).issubset({CardTrait.EXHAUST, CardTrait.INNATE, CardTrait.PLAYABLE_CURSE}) and CardTrait.EXHAUST in self.traits
        if not self.effects and not empty_exhaust and not passive_traits.intersection(self.traits):
            raise ValueError("empty card recipes must declare only self Exhaust or an explicit unplayable trait")
        if passive_traits.issubset(self.traits):
            raise ValueError("status and curse play permissions cannot coexist")
        if self.master_deck is not None and not isinstance(self.master_deck, MasterDeckLifecycle):
            raise ValueError("master-deck lifecycle requires a typed contract")
        if self.resident_rule is not None and not isinstance(self.resident_rule, ResidentCardRule):
            raise ValueError("resident effects require a typed contract")
        if self.self_copy is not None and not isinstance(self.self_copy, SelfCopyCallback):
            raise ValueError("self-copy requires a typed callback")
        if CardTrait.PLAYABLE_CURSE in self.traits and passive_traits.intersection(self.traits):
            raise ValueError("natural curse play cannot also be unplayable")
        if any(not isinstance(effect, IntrinsicCardEffect) for effect in self.intrinsic_effects):
            raise ValueError("intrinsic card effects must use typed contracts")
        if len(set(self.traits)) != len(self.traits):
            raise ValueError("card recipe traits must be unique")
        if CardTrait.EXHAUST in self.traits and CardTrait.BASE_EXHAUST in self.traits:
            raise ValueError("exhaust traits cannot contradict variant lifecycle")
        if CardTrait.INNATE in self.traits and CardTrait.UPGRADE_INNATE in self.traits:
            raise ValueError("innate traits cannot contradict variant lifecycle")
        if (
            any(
                step.condition is PublicCondition.TARGET_HAS_VULNERABLE
                for step in self.effects
            )
            and not any(
                step.target is EffectTarget.TARGET_ENEMY for step in self.effects
            )
            and CardTrait.TARGETED_ENEMY not in self.traits
        ):
            raise ValueError("target conditions require a targeted recipe")
        if (
            any(
                step.condition is PublicCondition.TARGET_ATTACK_INTENT
                for step in self.effects
            )
            and CardTrait.TARGETED_ENEMY not in self.traits
        ):
            raise ValueError("attack-intent conditions require a targeted recipe")
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "traits", tuple(self.traits))
        object.__setattr__(self, "intrinsic_effects", tuple(self.intrinsic_effects))

    def exhausts(self, upgraded: bool) -> bool:
        return CardTrait.EXHAUST in self.traits or (CardTrait.BASE_EXHAUST in self.traits and not upgraded)

    def ethereal(self, upgraded: bool) -> bool:
        return CardTrait.ETHEREAL in self.traits or (CardTrait.BASE_ETHEREAL in self.traits and not upgraded)

    def innate(self, upgraded: bool) -> bool:
        return CardTrait.INNATE in self.traits or (CardTrait.UPGRADE_INNATE in self.traits and upgraded)

    @property
    def composition(self) -> str:
        if CardTrait.UNPLAYABLE_CURSE in self.traits and not self.effects:
            return "passive_curse"
        if CardTrait.UNPLAYABLE_STATUS in self.traits and not self.effects:
            return "passive_status"
        if not self.effects:
            return "exhaust_only"
        if any(step.condition is not None for step in self.effects):
            return "conditional_sequence"
        return "single" if len(self.effects) == 1 else "ordered_sequence"

    @property
    def mechanisms(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for step in self.effects:
            if step.upgraded_target is not None:
                ordered.append("upgrade_target_scope")
            if step.condition is not None:
                ordered.append(step.condition.value)
            ordered.append(step.primitive.value)
            if step.attack_sequence is not None:
                if step.target is EffectTarget.RANDOM_ENEMY:
                    ordered.extend(("random_target", "per_hit_recalculation"))
                if step.attack_sequence.count_source is AttackCountSource.ENERGY_ON_USE:
                    ordered.extend(("x_cost", "multi_hit", "area_damage"))
                if max(step.attack_sequence.base_hits, step.attack_sequence.upgraded_hits) > 1:
                    ordered.append("multi_hit")
                if max(step.attack_sequence.base_strength_multiplier,
                       step.attack_sequence.upgraded_strength_multiplier) > 1:
                    ordered.append("strength_scaling")
        ordered.extend(effect.signature for effect in self.intrinsic_effects)
        if self.self_copy is not None:
            ordered.append("self_copy:" + self.self_copy.signature)
        if self.resident_rule is not None:
            ordered.append("resident:" + self.resident_rule.signature)
        if self.master_deck is not None:
            ordered.append("master_deck:" + self.master_deck.signature)
        ordered.extend(trait.value for trait in self.traits)
        return tuple(dict.fromkeys(ordered))

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(step.signature for step in self.effects) + tuple(
            f"intrinsic:{effect.signature}" for effect in self.intrinsic_effects
        ) + (("self_copy:" + self.self_copy.signature,) if self.self_copy else ()) + (("resident:" + self.resident_rule.signature,) if self.resident_rule else ()) + (("master_deck:" + self.master_deck.signature,) if self.master_deck else ()) + tuple(
            f"trait:{trait.value}" for trait in self.traits
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "version": CARD_RECIPE_VERSION,
            "composition": self.composition,
            "signature": list(self.signature),
        }


class CardRecipeCatalog:
    version = CARD_RECIPE_VERSION

    def __init__(self, recipes: Iterable[CardEffectRecipe]) -> None:
        indexed: dict[str, CardEffectRecipe] = {}
        for recipe in recipes:
            if recipe.identifier in indexed:
                raise ValueError(f"duplicate card recipe: {recipe.identifier}")
            indexed[recipe.identifier] = recipe
        self._recipes: Mapping[str, CardEffectRecipe] = MappingProxyType(indexed)

    def lookup(self, value: object) -> CardEffectRecipe | None:
        identifier = canonical_identifier(value, "card")
        return self._recipes.get(identifier or "<missing>")

    def identifiers(self) -> frozenset[str]:
        return frozenset(self._recipes)

    def recipes(self) -> tuple[CardEffectRecipe, ...]:
        return tuple(self._recipes.values())


def _fixed(
    primitive: EffectPrimitive,
    target: EffectTarget,
    base: int,
    upgraded: int,
) -> EffectStep:
    return EffectStep(primitive, target, base, upgraded)


def _dynamic(
    primitive: EffectPrimitive,
    target: EffectTarget,
    source: AmountSource,
) -> EffectStep:
    return EffectStep(primitive, target, amount_source=source)


def _conditional_fixed(
    condition: PublicCondition,
    primitive: EffectPrimitive,
    target: EffectTarget,
    base: int,
    upgraded: int,
) -> EffectStep:
    return EffectStep(
        primitive,
        target,
        base,
        upgraded,
        condition=condition,
    )


def _generate(
    source: GeneratedCardSource,
    destination: GeneratedCardDestination,
    card_id: str | None = None,
    amount: int = 1,
) -> EffectStep:
    return EffectStep(
        EffectPrimitive.GENERATE_CARD,
        EffectTarget.PLAYER,
        amount,
        amount,
        generated_card_source=source,
        generated_card_id=card_id,
        generated_card_destination=destination,
    )


def _move_selected(
    source: SelectedCardSource,
    destination: SelectedCardDestination,
) -> EffectStep:
    return EffectStep(
        EffectPrimitive.MOVE_SELECTED_CARD,
        EffectTarget.PLAYER,
        1,
        1,
        selected_card_source=source,
        selected_card_destination=destination,
    )


_T = EffectTarget.TARGET_ENEMY
_A = EffectTarget.ALL_ENEMIES
_P = EffectTarget.PLAYER


DEFAULT_CARD_RECIPES = CardRecipeCatalog((
    CardEffectRecipe("deep_breath", (
        _fixed(EffectPrimitive.SHUFFLE_DISCARD_INTO_DRAW, _P, 1, 1),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 2),
    )),
    CardEffectRecipe("impatience", (
        _conditional_fixed(PublicCondition.HAND_HAS_NO_ATTACK, EffectPrimitive.DRAW_CARDS, _P, 2, 3),
    )),
    CardEffectRecipe("blind", (
        EffectStep(EffectPrimitive.APPLY_WEAK, _T, 2, 2, upgraded_target=_A),
    )),
    CardEffectRecipe("trip", (
        EffectStep(EffectPrimitive.APPLY_VULNERABLE, _T, 2, 2, upgraded_target=_A),
    )),
    CardEffectRecipe("discovery", (EffectStep(
        EffectPrimitive.GENERATE_CHOICE_HAND, _P, 1, 1,
        discovery_generation=DiscoveryGeneration.IRONCLAD_THREE,
    ),), (CardTrait.BASE_EXHAUST,)),
    CardEffectRecipe("chrysalis", (EffectStep(
        EffectPrimitive.GENERATE_RANDOM_DRAW, _P, 3, 5,
        random_draw_generation=RandomDrawGeneration.IRONCLAD_SKILL,
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("metamorphosis", (EffectStep(
        EffectPrimitive.GENERATE_RANDOM_DRAW, _P, 3, 5,
        random_draw_generation=RandomDrawGeneration.IRONCLAD_ATTACK,
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("infernal_blade", (EffectStep(
        EffectPrimitive.GENERATE_RANDOM_HAND, _P, 1, 1,
        random_hand_generation=RandomHandGeneration.IRONCLAD_ATTACK_TURN_ZERO,
    ),), (CardTrait.EXHAUST, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("jack_of_all_trades", (EffectStep(
        EffectPrimitive.GENERATE_RANDOM_HAND, _P, 1, 2,
        random_hand_generation=RandomHandGeneration.COLORLESS_BASE,
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("transmutation", (EffectStep(
        EffectPrimitive.GENERATE_X_HAND, _P, 1, 1,
        random_hand_generation=RandomHandGeneration.COLORLESS_X_TURN_ZERO,
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("the_bomb", (EffectStep(
        EffectPrimitive.REGISTER_DELAYED_DAMAGE, _P, 40, 50,
        delayed_damage=DelayedDamage(),
    ),)),
    CardEffectRecipe("mayhem", (EffectStep(
        EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
        turn_trigger=TurnTrigger("mayhem", TurnTiming.START, (
            TurnEffect(TurnEffectKind.AUTOPLAY_TOP, 1, 1,
                       top_card_autoplay=TopCardAutoplayContract.MAYHEM),
        )),
    ),), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("magnetism", (EffectStep(
        EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
        turn_trigger=TurnTrigger("magnetism", TurnTiming.START, (
            TurnEffect(TurnEffectKind.GENERATE_RANDOM_HAND, 1, 1,
                       RandomHandGeneration.COLORLESS_TURN_START),
        )),
    ),), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("havoc", (EffectStep(
        EffectPrimitive.AUTOPLAY_TOP_CARD, _P, 1, 1,
        top_card_autoplay=TopCardAutoplayContract.HAVOC,
    ),), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("double_tap", (EffectStep(
        EffectPrimitive.REGISTER_ATTACK_REPEAT, _P, 1, 2,
        attack_repeat=AttackRepeatContract.SAME_ID_PURGED_AUTOPLAY,
    ),)),
    CardEffectRecipe("dual_wield", (EffectStep(
        EffectPrimitive.COPY_SELECTED_CARD, _P, 1, 2,
        selected_card_copy=SelectedCardCopy(),
    ),)),
    CardEffectRecipe("exhume", (EffectStep(
        EffectPrimitive.MOVE_SELECTED_CARD, _P, 1, 1,
        selected_card_source=SelectedCardSource.EXHAUST_PILE,
        selected_card_destination=SelectedCardDestination.HAND_UNLESS_FULL,
        selected_card_filter=SelectedCardFilter.NON_EXHUME,
    ),), (CardTrait.EXHAUST, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("secret_technique", (EffectStep(
        EffectPrimitive.MOVE_SELECTED_CARD, _P, 1, 1,
        selected_card_source=SelectedCardSource.DRAW_PILE,
        selected_card_destination=SelectedCardDestination.HAND_OVERFLOW_DISCARD,
        selected_card_filter=SelectedCardFilter.SKILL,
    ),), (CardTrait.BASE_EXHAUST, CardTrait.PUBLIC_HAND_PLAYABILITY)),
    CardEffectRecipe("secret_weapon", (EffectStep(
        EffectPrimitive.MOVE_SELECTED_CARD, _P, 1, 1,
        selected_card_source=SelectedCardSource.DRAW_PILE,
        selected_card_destination=SelectedCardDestination.HAND_OVERFLOW_DISCARD,
        selected_card_filter=SelectedCardFilter.ATTACK,
    ),), (CardTrait.BASE_EXHAUST, CardTrait.PUBLIC_HAND_PLAYABILITY)),
    CardEffectRecipe("purity", (
        _fixed(EffectPrimitive.EXHAUST_UP_TO_HAND, _P, 3, 5),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("forethought", (
        _fixed(EffectPrimitive.SELECT_BOTTOM_FREE, _P, 1, 10),
    )),
    CardEffectRecipe("panacea", (
        _fixed(EffectPrimitive.GAIN_ARTIFACT, _P, 1, 2),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("panache", (EffectStep(
        EffectPrimitive.REGISTER_EVENT_TRIGGER, _P, 10, 14,
        event_trigger=EventTrigger("panache", CombatEvent.FIFTH_CARD_PLAYED, EventEffect.AREA_DAMAGE),
    ),)),
    CardEffectRecipe("sadistic_nature", (EffectStep(
        EffectPrimitive.REGISTER_EVENT_TRIGGER, _P, 5, 7,
        event_trigger=EventTrigger("sadistic", CombatEvent.ENEMY_DEBUFF_APPLIED, EventEffect.ENEMY_DAMAGE),
    ),)),
    CardEffectRecipe("panic_button", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 30, 40),
        _fixed(EffectPrimitive.APPLY_NO_BLOCK, _P, 2, 2),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("violence", (
        _fixed(EffectPrimitive.RETRIEVE_RANDOM_ATTACKS, _P, 3, 4),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("dramatic_entrance", (
        _fixed(EffectPrimitive.AREA_DAMAGE, _A, 8, 12),
    ), (CardTrait.INNATE, CardTrait.EXHAUST)),
    CardEffectRecipe("mind_blast", (EffectStep(
        EffectPrimitive.DYNAMIC_DAMAGE, _T, 0, 0,
        dynamic_damage=DynamicDamage(DamageScaling.DRAW_PILE_COUNT),
    ),), (CardTrait.INNATE, CardTrait.TARGETED_ENEMY, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("perfected_strike", (EffectStep(
        EffectPrimitive.DYNAMIC_DAMAGE, _T, 6, 6,
        dynamic_damage=DynamicDamage(DamageScaling.STRIKE_COUNT),
    ),), (CardTrait.TARGETED_ENEMY,)),
    CardEffectRecipe("rampage", (EffectStep(
        EffectPrimitive.DYNAMIC_DAMAGE, _T, 8, 8,
        dynamic_damage=DynamicDamage(DamageScaling.CARD_BASE_DAMAGE),
    ), _fixed(EffectPrimitive.GROW_PLAYED_DAMAGE, _P, 5, 8)), (CardTrait.TARGETED_ENEMY,)),
    CardEffectRecipe("searing_blow", (EffectStep(
        EffectPrimitive.DYNAMIC_DAMAGE, _T, 12, 16,
        dynamic_damage=DynamicDamage(DamageScaling.UPGRADE_COUNT),
    ),), (CardTrait.TARGETED_ENEMY,)),
    CardEffectRecipe("corruption", (
        _fixed(EffectPrimitive.REGISTER_CORRUPTION, _P, 1, 1),
    ), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("madness", (EffectStep(
        EffectPrimitive.MODIFY_HAND_COST, _P, 0, 0,
        hand_cost=HandCostChange(CostSelection.RANDOM_POSITIVE, CostDuration.COMBAT, CostDuration.COMBAT),
    ),), (CardTrait.EXHAUST, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("enlightenment", (EffectStep(
        EffectPrimitive.MODIFY_HAND_COST, _P, 1, 1,
        hand_cost=HandCostChange(CostSelection.ALL_HAND, CostDuration.TURN, CostDuration.COMBAT),
    ),)),
    CardEffectRecipe("armaments", (
        EffectStep(EffectPrimitive.GAIN_BLOCK, _P, 5, 5),
        EffectStep(EffectPrimitive.UPGRADE_CARDS, _P, 1, 1,
                   combat_upgrade=CombatUpgrade(UpgradeScope.CHOSEN_HAND, UpgradeScope.ALL_HAND)),
    )),
    CardEffectRecipe("apotheosis", (EffectStep(
        EffectPrimitive.UPGRADE_CARDS, _P, 1, 1,
        combat_upgrade=CombatUpgrade(UpgradeScope.ALL_COMBAT_PILES, UpgradeScope.ALL_COMBAT_PILES),
    ),), (CardTrait.EXHAUST, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("disarm", (EffectStep(
        EffectPrimitive.ENEMY_STRENGTH_LOSS, _T, 2, 3,
        strength_reduction=StrengthReduction(),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("dark_shackles", (EffectStep(
        EffectPrimitive.ENEMY_STRENGTH_LOSS, _T, 9, 15,
        strength_reduction=StrengthReduction(StrengthRestoration.ENEMY_TURN_END),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("limit_break", (EffectStep(
        EffectPrimitive.GAIN_CURRENT_STRENGTH, _P, amount_source=AmountSource.CURRENT_STRENGTH,
    ),), (CardTrait.BASE_EXHAUST,)),
    CardEffectRecipe("rupture", (EffectStep(
        EffectPrimitive.REGISTER_EVENT_TRIGGER, _P, 1, 2,
        event_trigger=EventTrigger("rupture", CombatEvent.SELF_HP_LOST, EventEffect.STRENGTH),
    ),)),
    CardEffectRecipe("rage", (EffectStep(
        EffectPrimitive.REGISTER_EVENT_TRIGGER, _P, 3, 5,
        event_trigger=EventTrigger("rage", CombatEvent.ATTACK_PLAYED, EventEffect.NON_CARD_BLOCK, TriggerDuration.TURN),
    ),)),
    CardEffectRecipe("juggernaut", (EffectStep(
        EffectPrimitive.REGISTER_EVENT_TRIGGER, _P, 5, 7,
        event_trigger=EventTrigger("juggernaut", CombatEvent.BLOCK_GAINED, EventEffect.RANDOM_DAMAGE),
    ),)),
    CardEffectRecipe("blood_for_blood", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 18, 22),
        _fixed(EffectPrimitive.ON_HP_LOSS_COST_REDUCTION, _P, 1, 1),
    ), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("second_wind", (EffectStep(
        EffectPrimitive.EXHAUST_HAND_GROUP, _P, 5, 7,
        hand_exhaust=HandExhaust(HandExhaustFilter.NON_ATTACK, HandExhaustPayoff.BLOCK),
    ),)),
    CardEffectRecipe("sever_soul", (
        EffectStep(EffectPrimitive.EXHAUST_HAND_GROUP, _P, 0, 0,
                   hand_exhaust=HandExhaust(HandExhaustFilter.NON_ATTACK, HandExhaustPayoff.NONE)),
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 16, 22),
    )),
    CardEffectRecipe("fiend_fire", (EffectStep(
        EffectPrimitive.EXHAUST_HAND_GROUP, _T, 7, 10,
        hand_exhaust=HandExhaust(HandExhaustFilter.ALL, HandExhaustPayoff.ATTACK),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("sentinel", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 5, 8),
        _fixed(EffectPrimitive.ON_EXHAUST_ENERGY, _P, 2, 3),
    )),
    CardEffectRecipe("brutality", (EffectStep(
        EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
        turn_trigger=TurnTrigger("brutality", TurnTiming.POST_DRAW, (
            TurnEffect(TurnEffectKind.DRAW, 1, 1),
            TurnEffect(TurnEffectKind.SELF_HP_LOSS, 1, 1),
        )),
    ),), (CardTrait.UPGRADE_INNATE,)),
    CardEffectRecipe("combust", (EffectStep(
        EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
        turn_trigger=TurnTrigger("combust", TurnTiming.END, (
            TurnEffect(TurnEffectKind.SELF_HP_LOSS, 1, 1),
            TurnEffect(TurnEffectKind.NON_ATTACK_AREA_DAMAGE, 5, 7),
        )),
    ),)),
    CardEffectRecipe("demon_form", (EffectStep(
        EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
        turn_trigger=TurnTrigger("demon_form", TurnTiming.POST_DRAW, (
            TurnEffect(TurnEffectKind.STRENGTH, 2, 3),
        )),
    ),)),
    CardEffectRecipe("berserk", (
        _fixed(EffectPrimitive.SELF_VULNERABLE, _P, 2, 1),
        EffectStep(EffectPrimitive.REGISTER_TURN_TRIGGER, _P, 1, 1,
                   turn_trigger=TurnTrigger("berserk", TurnTiming.START, (
                       TurnEffect(TurnEffectKind.ENERGY, 1, 1),
                   ))),
    )),
    CardEffectRecipe("barricade", (_fixed(EffectPrimitive.RETAIN_BLOCK, _P, 1, 1),),
                     (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("entrench", (EffectStep(
        EffectPrimitive.CURRENT_BLOCK_GAIN, _P, amount_source=AmountSource.CURRENT_BLOCK,
    ),), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("metallicize", (_fixed(EffectPrimitive.END_TURN_BLOCK, _P, 3, 4),)),
    CardEffectRecipe("flame_barrier", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 12, 16),
        _fixed(EffectPrimitive.TEMP_RETALIATION, _P, 4, 6),
    )),
    CardEffectRecipe("bandage_up", (_fixed(EffectPrimitive.HEAL, _P, 4, 6),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("apparition", (_fixed(EffectPrimitive.GAIN_INTANGIBLE, _P, 1, 1),),
                     (CardTrait.EXHAUST, CardTrait.BASE_ETHEREAL)),
    CardEffectRecipe("ritual_dagger", (EffectStep(
        EffectPrimitive.FATAL_ATTACK, _T, 15, 15,
        dynamic_damage=DynamicDamage(DamageScaling.PERSISTENT_CARD_BASE_DAMAGE),
        fatal_reward=FatalReward(FatalRewardKind.CARD_DAMAGE, 3, 5),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("bite", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 7, 8),
        _fixed(EffectPrimitive.HEAL, _P, 2, 3),
    )),
    CardEffectRecipe("reaper", (
        _fixed(EffectPrimitive.AREA_LIFESTEAL, _A, 4, 5),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("feed", (EffectStep(
        EffectPrimitive.FATAL_ATTACK, _T, 10, 12,
        fatal_reward=FatalReward(FatalRewardKind.MAX_HP, 3, 4),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("hand_of_greed", (EffectStep(
        EffectPrimitive.FATAL_ATTACK, _T, 20, 25,
        fatal_reward=FatalReward(FatalRewardKind.GOLD, 20, 25),
    ),)),
    CardEffectRecipe("sword_boomerang", (EffectStep(
        EffectPrimitive.ATTACK_SEQUENCE, EffectTarget.RANDOM_ENEMY, 3, 3,
        attack_sequence=AttackSequence(3, 4, damage_timing=AttackDamageTiming.EACH_HIT),
    ),)),
    CardEffectRecipe("whirlwind", (EffectStep(
        EffectPrimitive.ATTACK_SEQUENCE, _A, 5, 8,
        attack_sequence=AttackSequence(count_source=AttackCountSource.ENERGY_ON_USE),
    ),)),
    CardEffectRecipe("twin_strike", (EffectStep(
        EffectPrimitive.ATTACK_SEQUENCE, _T, 5, 7,
        attack_sequence=AttackSequence(2, 2),
    ),)),
    CardEffectRecipe("pummel", (EffectStep(
        EffectPrimitive.ATTACK_SEQUENCE, _T, 2, 2,
        attack_sequence=AttackSequence(4, 5),
    ),), (CardTrait.EXHAUST,)),
    CardEffectRecipe("heavy_blade", (EffectStep(
        EffectPrimitive.ATTACK_SEQUENCE, _T, 14, 14,
        attack_sequence=AttackSequence(1, 1, 3, 5),
    ),)),
    CardEffectRecipe("swift_strike", (_fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 7, 10),)),
    CardEffectRecipe("good_instincts", (_fixed(EffectPrimitive.GAIN_BLOCK, _P, 6, 9),)),
    CardEffectRecipe("flash_of_steel", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 3, 6),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 1),
    )),
    CardEffectRecipe("finesse", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 2, 4),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 1),
    )),
    CardEffectRecipe("master_of_strategy", (
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 3, 4),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("carnage", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 20, 28),
    ), (CardTrait.ETHEREAL,)),
    CardEffectRecipe("hemokinesis", (
        _fixed(EffectPrimitive.LOSE_HP, _P, 2, 2),
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 15, 20),
    )),
    CardEffectRecipe("jax", (
        _fixed(EffectPrimitive.LOSE_HP, _P, 3, 3),
        _fixed(EffectPrimitive.GAIN_STRENGTH, _P, 2, 3),
    )),
    CardEffectRecipe("feel_no_pain", (EffectStep(
        EffectPrimitive.REGISTER_EXHAUST_TRIGGER, _P, 3, 4,
        exhaust_trigger=ExhaustTrigger("feel_no_pain", ExhaustTriggerEffect.NON_CARD_BLOCK),
    ),)),
    CardEffectRecipe("dark_embrace", (EffectStep(
        EffectPrimitive.REGISTER_EXHAUST_TRIGGER, _P, 1, 1,
        exhaust_trigger=ExhaustTrigger("dark_embrace", ExhaustTriggerEffect.DRAW),
    ),), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("evolve", (EffectStep(
        EffectPrimitive.REGISTER_DRAW_TRIGGER, _P, 1, 2,
        draw_trigger=DrawTrigger("evolve", ("STATUS",), DrawTriggerEffect.DRAW),
    ),)),
    CardEffectRecipe("fire_breathing", (EffectStep(
        EffectPrimitive.REGISTER_DRAW_TRIGGER, _P, 6, 10,
        draw_trigger=DrawTrigger(
            "fire_breathing", ("STATUS", "CURSE"), DrawTriggerEffect.NON_ATTACK_AREA_DAMAGE,
        ),
    ),)),
    CardEffectRecipe("slimed", (), (CardTrait.EXHAUST,)),
    CardEffectRecipe("wound", (), (CardTrait.UNPLAYABLE_STATUS,)),
    CardEffectRecipe("decay", (), (CardTrait.UNPLAYABLE_CURSE,), intrinsic_effects=(
        IntrinsicCardEffect(IntrinsicTiming.END_TURN_HAND, IntrinsicEffect.SELF_DAMAGE, 2, 2),
    ), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("doubt", (), (CardTrait.UNPLAYABLE_CURSE,), intrinsic_effects=(
        IntrinsicCardEffect(IntrinsicTiming.END_TURN_HAND, IntrinsicEffect.SELF_WEAK, 1, 1),
    ), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("shame", (), (CardTrait.UNPLAYABLE_CURSE,), intrinsic_effects=(
        IntrinsicCardEffect(IntrinsicTiming.END_TURN_HAND, IntrinsicEffect.SELF_FRAIL, 1, 1),
    ), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("regret", (), (CardTrait.UNPLAYABLE_CURSE,), intrinsic_effects=(
        IntrinsicCardEffect(IntrinsicTiming.END_TURN_HAND, IntrinsicEffect.SELF_HP_LOSS, 1, 1,
                            IntrinsicAmountSource.END_HAND_SNAPSHOT),
    ), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("injury", (), (CardTrait.UNPLAYABLE_CURSE,), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("necronomicurse", (), (CardTrait.UNPLAYABLE_CURSE,),
                     master_deck=MasterDeckLifecycle(RemovalEligibility.EXCLUDED),
                     self_copy=SelfCopyCallback(SelfCopyTiming.EXHAUST, SelfCopyMode.BASE_HAND)),
    CardEffectRecipe("pride", (), (CardTrait.EXHAUST, CardTrait.INNATE, CardTrait.PLAYABLE_CURSE),
                     master_deck=MasterDeckLifecycle(),
                     self_copy=SelfCopyCallback(SelfCopyTiming.END_HAND, SelfCopyMode.STAT_DRAW_TOP)),
    CardEffectRecipe("pain", (), (CardTrait.UNPLAYABLE_CURSE,), master_deck=MasterDeckLifecycle(),
                     resident_rule=ResidentCardRule(ResidentRuleKind.OTHER_PLAY_HP_LOSS, 1)),
    CardEffectRecipe("normality", (), (CardTrait.UNPLAYABLE_CURSE,), master_deck=MasterDeckLifecycle(),
                     resident_rule=ResidentCardRule(ResidentRuleKind.TURN_PLAY_LIMIT, 3)),
    CardEffectRecipe("clumsy", (), (CardTrait.UNPLAYABLE_CURSE, CardTrait.ETHEREAL), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("writhe", (), (CardTrait.UNPLAYABLE_CURSE, CardTrait.INNATE), master_deck=MasterDeckLifecycle()),
    CardEffectRecipe("ascenders_bane", (), (CardTrait.UNPLAYABLE_CURSE, CardTrait.ETHEREAL),
                     master_deck=MasterDeckLifecycle(RemovalEligibility.EXCLUDED)),
    CardEffectRecipe("curse_of_the_bell", (), (CardTrait.UNPLAYABLE_CURSE,),
                     master_deck=MasterDeckLifecycle(RemovalEligibility.EXCLUDED)),
    CardEffectRecipe("parasite", (), (CardTrait.UNPLAYABLE_CURSE,),
                     master_deck=MasterDeckLifecycle(max_hp_loss_on_remove=3)),
    CardEffectRecipe("dazed", (), (CardTrait.UNPLAYABLE_STATUS, CardTrait.ETHEREAL)),
    CardEffectRecipe("burn", (), (CardTrait.UNPLAYABLE_STATUS,), (
        IntrinsicCardEffect(IntrinsicTiming.END_TURN_HAND, IntrinsicEffect.SELF_DAMAGE, 2, 4),
    )),
    CardEffectRecipe("void", (), (CardTrait.UNPLAYABLE_STATUS, CardTrait.ETHEREAL), (
        IntrinsicCardEffect(IntrinsicTiming.DRAW, IntrinsicEffect.LOSE_ENERGY, 1, 1),
    )),
    CardEffectRecipe("strike_red", (_fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 6, 9),)),
    CardEffectRecipe("defend_red", (_fixed(EffectPrimitive.GAIN_BLOCK, _P, 5, 8),)),
    CardEffectRecipe("bash", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 8, 10),
        _fixed(EffectPrimitive.APPLY_VULNERABLE, _T, 2, 3),
    )),
    CardEffectRecipe("bludgeon", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 32, 42),
    )),
    CardEffectRecipe("anger", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 6, 8),
        _generate(
            GeneratedCardSource.PLAYED_COPY,
            GeneratedCardDestination.DISCARD_PILE,
        ),
    )),
    CardEffectRecipe("wild_strike", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 12, 17),
        _generate(
            GeneratedCardSource.FIXED_BASE,
            GeneratedCardDestination.DRAW_PILE_RANDOM,
            "wound",
        ),
    )),
    CardEffectRecipe("power_through", (
        _generate(
            GeneratedCardSource.FIXED_BASE,
            GeneratedCardDestination.HAND,
            "wound", 2,
        ),
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 15, 20),
    )),
    CardEffectRecipe("reckless_charge", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 7, 10),
        _generate(
            GeneratedCardSource.FIXED_BASE,
            GeneratedCardDestination.DRAW_PILE_RANDOM,
            "dazed",
        ),
    )),
    CardEffectRecipe("immolate", (
        _fixed(EffectPrimitive.AREA_DAMAGE, _A, 21, 28),
        _generate(
            GeneratedCardSource.FIXED_BASE,
            GeneratedCardDestination.DISCARD_PILE,
            "burn",
        ),
    )),
    CardEffectRecipe("headbutt", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 9, 12),
        _move_selected(
            SelectedCardSource.DISCARD_PILE,
            SelectedCardDestination.DRAW_PILE_TOP,
        ),
    )),
    CardEffectRecipe("battle_trance", (
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 3, 4),
        _fixed(EffectPrimitive.APPLY_NO_DRAW, _P, 1, 1),
    )),
    CardEffectRecipe("warcry", (
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 2),
        _move_selected(SelectedCardSource.HAND, SelectedCardDestination.DRAW_PILE_TOP),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("thinking_ahead", (
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 2, 2),
        _move_selected(SelectedCardSource.HAND, SelectedCardDestination.DRAW_PILE_TOP),
    ), (CardTrait.BASE_EXHAUST,)),
    CardEffectRecipe("dropkick", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 5, 8),
        _conditional_fixed(
            PublicCondition.TARGET_HAS_VULNERABLE,
            EffectPrimitive.GAIN_ENERGY,
            _P,
            1,
            1,
        ),
        _conditional_fixed(
            PublicCondition.TARGET_HAS_VULNERABLE,
            EffectPrimitive.DRAW_CARDS,
            _P,
            1,
            1,
        ),
    )),
    CardEffectRecipe("seeing_red", (
        _fixed(EffectPrimitive.GAIN_ENERGY, _P, 2, 2),
    ), (CardTrait.EXHAUST, CardTrait.UPGRADE_COST_CHANGE)),
    CardEffectRecipe("bloodletting", (
        _fixed(EffectPrimitive.LOSE_HP, _P, 3, 3),
        _fixed(EffectPrimitive.GAIN_ENERGY, _P, 2, 3),
    )),
    CardEffectRecipe("offering", (
        _fixed(EffectPrimitive.LOSE_HP, _P, 6, 6),
        _fixed(EffectPrimitive.GAIN_ENERGY, _P, 2, 2),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 3, 5),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("flex", (
        _fixed(EffectPrimitive.GAIN_STRENGTH, _P, 2, 4),
        _fixed(EffectPrimitive.APPLY_LOSE_STRENGTH, _P, 2, 4),
    )),
    CardEffectRecipe("inflame", (
        _fixed(EffectPrimitive.GAIN_STRENGTH, _P, 2, 3),
    )),
    CardEffectRecipe("spot_weakness", (
        _conditional_fixed(
            PublicCondition.TARGET_ATTACK_INTENT,
            EffectPrimitive.GAIN_STRENGTH,
            _P,
            3,
            4,
        ),
    ), (CardTrait.TARGETED_ENEMY,)),
    CardEffectRecipe("pommel_strike", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 9, 10),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 2),
    )),
    CardEffectRecipe("shrug_it_off", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 8, 11),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 1, 1),
    )),
    CardEffectRecipe("true_grit", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 7, 9),
        _fixed(EffectPrimitive.EXHAUST_RANDOM_HAND, _P, 1, 0),
        _fixed(EffectPrimitive.EXHAUST_CHOSEN_HAND, _P, 0, 1),
    )),
    CardEffectRecipe("burning_pact", (
        _fixed(EffectPrimitive.EXHAUST_CHOSEN_HAND, _P, 1, 1),
        _fixed(EffectPrimitive.DRAW_CARDS, _P, 2, 3),
    )),
    CardEffectRecipe("ghostly_armor", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 10, 13),
    ), (CardTrait.ETHEREAL,)),
    CardEffectRecipe("impervious", (
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 30, 40),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("iron_wave", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 5, 7),
        _fixed(EffectPrimitive.GAIN_BLOCK, _P, 5, 7),
    )),
    CardEffectRecipe("cleave", (
        _fixed(EffectPrimitive.AREA_DAMAGE, _A, 8, 11),
    )),
    CardEffectRecipe("body_slam", (
        _dynamic(
            EffectPrimitive.CURRENT_BLOCK_DAMAGE,
            _T,
            AmountSource.CURRENT_BLOCK,
        ),
    ), (CardTrait.UPGRADE_COST_CHANGE,)),
    CardEffectRecipe("clash", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 14, 18),
    ), (CardTrait.PUBLIC_HAND_PLAYABILITY,)),
    CardEffectRecipe("clothesline", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 12, 14),
        _fixed(EffectPrimitive.APPLY_WEAK, _T, 2, 3),
    )),
    CardEffectRecipe("thunderclap", (
        _fixed(EffectPrimitive.AREA_DAMAGE, _A, 4, 7),
        _fixed(EffectPrimitive.APPLY_VULNERABLE, _A, 1, 1),
    )),
    CardEffectRecipe("uppercut", (
        _fixed(EffectPrimitive.DIRECT_DAMAGE, _T, 13, 13),
        _fixed(EffectPrimitive.APPLY_WEAK, _T, 1, 2),
        _fixed(EffectPrimitive.APPLY_VULNERABLE, _T, 1, 2),
    )),
    CardEffectRecipe("shockwave", (
        _fixed(EffectPrimitive.APPLY_WEAK, _A, 3, 5),
        _fixed(EffectPrimitive.APPLY_VULNERABLE, _A, 3, 5),
    ), (CardTrait.EXHAUST,)),
    CardEffectRecipe("intimidate", (
        _fixed(EffectPrimitive.APPLY_WEAK, _A, 1, 2),
    ), (CardTrait.EXHAUST,)),
))
