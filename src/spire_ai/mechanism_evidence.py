from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .card_recipes import (
    CARD_RECIPE_VERSION,
    DEFAULT_CARD_RECIPES,
    CardEffectRecipe,
    CardTrait,
    DrawTriggerEffect,
    ExhaustTriggerEffect,
    EffectPrimitive,
    HandExhaustFilter,
    UpgradeScope,
    StrengthRestoration,
    HandExhaustPayoff,
    PublicCondition,
)
from .encounter_recipes import (
    ENCOUNTER_RECIPE_VERSION,
    DEFAULT_ENCOUNTER_RECIPES,
    EncounterRecipe,
)
from .identifiers import canonical_identifier


MECHANISM_EVIDENCE_VERSION = "combat-mechanism-evidence-v61"


class EvidenceLayer(str, Enum):
    STATIC_METADATA = "static_metadata"
    PUBLIC_HISTORY = "public_history_safety"
    NATIVE_BEHAVIOR = "native_behavior"
    RULE_POLICY = "rule_policy_model"
    RESAMPLING = "resampling_contract"
    ROOT_ACCEPTANCE = "root_acceptance"
    ROLLOUT_QUALITY = "rollout_quality"


class EvidenceCapability(str, Enum):
    PUBLIC_HISTORY = "public_history"
    SEARCH = "search"
    COMBAT_RESAMPLING = "combat_resampling"
    ROLLOUT_QUALITY = "rollout_quality"


_CARD_SEARCH_LAYERS = (
    EvidenceLayer.STATIC_METADATA,
    EvidenceLayer.PUBLIC_HISTORY,
    EvidenceLayer.NATIVE_BEHAVIOR,
    EvidenceLayer.RULE_POLICY,
    EvidenceLayer.RESAMPLING,
    EvidenceLayer.ROOT_ACCEPTANCE,
)
_CAPABILITY_LAYERS = {
    EvidenceCapability.PUBLIC_HISTORY: (
        EvidenceLayer.STATIC_METADATA,
        EvidenceLayer.PUBLIC_HISTORY,
        EvidenceLayer.NATIVE_BEHAVIOR,
    ),
    EvidenceCapability.SEARCH: _CARD_SEARCH_LAYERS,
    EvidenceCapability.COMBAT_RESAMPLING: (
        EvidenceLayer.STATIC_METADATA,
        EvidenceLayer.NATIVE_BEHAVIOR,
        EvidenceLayer.RESAMPLING,
        EvidenceLayer.ROOT_ACCEPTANCE,
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    grade: str
    reference: str
    scope: str

    def as_dict(self) -> dict[str, str]:
        return {
            "grade": self.grade,
            "reference": self.reference,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class MechanismRecord:
    kind: str
    identifier: str
    mechanisms: tuple[str, ...]
    claims: Mapping[EvidenceLayer, EvidenceClaim]
    recipe_signature: tuple[str, ...] = ()
    composition: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"card", "encounter"}:
            raise ValueError(f"unsupported evidence kind: {self.kind}")
        if not self.identifier or not self.mechanisms:
            raise ValueError("mechanism evidence needs an identifier and mechanisms")
        if self.identifier != _canonical(self.kind, self.identifier):
            raise ValueError("mechanism evidence identifiers must be canonical")
        if len(set(self.mechanisms)) != len(self.mechanisms):
            raise ValueError("mechanism tags must be unique")
        if bool(self.recipe_signature) != bool(self.composition):
            raise ValueError("recipe signature and composition must be declared together")
        if any(not isinstance(layer, EvidenceLayer) for layer in self.claims):
            raise ValueError("mechanism evidence claims require EvidenceLayer keys")
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))
        object.__setattr__(self, "recipe_signature", tuple(self.recipe_signature))


class MechanismEvidenceRegistry:
    """Versioned scoped evidence; a claim is not global mechanic verification."""

    version = MECHANISM_EVIDENCE_VERSION

    def __init__(self, records: Iterable[MechanismRecord]) -> None:
        indexed: dict[tuple[str, str], MechanismRecord] = {}
        for record in records:
            key = (record.kind, record.identifier)
            if key in indexed:
                raise ValueError(f"duplicate mechanism evidence record: {key}")
            indexed[key] = record
        self._records = MappingProxyType(indexed)

    def record(self, kind: str, value: object) -> MechanismRecord | None:
        return self._records.get((kind, _canonical(kind, value)))

    def registered_ids(
        self, kind: str, capability: EvidenceCapability,
    ) -> frozenset[str]:
        return frozenset(
            identifier
            for (record_kind, identifier), record in self._records.items()
            if record_kind == kind
            and not self._missing(kind, record, capability, imported=True)
        )

    def supports(
        self, kind: str, value: object, capability: EvidenceCapability, *, imported: bool,
    ) -> bool:
        record = self.record(kind, value)
        return record is not None and not self._missing(
            kind, record, capability, imported=imported,
        )

    def first_missing(
        self, kind: str, value: object, capability: EvidenceCapability, *, imported: bool,
    ) -> EvidenceLayer | None:
        record = self.record(kind, value)
        missing = self._missing(kind, record, capability, imported=imported)
        return missing[0] if missing else None

    def status(self, kind: str, value: object, *, imported: bool) -> dict:
        identifier = _canonical(kind, value)
        record = self.record(kind, identifier)
        claims = self._claims(record, imported=imported)
        capabilities = {}
        for capability in EvidenceCapability:
            missing = self._missing(kind, record, capability, imported=imported)
            capabilities[capability.value] = {
                "supported": not missing,
                "missing_layers": [layer.value for layer in missing],
            }
        return {
            "registry_version": self.version,
            "kind": kind,
            "id": identifier,
            "mechanisms": list(record.mechanisms) if record is not None else [],
            "recipe": (
                {
                    "version": (
                        CARD_RECIPE_VERSION
                        if record.kind == "card" else ENCOUNTER_RECIPE_VERSION
                    ),
                    "composition": record.composition,
                    "signature": list(record.recipe_signature),
                }
                if record is not None and record.recipe_signature else None
            ),
            "layers": {
                layer.value: claims[layer].as_dict() if layer in claims else None
                for layer in EvidenceLayer
            },
            "capabilities": capabilities,
            "scope_note": (
                "claims are scoped evidence; they do not certify every interaction"
            ),
        }

    @staticmethod
    def _claims(
        record: MechanismRecord | None, *, imported: bool,
    ) -> dict[EvidenceLayer, EvidenceClaim]:
        claims = dict(record.claims) if record is not None else {}
        if imported and EvidenceLayer.STATIC_METADATA not in claims:
            claims[EvidenceLayer.STATIC_METADATA] = EvidenceClaim(
                "source_generated",
                "rule-catalog-v1",
                "static definition only; no executable behavior implied",
            )
        if not imported:
            claims.pop(EvidenceLayer.STATIC_METADATA, None)
        return claims

    def _missing(
        self, kind: str, record: MechanismRecord | None,
        capability: EvidenceCapability, *, imported: bool,
    ) -> tuple[EvidenceLayer, ...]:
        claims = self._claims(record, imported=imported)
        if capability is EvidenceCapability.ROLLOUT_QUALITY:
            base = (
                _CARD_SEARCH_LAYERS
                if kind == "card"
                else _CAPABILITY_LAYERS[EvidenceCapability.COMBAT_RESAMPLING]
            )
            required = base + (EvidenceLayer.ROLLOUT_QUALITY,)
        else:
            required = _CAPABILITY_LAYERS[capability]
        return tuple(layer for layer in required if layer not in claims)


def _canonical(kind: str, value: object) -> str:
    if kind not in {"card", "encounter"}:
        raise ValueError(f"unsupported evidence kind: {kind}")
    namespace = "monster" if kind == "encounter" else kind
    return canonical_identifier(value, namespace) or "<missing>"


def _claim(grade: str, reference: str, scope: str) -> EvidenceClaim:
    return EvidenceClaim(grade, reference, scope)


_CARD_STATIC = _claim(
    "source_generated", "rule-catalog-v1", "base and upgraded static definitions",
)
_CARD_HISTORY = _claim(
    "derived_composition",
    f"{CARD_RECIPE_VERSION}/combat-public-card-history-v58",
    "every primitive is history-safe and the ordered composition preserves public constraints on unknown draw order",
)
_LEGACY_CARD_NATIVE = _claim(
    "source_review", "pinned-native-card-source-audit-v2", "declared card effects only",
)
_BATCH_CARD_NATIVE = _claim(
    "native_fixture", "deterministic-card-effects-v1", "declared base/upgraded effects",
)
_STATUS_FAMILY_NATIVE = _claim(
    "native_fixture",
    "weak-vulnerable-card-recipes-v1",
    "six-card base/upgraded amounts, targets, ordering, Artifact interaction and exhaust movement",
)
_DRAW_FAMILY_NATIVE = _claim(
    "native_fixture",
    "deterministic-draw-card-recipes-v1",
    "Pommel Strike and Shrug It Off base/upgraded effects, draw limits, shuffle and Frozen Eye",
)
_EXHAUST_FAMILY_NATIVE = _claim(
    "native_fixture",
    "hand-exhaust-card-recipes-v1",
    "True Grit and Burning Pact base/upgraded random/selected exhaust and queued draw",
)
_GENERATED_CARD_FAMILY_NATIVE = _claim(
    "native_fixture",
    "generated-card-insertion-recipes-v1",
    "Anger copy generation and Wild Strike Wound insertion at base/upgraded values",
)
_SELECTED_TOPDECK_FAMILY_NATIVE = _claim(
    "native_fixture",
    "selected-topdeck-card-recipes-v1",
    "Headbutt base/upgraded damage and selected discard-to-known-draw-top movement",
)
_STATUS_INSERTION_FAMILY_NATIVE = _claim(
    "native_fixture",
    "status-insertion-card-recipes-v1",
    "Power Through, Reckless Charge and Immolate base/upgraded effects, hand overflow, status lifecycle, exhaust interactions and terminal cleanup",
)
_CONDITIONAL_DRAW_ENERGY_FAMILY_NATIVE = _claim(
    "native_fixture",
    "conditional-draw-energy-card-recipes-v1",
    "Dropkick base/upgraded damage and public Vulnerable-gated energy/draw branches",
)
_ENERGY_SELF_HP_FAMILY_NATIVE = _claim(
    "native_fixture",
    "energy-self-hp-card-recipes-v1",
    "Seeing Red, Bloodletting and Offering base/upgraded ordered effects, lethal interruption, Buffer, Intangible, Tungsten Rod and Rupture interaction",
)
_STRENGTH_FAMILY_NATIVE = _claim(
    "native_fixture",
    "self-strength-card-recipes-v1",
    "Flex, Inflame and Spot Weakness base/upgraded Strength, temporary loss, public attack-intent predicate and Artifact interaction",
)
_CARD_RULE = _claim(
    "derived_composition",
    f"{CARD_RECIPE_VERSION}/combat-recipe-rule-v48",
    "ordered public primitive scorer; native legal actions remain authoritative",
)
_CARD_RESAMPLING = _claim(
    "derived_composition",
    f"{CARD_RECIPE_VERSION}/public-combat-resampling-v1",
    "history-safe recipe effects preserve public pile membership and any visible "
    "order during continuation sampling",
)
_CORPUS_CARD_ROOT = _claim(
    "measured", "public-run-combat-root-audit-v2", "replayed fixed development root",
)
_STATUS_FAMILY_ROOT = _claim(
    "native_fixture",
    "weak-vulnerable-card-recipes-v1",
    "fixed executable decision root for every family recipe and upgrade variant",
)
_DRAW_FAMILY_ROOT = _claim(
    "native_fixture",
    "deterministic-draw-card-recipes-v1",
    "fixed executable roots before and after an observed draw-card action",
)
_EXHAUST_FAMILY_ROOT = _claim(
    "native_fixture",
    "hand-exhaust-card-recipes-v1",
    "fixed executable roots before and after random or selected hand exhaustion",
)
_GENERATED_CARD_FAMILY_ROOT = _claim(
    "native_fixture",
    "generated-card-insertion-recipes-v1",
    "fixed executable roots before and after discard/draw-pile card generation",
)
_SELECTED_TOPDECK_FAMILY_ROOT = _claim(
    "native_fixture",
    "selected-topdeck-card-recipes-v1",
    "fixed executable roots before and after Headbutt selection, draw and resampling",
)
_STATUS_INSERTION_FAMILY_ROOT = _claim(
    "native_fixture",
    "status-insertion-card-recipes-v1",
    "fixed executable roots before/after generation into hand, draw and discard; hidden-order resampling and Frozen Eye",
)
_CONDITIONAL_DRAW_ENERGY_FAMILY_ROOT = _claim(
    "native_fixture",
    "conditional-draw-energy-card-recipes-v1",
    "fixed executable roots for both public Vulnerable predicate outcomes",
)
_ENERGY_SELF_HP_FAMILY_ROOT = _claim(
    "native_fixture",
    "energy-self-hp-card-recipes-v1",
    "fixed executable roots for energy generation, ordered self-HP loss and low-HP rejection",
)
_STRENGTH_FAMILY_ROOT = _claim(
    "native_fixture",
    "self-strength-card-recipes-v1",
    "fixed executable roots for persistent and temporary Strength and both visible intent branches",
)
_CARD_QUALITY = _claim(
    "scoped_measurement",
    "combat-survival-benchmark-v1",
    "supported first-combat fixtures; not a population estimate",
)
_BATCH_CARD_QUALITY = _claim(
    "scoped_measurement",
    "deterministic-card-batch-development-v1",
    "three fixed later-combat development roots; comparison, not superiority",
)


_DERIVED_LAYERS = frozenset({
    EvidenceLayer.PUBLIC_HISTORY,
    EvidenceLayer.RULE_POLICY,
    EvidenceLayer.RESAMPLING,
})
_PRIMITIVE_EVIDENCE = MappingProxyType({
    EffectPrimitive.SELECT_BOTTOM_FREE: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_INTANGIBLE: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_ARTIFACT: _DERIVED_LAYERS,
    EffectPrimitive.APPLY_NO_BLOCK: _DERIVED_LAYERS,
    EffectPrimitive.EXHAUST_UP_TO_HAND: _DERIVED_LAYERS,
    EffectPrimitive.RETRIEVE_RANDOM_ATTACKS: _DERIVED_LAYERS,
    EffectPrimitive.GENERATE_X_HAND: _DERIVED_LAYERS,
    EffectPrimitive.AUTOPLAY_TOP_CARD: _DERIVED_LAYERS,
    EffectPrimitive.MODIFY_HAND_COST: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_CORRUPTION: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_ATTACK_REPEAT: _DERIVED_LAYERS,
    EffectPrimitive.APPLY_NO_DRAW: _DERIVED_LAYERS,
    EffectPrimitive.UPGRADE_CARDS: _DERIVED_LAYERS,
    EffectPrimitive.EXHAUST_HAND_GROUP: _DERIVED_LAYERS,
    EffectPrimitive.ENEMY_STRENGTH_LOSS: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_CURRENT_STRENGTH: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_EVENT_TRIGGER: _DERIVED_LAYERS,
    EffectPrimitive.ON_HP_LOSS_COST_REDUCTION: _DERIVED_LAYERS,
    EffectPrimitive.ON_EXHAUST_ENERGY: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_TURN_TRIGGER: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_DELAYED_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.SELF_VULNERABLE: _DERIVED_LAYERS,
    EffectPrimitive.RETAIN_BLOCK: _DERIVED_LAYERS,
    EffectPrimitive.CURRENT_BLOCK_GAIN: _DERIVED_LAYERS,
    EffectPrimitive.END_TURN_BLOCK: _DERIVED_LAYERS,
    EffectPrimitive.TEMP_RETALIATION: _DERIVED_LAYERS,
    EffectPrimitive.HEAL: _DERIVED_LAYERS,
    EffectPrimitive.AREA_LIFESTEAL: _DERIVED_LAYERS,
    EffectPrimitive.FATAL_ATTACK: _DERIVED_LAYERS,
    EffectPrimitive.ATTACK_SEQUENCE: _DERIVED_LAYERS,
    EffectPrimitive.DYNAMIC_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.GROW_PLAYED_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.DIRECT_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.AREA_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.CURRENT_BLOCK_DAMAGE: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_BLOCK: _DERIVED_LAYERS,
    EffectPrimitive.LOSE_HP: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_ENERGY: _DERIVED_LAYERS,
    EffectPrimitive.GAIN_STRENGTH: _DERIVED_LAYERS,
    EffectPrimitive.APPLY_LOSE_STRENGTH: _DERIVED_LAYERS,
    EffectPrimitive.DRAW_CARDS: _DERIVED_LAYERS,
    EffectPrimitive.SHUFFLE_DISCARD_INTO_DRAW: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_DRAW_TRIGGER: _DERIVED_LAYERS,
    EffectPrimitive.REGISTER_EXHAUST_TRIGGER: _DERIVED_LAYERS,
    EffectPrimitive.EXHAUST_RANDOM_HAND: _DERIVED_LAYERS,
    EffectPrimitive.EXHAUST_CHOSEN_HAND: _DERIVED_LAYERS,
    EffectPrimitive.GENERATE_CARD: _DERIVED_LAYERS,
    EffectPrimitive.GENERATE_RANDOM_HAND: _DERIVED_LAYERS,
    EffectPrimitive.GENERATE_RANDOM_DRAW: _DERIVED_LAYERS,
    EffectPrimitive.GENERATE_CHOICE_HAND: _DERIVED_LAYERS,
    EffectPrimitive.MOVE_SELECTED_CARD: _DERIVED_LAYERS,
    EffectPrimitive.COPY_SELECTED_CARD: _DERIVED_LAYERS,
    EffectPrimitive.APPLY_WEAK: _DERIVED_LAYERS,
    EffectPrimitive.APPLY_VULNERABLE: _DERIVED_LAYERS,
})
_CONDITION_EVIDENCE = MappingProxyType({
    PublicCondition.HAND_HAS_NO_ATTACK: _DERIVED_LAYERS,
    PublicCondition.TARGET_HAS_VULNERABLE: _DERIVED_LAYERS,
    PublicCondition.TARGET_ATTACK_INTENT: _DERIVED_LAYERS,
})
_DRAW_TRIGGER_EVIDENCE = MappingProxyType({
    ("evolve", ("STATUS",), DrawTriggerEffect.DRAW): _DERIVED_LAYERS,
    ("fire_breathing", ("STATUS", "CURSE"), DrawTriggerEffect.NON_ATTACK_AREA_DAMAGE): _DERIVED_LAYERS,
})
_EXHAUST_TRIGGER_EVIDENCE = MappingProxyType({
    ("feel_no_pain", ExhaustTriggerEffect.NON_CARD_BLOCK): _DERIVED_LAYERS,
    ("dark_embrace", ExhaustTriggerEffect.DRAW): _DERIVED_LAYERS,
})
_TRAIT_EVIDENCE = MappingProxyType({
    CardTrait.PLAYABLE_CURSE: _DERIVED_LAYERS,
    CardTrait.UNPLAYABLE_CURSE: _DERIVED_LAYERS,
    CardTrait.UNPLAYABLE_STATUS: _DERIVED_LAYERS,
    CardTrait.INNATE: _DERIVED_LAYERS,
    CardTrait.UPGRADE_INNATE: _DERIVED_LAYERS,
    CardTrait.ETHEREAL: _DERIVED_LAYERS,
    CardTrait.BASE_ETHEREAL: _DERIVED_LAYERS,
    CardTrait.EXHAUST: _DERIVED_LAYERS,
    CardTrait.BASE_EXHAUST: _DERIVED_LAYERS,
    CardTrait.PUBLIC_HAND_PLAYABILITY: _DERIVED_LAYERS,
    CardTrait.UPGRADE_COST_CHANGE: _DERIVED_LAYERS,
    CardTrait.TARGETED_ENEMY: _DERIVED_LAYERS,
})
_COMPOSITION_EVIDENCE = MappingProxyType({
    "passive_curse": _DERIVED_LAYERS,
    "passive_status": _DERIVED_LAYERS,
    "exhaust_only": _DERIVED_LAYERS,
    "single": _DERIVED_LAYERS,
    "ordered_sequence": _DERIVED_LAYERS,
    "conditional_sequence": _DERIVED_LAYERS,
})

_DETERMINISTIC_BATCH_IDS = frozenset({
    "body_slam", "clash", "cleave", "iron_wave",
})
_STATUS_FAMILY_IDS = frozenset({
    "bash", "clothesline", "intimidate", "shockwave", "thunderclap", "uppercut",
})
_DRAW_FAMILY_IDS = frozenset({"pommel_strike", "shrug_it_off"})
_EXHAUST_FAMILY_IDS = frozenset({"burning_pact", "true_grit"})
_GENERATED_CARD_FAMILY_IDS = frozenset({"anger", "wild_strike"})
_STATUS_INSERTION_FAMILY_IDS = frozenset({"power_through", "reckless_charge", "immolate"})
_EXHAUST_ONLY_IDS = frozenset({"slimed"})
_DRAW_TRIGGER_IDS = frozenset({"evolve", "fire_breathing"})
_EXHAUST_TRIGGER_IDS = frozenset({"feel_no_pain", "dark_embrace"})
_EXHAUST_TRIGGER_NATIVE = _claim(
    "native_fixture", "exhaust-trigger-card-recipes-v1",
    "base/upgraded costs, non-card block, exhaust draws, stacking, limits, nested triggers, acquisition order, cloning and public resampling",
)
_DRAW_TRIGGER_NATIVE = _claim(
    "native_fixture", "draw-trigger-card-recipes-v1",
    "base/upgraded draw filters, stacking, No Draw/hand limits, chain/shuffle, non-attack damage, lethal acquisition order and cloning/resampling",
)
_EXHAUST_ONLY_NATIVE = _claim(
    "native_fixture", "exhaust-only-card-recipes-v1",
    "Slimed cost, legality, exact self-exhaust identity, no direct damage/block, unplayed discard, exhaust callbacks and public resampling",
)
_EXHAUST_ONLY_ROOT = _claim(
    "native_fixture", "exhaust-only-card-recipes-v1",
    "fixed resolved roots before/after Slimed self-exhaust, public hand/exhaust membership and hidden/Frozen Eye order",
)
_SELECTED_TOPDECK_FAMILY_IDS = frozenset({"headbutt"})
_CONDITIONAL_DRAW_ENERGY_FAMILY_IDS = frozenset({"dropkick"})
_ENERGY_SELF_HP_FAMILY_IDS = frozenset({
    "bloodletting", "offering", "seeing_red",
})
_STRENGTH_FAMILY_IDS = frozenset({"flex", "inflame", "spot_weakness"})
_CROSS_COLOR_IDS = frozenset({
    "swift_strike", "good_instincts", "flash_of_steel", "finesse",
    "master_of_strategy", "carnage", "hemokinesis", "jax",
})
_CROSS_COLOR_NATIVE = _claim(
    "native_fixture", "cross-color-card-recipes-v1",
    "eight-card base/upgrade effects, costs, discard/exhaust/ethereal lifecycle, "
    "draw limits/shuffle, HP-loss interruption and selected public power interactions",
)
_CROSS_COLOR_ROOT = _claim(
    "native_fixture", "cross-color-card-recipes-v1",
    "fixed roots before/after each card; hidden-order invariance and Frozen Eye order; "
    "not naturally collected runs or full interaction verification",
)


_TURN_TRIGGER_SIGNATURES = frozenset({
    "mayhem:start_of_turn:deferred_top_card_autoplay:1/1:per_stack_wrapper:target_at_wrapper:append_after_queued_draws:draw_top:shuffle_discard_if_empty:fifo_card_queue:free:queued_energy_on_use:not_draw:no_forced_exhaust:rejected_play_still_settles",
    "magnetism:start_of_turn:generate_random_cards_into_hand:1/1:colorless_pool_no_healing:base_copy:normal_cost:per_power_stack:before_normal_draw:sample_at_turn_trigger:with_replacement:fresh_identity:ordered_hand_overflow_discard:not_draw",
    "brutality:start_of_turn_post_draw:draw_cards:1/1>self_hp_loss:1/1",
    "combust:end_of_player_turn:self_hp_loss:1/1>non_attack_area_damage:5/7",
    "demon_form:start_of_turn_post_draw:gain_strength:2/3",
    "berserk:start_of_turn:gain_energy:1/1",
})

_HAND_EXHAUST_COMBINATIONS = frozenset({
    (HandExhaustFilter.NON_ATTACK, HandExhaustPayoff.NONE),
    (HandExhaustFilter.NON_ATTACK, HandExhaustPayoff.BLOCK),
    (HandExhaustFilter.ALL, HandExhaustPayoff.ATTACK),
})

_EVENT_TRIGGER_SIGNATURES = frozenset({
    "panache:every_fifth_card_play_reset_each_turn:all_living_enemies_non_attack_damage:combat",
    "sadistic:successful_enemy_debuff_packet_excluding_shackled:debuff_target_non_attack_damage:combat",
    "rupture:positive_self_sourced_hp_loss:strength:combat",
    "rage:attack_card_played_once:non_card_block:until_end_of_turn",
    "juggernaut:positive_block_gain_including_at_cap:random_living_enemy_non_attack_damage:combat",
})


def _recipe_derives(
    recipe: CardEffectRecipe,
    layer: EvidenceLayer,
) -> bool:
    return (
        layer in _COMPOSITION_EVIDENCE.get(recipe.composition, ())
        and (recipe.self_copy is None or recipe.self_copy.signature in {
            "after_exhaust_callbacks:fresh_base_copy_to_hand_overflow_discard:1",
            "end_turn_hand_snapshot_no_use:fresh_stat_equivalent_copy_to_draw_top:1",
        })
        and (recipe.resident_rule is None or recipe.resident_rule.signature in {
            "other_card_use_queue_top_self_hp_loss_per_resident_identity:1",
            "hand_resident_limit_all_successful_turn_plays_including_auto_repeat:3",
        })
        and all(effect.signature in {
            "on_actual_draw_queue_bottom:lose_energy_floor_zero:1/1",
            "end_turn_in_hand_no_use_triggers:self_sourced_blockable_non_attack_damage:2/4",
            "end_turn_in_hand_no_use_triggers:self_sourced_blockable_non_attack_damage:2/2",
            "end_turn_in_hand_no_use_triggers:self_weak_artifact_blockable_fresh_skip_first_round_decay:1/1",
            "end_turn_in_hand_no_use_triggers:self_frail_artifact_blockable_fresh_skip_first_round_decay:1/1",
            "end_turn_in_hand_no_use_triggers:self_sourced_block_bypassing_hp_loss:1/1:amount_source=hand_count_including_self_at_end_turn_callback",
        } for effect in recipe.intrinsic_effects)
        and all(step.combat_upgrade is None or
                (step.combat_upgrade.base_scope, step.combat_upgrade.upgraded_scope) in {
                    (UpgradeScope.CHOSEN_HAND, UpgradeScope.ALL_HAND),
                    (UpgradeScope.ALL_COMBAT_PILES, UpgradeScope.ALL_COMBAT_PILES),
                } for step in recipe.effects)
        and all(step.strength_reduction is None or step.strength_reduction.restoration in {
            StrengthRestoration.NONE, StrengthRestoration.ENEMY_TURN_END,
        } for step in recipe.effects)
        and all(step.event_trigger is None or step.event_trigger.signature in _EVENT_TRIGGER_SIGNATURES
                for step in recipe.effects)
        and all(step.hand_exhaust is None or (step.hand_exhaust.filter, step.hand_exhaust.payoff)
                in _HAND_EXHAUST_COMBINATIONS for step in recipe.effects)
        and all(step.turn_trigger is None or step.turn_trigger.signature in _TURN_TRIGGER_SIGNATURES
                for step in recipe.effects)
        and all(step.delayed_damage is None or step.delayed_damage.signature ==
                "3:end_of_player_turn:independent:one_shot_non_attack_area:counts_played_turn:survives_source_removal"
                for step in recipe.effects)
        and all(
            layer in _PRIMITIVE_EVIDENCE.get(step.primitive, ())
            for step in recipe.effects
        )
        and all(
            step.condition is None
            or layer in _CONDITION_EVIDENCE.get(step.condition, ())
            for step in recipe.effects
        )
        and all(layer in _TRAIT_EVIDENCE.get(trait, ()) for trait in recipe.traits)
        and all(
            step.draw_trigger is None or layer in _DRAW_TRIGGER_EVIDENCE.get((
                step.draw_trigger.power_id, step.draw_trigger.card_types, step.draw_trigger.effect,
            ), ())
            for step in recipe.effects
        )
        and all(
            step.exhaust_trigger is None or layer in _EXHAUST_TRIGGER_EVIDENCE.get((
                step.exhaust_trigger.power_id, step.exhaust_trigger.effect,
            ), ())
            for step in recipe.effects
        )
    )


def _recipe_card(recipe: CardEffectRecipe) -> MechanismRecord:
    identifier = recipe.identifier
    native = _LEGACY_CARD_NATIVE
    root = _CORPUS_CARD_ROOT
    quality: EvidenceClaim | None = _CARD_QUALITY
    if identifier in {"apparition", "ritual_dagger"}:
        native = _claim("native_fixture", "event-special-cards-v1",
                        "base/plus costs and exhaustion, base-only Ethereal, Intangible packets/block/round decay/Buffer/Rod/Rupture, Dagger fatal/minion/blocked/terminal/generated-master identity and attack callbacks")
        root = _claim("native_fixture", "event-special-cards-v1",
                      "paired hidden/Frozen Eye pre/post roots; public mutable base damage required in all piles; no live parity, overflow or arbitrary relic ordering guarantee")
        quality = None
    if identifier in {"necronomicurse", "pride"}:
        native = _claim("native_fixture", "replicating-curse-v1",
                        "fresh base exhaust copies and stat-equivalent end-hand top copies; callbacks, overflow, Candle, opening hand and actual turns")
        root = _claim("native_fixture", "replicating-curse-v1",
                      "tracked public top suffix and bottom constraints, public identities and paired hidden-order sampling; missing Pride tracking refused; not live or full-run removal parity")
        quality = None
    if identifier in {"pain", "normality"}:
        native = _claim("native_fixture", "hand-resident-curse-v1",
                        "Pain identity/source/packets and Normality manual/automatic/repeated use limits; draw/exhaust/turn reset and public counter resampling")
        root = _claim("native_fixture", "hand-resident-curse-v1",
                      "Normality requires schema1 public actual turn-play count in every pile; missing/invalid state refused; no live curse parity or optimal disposal claim")
        quality = None
    if identifier in {"decay", "doubt", "shame", "regret"}:
        native = _claim("native_fixture", "end-hand-curse-cards-v1",
                        "end-hand no-use callbacks, blockable damage vs HP loss, Regret hand snapshot, fresh/stacked debuff decay, Artifact/Ginger/Turnip, Candle/rejected autoplay, Buffer/Rod/Rupture")
        root = _claim("native_fixture", "end-hand-curse-cards-v1",
                      "public identities and paired hidden/Frozen Eye roots; not live curse parity, optimal cleanup policy or arbitrary relic ordering")
        quality = None
    if identifier in {"injury", "clumsy", "writhe", "ascenders_bane", "curse_of_the_bell", "parasite"}:
        native = _claim("native_fixture", "basic-curse-cards-v1",
                        "unplayability/Blue Candle, Ethereal/Innate, Fire Breathing not Evolve, discard/exhaust/HP loss, Buffer/Rod/Rupture and rejected autoplay; removal fixture is separate from combat exhaust")
        root = _claim("native_fixture", "basic-curse-cards-v1",
                      "public identities and paired hidden/Frozen Eye roots; no live parity, full run policy or arbitrary relic-order guarantee")
        quality = None
    if identifier in {"wound", "dazed", "burn", "void"}:
        native = _claim("native_fixture", "passive-status-cards-v1",
                        "unplayability/Medical Kit, blockable Burn2/4, Ethereal exhaust callbacks, actual-draw filters, Evolve/Fire Breathing, Void queued loss with Sundial; not all relic ordering")
        root = _claim("native_fixture", "passive-status-cards-v1",
                      "resolved roots preserve public status identities/variants and hidden-order invariance/Frozen Eye; not live-game parity or rollout quality")
        quality = None
    if identifier == "the_bomb":
        native = _claim("native_fixture", "the-bomb-delayed-v1",
                        "base/plus independent non-attack packets, third-end timing, block/Intangible, source exhaustion, Combust relative order; retained packet correction")
        root = _claim("native_fixture", "the-bomb-delayed-v1",
                      "96 full-turn paired roots, actual enemy turns and Frozen Eye; public schema/completeness/readiness and companion gates; not arbitrary relic interactions, live-game parity or rollout quality")
        quality = None
    if identifier in {"panache", "sadistic_nature"}:
        native = _claim("native_fixture", "play-debuff-triggers-v1",
                        "power registration/stacking, fifth-play reset and turn reset, repeated/multi-hit plays, successful debuff packets and Artifact, representative non-attack interactions; local Panache counter fix")
        root = _claim("native_fixture", "play-debuff-triggers-v1",
                      "explicit public counter/damage, resolved paired roots and Frozen Eye; missing active Panache fields fail closed, not live-game parity or all trigger-order evidence")
        quality = None
    if identifier in {"panacea", "panic_button"}:
        native = _claim("native_fixture", "artifact-block-restriction-v1",
                        "base/plus charges, block-before-debuff, one Artifact per packet, stacking/two round expiration, card modifiers and representative non-card block/exhaust callbacks")
        root = _claim("native_fixture", "artifact-block-restriction-v1",
                      "resolved paired pre/post roots, Frozen Eye and public-state scoring; not all power/relic ordering, live-game parity or rollout quality")
        quality = None
    if identifier in {"purity", "violence"}:
        native = _claim("native_fixture", "bulk-exhaust-retrieval-v1",
                        "zero-to-limit subsets, exhaustion callbacks, random attack movement, short supply identity conservation, overflow/No Draw; local Violence cleanup fix")
        root = _claim("native_fixture", "bulk-exhaust-retrieval-v1",
                      "bounded public subset choices and resolved paired roots, mutable identities and Frozen Eye; pending selection uses rule fallback, not live multi-selection or exhaustive relic evidence")
        quality = None
    if identifier == "mayhem":
        native = _claim("native_fixture", "mayhem-turn-autoplay-v1",
                        "base/upgraded registration, deferred normal/post-draw timing, FIFO stacks, target timing, ordinary and rejected lifecycle, X energy, selected exhaust callbacks and suspended selections")
        root = _claim("native_fixture", "mayhem-turn-autoplay-v1",
                      "paired sampled turn roots, Frozen Eye and known bottom/top constraints; public stack/candidate closure gates and turn-boundary top invalidation; no fresh live or all-relic/quality claim")
        quality = None
    if identifier == "forethought":
        native = _claim("native_fixture", "forethought-bottom-free-v1",
                        "ordered optional selections, identity conservation, positive-combat-cost one-use free flag and actual redraw/play; local upgraded-selection fix")
        root = _claim("native_fixture", "forethought-bottom-free-v1",
                      "known bottom/top constraints and public free flag, bounded ordered choices and resolved sampled roots; random insertion into known bottom refused; no live multi-selection or strategic-quality claim")
        quality = None
    if identifier in {"dramatic_entrance", "mind_blast"}:
        native = _claim("native_fixture", "opening-attack-recipes-v1",
                        "base/upgraded innate opening attacks, area and draw-pile-count damage, modifiers, lifecycle and selected callbacks")
        root = _claim("native_fixture", "opening-attack-recipes-v1",
                      "public pile-count scoring, paired resolved pre/post roots and Frozen Eye; not arbitrary start-of-combat/relic ordering or live-game quality evidence")
        quality = None
    if identifier in {"deep_breath", "impatience"}:
        native = _claim("native_fixture", "shuffle-conditional-draw-v1",
                        "base/upgraded shuffle-before-draw and queued no-attack condition; No Draw/capacity/empty piles, representative shuffle and draw callbacks")
        root = _claim("native_fixture", "shuffle-conditional-draw-v1",
                      "public hand scoring, known-top invalidation, resolved paired roots with Frozen Eye; not arbitrary relic ordering, live-game or quality evidence")
        quality = None
    if identifier in {"blind", "trip"}:
        native = _claim("native_fixture", "upgrade-target-scope-v1",
                        "base targeted/upgraded area Weak or Vulnerable, fixed two stacks, zero cost, independent Artifact, stacking/duration, dead-target exclusion and representative skill/relic callbacks")
        root = _claim("native_fixture", "upgrade-target-scope-v1",
                      "variant-aware public scoring and legal actions; paired pre/post-play sampled roots; not all relic interactions or live-game/rollout-quality evidence")
        quality = None
    if identifier in {"transmutation", "magnetism"}:
        native = _claim("native_fixture", "colorless-x-turn-generation-v1",
                        "Transmutation X/Chemical X/free count, base/upgraded temporary costs, ordered overflow and reset; Magnetism registration/stacking/start-before-draw generation, normal-cost base copies and representative callbacks")
        root = _claim("native_fixture", "colorless-x-turn-generation-v1",
                      "public pool scoring, active-power scope guards, resolved roots before/after generation and paired hidden-order worlds; not full relic coverage or live-game evidence")
        quality = None
    if identifier == "discovery":
        native = _claim("native_fixture", "discovery-generation-v1",
                        "three distinct Ironclad candidates; choose one base copy, temporary zero except X, overflow/reset, base-only intrinsic exhaust, representative callbacks and prior Blood for Blood damage discount")
        root = _claim("native_fixture", "discovery-generation-v1",
                      "public structured candidates and one-copy choice; pending selections use rule fallback, only resolved roots resample; paired hidden worlds; no live frame-dependent RNG parity or potion/Master Reality coverage")
        quality = None
    if identifier in {"chrysalis", "metamorphosis"}:
        native = _claim("native_fixture", "random-draw-generation-v1",
                        "Ironclad type pools, 3/5 base copies, positive combat-cost zero with X retained, use-time selection, random insertion below existing top, No Draw/Corruption and exhaust callbacks")
        root = _claim("native_fixture", "random-draw-generation-v1",
                      "paired hidden worlds and resolved roots preserve generated public multiset/costs and known top; explicit Ironclad required; Master Reality and future unreviewed effects remain guarded; no all-relic or live-game claim")
        quality = None
    if identifier in {"infernal_blade", "jack_of_all_trades"}:
        native = _claim("native_fixture", "random-hand-generation-v1",
                        "Ironclad attack and colorless pools, variants, costs/reset, identity/order/overflow, no-draw and exhaust callbacks; generated Blood for Blood inherits prior HP-loss events; local cost and generation-order corrections")
        root = _claim("native_fixture", "random-hand-generation-v1",
                      "paired hidden worlds and resolved roots preserve public generated identities/costs; Infernal Blade requires public Ironclad and times_damaged; future unreviewed cards retain existing guards; not arbitrary cross-character/pool-modifying or all relic combinations")
        quality = None
    if identifier == "havoc":
        native = _claim("native_fixture", "top-card-autoplay-recipes-v1",
                        "top Attack/Skill/Power/X, empty/shuffle, No Draw/full hand, nested queue, rejected lifecycle and representative callbacks; local rejected-autoplay correction")
        root = _claim("native_fixture", "top-card-autoplay-recipes-v1",
                      "paired hidden worlds, known-top/Frozen Eye and resolved post-play roots; suspended selections rejected; unreviewed automatic candidates fail closed; no fresh live-game coverage")
        quality = None
    if identifier == "double_tap":
        native = _claim("native_fixture", "queued-attack-repeat-card-recipes-v1",
                        "charges/expiry, same-identity growth, X energy, Nunchaku regression, lifecycle and representative callbacks; local autoplay-X fix")
        root = _claim("native_fixture", "queued-attack-repeat-card-recipes-v1",
                      "resolved pre/post repeat roots and paired hidden worlds; public charge count preserved; no fresh live-game or exhaustive queue/relic evidence")
        quality = None
    if identifier == "dual_wield":
        native = _claim("native_fixture", "selected-copy-card-recipes-v1",
                        "base/plus, automatic/manual identity paths, stat/cost copies, overflow, counters and no-draw controls")
        root = _claim("native_fixture", "selected-copy-card-recipes-v1",
                      "resolved pre/post roots and paired public worlds; pending choice uses public candidates/count and rule fallback; no live-game or all-relic coverage")
        quality = None
    if identifier in {"exhume", "secret_technique", "secret_weapon"}:
        native = _claim("native_fixture", "pile-retrieval-card-recipes-v1",
                        "six variants, eligibility/empty/full-hand controls, no draw callbacks, lifecycle and mutable identity")
        root = _claim("native_fixture", "pile-retrieval-card-recipes-v1",
                      "resolved pre/post roots, canonical public selection candidates and private native index mapping; "
                      "pending choice uses rule fallback, not mid-queue resampling; no fresh live-game coverage")
        quality = None
    if identifier in {"perfected_strike", "rampage", "searing_blow"}:
        native = _claim("native_fixture", "dynamic-damage-card-recipes-v1",
                        "Strike pile counts, per-card combat growth, repeated upgrades and selected damage modifiers")
        root = _claim("native_fixture", "dynamic-damage-card-recipes-v1",
                      "resolved pre/post roots retain growth and upgrades; hidden-order independence and Frozen Eye; "
                      "requires public Rampage base_damage, not arbitrary autoplay/copy combinations")
        quality = None
    if identifier in {"corruption", "madness", "enlightenment"}:
        native = _claim("native_fixture", "combat-cost-card-recipes-v1",
                        "six variants; independent combat/turn costs, random eligibility, duration, "
                        "four-pile Corruption initialization, skill exhaustion and selected upgrade/trigger controls")
        root = _claim("native_fixture", "combat-cost-card-recipes-v1",
                      "resolved pre/post roots preserve mutable costs; hidden-order independence and Frozen Eye; "
                      "not arbitrary autoplay/copies/generated-card/relic combinations")
        quality = None
    if identifier in {"battle_trance", "warcry", "thinking_ahead"}:
        native = _claim("native_fixture", "draw-restriction-topdeck-card-recipes-v1",
                        "six variants; draw then No Draw/Artifact, hand selection including drawn cards, "
                        "zero/single/multiple candidates, exhaustion and Dark Embrace redraw")
        root = _claim("native_fixture", "draw-restriction-topdeck-card-recipes-v1",
                      "resolved roots, known selected top, hidden-order independence and Frozen Eye; "
                      "pending selection uses public rule fallback, not mid-queue sampling")
        quality = None
    if identifier in {"armaments", "apotheosis"}:
        native = _claim("native_fixture", "combat-upgrade-card-recipes-v1",
                        "four variants, eligibility/selection, hand and four-pile scope, identity, "
                        "repeatable upgrades, temporary and Blood for Blood costs, Burn control")
        root = _claim("native_fixture", "combat-upgrade-card-recipes-v1",
                      "resolved pre/post roots with hidden-order independence and Frozen Eye; "
                      "pending selection uses public rule fallback, not mid-queue resampling")
        quality = None
    if identifier in {"disarm", "limit_break", "dark_shackles"}:
        native = _claim("native_fixture", "strength-modification-card-recipes-v1",
                        "six variants, signed Strength, Artifact, conditional enemy-turn restoration, "
                        "variant exhaustion, Flex/Inflame/Rupture composition and Strength bounds")
        root = _claim("native_fixture", "strength-modification-card-recipes-v1",
                      "fixed pre/post roots, hidden-order/RNG resampling, pending restoration and Frozen Eye; "
                      "not all callback ordering or fresh live-game differential coverage")
        quality = None
    if identifier in {"rupture", "rage", "juggernaut", "blood_for_blood"}:
        native = _claim("native_fixture", "event-trigger-card-recipes-v1",
                        "eight variants, source-filtered HP loss, once-per-play Rage, positive block events, "
                        "random non-attack damage and mutable cost/upgrade regressions")
        root = _claim("native_fixture", "event-trigger-card-recipes-v1",
                      "fixed pre/post roots, paired hidden RNG/order resampling and Frozen Eye; "
                      "not generated Blood for Blood initialization or all callback interactions")
        quality = None
    if identifier in {"second_wind", "sever_soul", "fiend_fire", "sentinel"}:
        native = _claim("native_fixture", "whole-hand-exhaust-recipes-v1",
                        "eight variants, hand filters/count snapshots, per-card block/multihit, "
                        "selected exhaust power callbacks, Sentinel refund and Sever Soul/Ashes ordering")
        root = _claim("native_fixture", "whole-hand-exhaust-recipes-v1",
                      "fixed roots before/after each card, paired hidden-order resampling and Frozen Eye; "
                      "not all relic combinations or live-game differential coverage")
        quality = None
    if identifier in {"brutality", "combust", "demon_form", "berserk"}:
        native = _claim("native_fixture", "turn-trigger-card-recipes-v1",
                        "eight variants, costs, turn timing, stacking, Brutality callback order/Rupture, "
                        "Combust non-attack damage, Artifact, Tungsten Rod and Ice Cream")
        root = _claim("native_fixture", "turn-trigger-card-recipes-v1",
                      "fixed roots before/after power play and first turn boundary; "
                      "paired hidden-order resampling and Frozen Eye; not all power interactions")
        quality = None
    if identifier in _DETERMINISTIC_BATCH_IDS:
        native = _BATCH_CARD_NATIVE
        quality = _BATCH_CARD_QUALITY
    if identifier in _STATUS_FAMILY_IDS:
        native = _STATUS_FAMILY_NATIVE
        root = _STATUS_FAMILY_ROOT
        # Bash retains its existing scoped first-combat measurement. The other five
        # recipes are executable/search-supported but gain no quality claim merely
        # from deterministic behavior fixtures.
        quality = _CARD_QUALITY if identifier == "bash" else None
    if identifier in _DRAW_FAMILY_IDS:
        native = _DRAW_FAMILY_NATIVE
        root = _DRAW_FAMILY_ROOT
        quality = None
    if identifier in _EXHAUST_FAMILY_IDS:
        native = _EXHAUST_FAMILY_NATIVE
        root = _EXHAUST_FAMILY_ROOT
        quality = None
    if identifier in _GENERATED_CARD_FAMILY_IDS:
        native = _GENERATED_CARD_FAMILY_NATIVE
        root = _GENERATED_CARD_FAMILY_ROOT
        quality = None
    if identifier in _STATUS_INSERTION_FAMILY_IDS:
        native = _STATUS_INSERTION_FAMILY_NATIVE
        root = _STATUS_INSERTION_FAMILY_ROOT
        quality = None
    if identifier in _EXHAUST_ONLY_IDS:
        native = _EXHAUST_ONLY_NATIVE
        root = _EXHAUST_ONLY_ROOT
        quality = None
    if identifier in _DRAW_TRIGGER_IDS:
        native = _DRAW_TRIGGER_NATIVE
        root = _claim("native_fixture", "draw-trigger-card-recipes-v1",
                      "resolved roots with public persistent trigger amounts and relative order")
        quality = None
    if identifier in _EXHAUST_TRIGGER_IDS:
        native = _EXHAUST_TRIGGER_NATIVE
        root = _claim("native_fixture", "exhaust-trigger-card-recipes-v1",
                      "resolved roots with public exhaust trigger amounts and relative order")
        quality = None
    if identifier in _SELECTED_TOPDECK_FAMILY_IDS:
        native = _SELECTED_TOPDECK_FAMILY_NATIVE
        root = _SELECTED_TOPDECK_FAMILY_ROOT
        quality = None
    if identifier in _CONDITIONAL_DRAW_ENERGY_FAMILY_IDS:
        native = _CONDITIONAL_DRAW_ENERGY_FAMILY_NATIVE
        root = _CONDITIONAL_DRAW_ENERGY_FAMILY_ROOT
        quality = None
    if identifier in _ENERGY_SELF_HP_FAMILY_IDS:
        native = _ENERGY_SELF_HP_FAMILY_NATIVE
        root = _ENERGY_SELF_HP_FAMILY_ROOT
        quality = None
    if identifier in _STRENGTH_FAMILY_IDS:
        native = _STRENGTH_FAMILY_NATIVE
        root = _STRENGTH_FAMILY_ROOT
        quality = None
    if identifier in _CROSS_COLOR_IDS:
        native = _CROSS_COLOR_NATIVE
        root = _CROSS_COLOR_ROOT
        quality = None
    if identifier in {"twin_strike", "pummel", "heavy_blade"}:
        native = _claim(
            "native_fixture", "attack-sequence-card-recipes-v1",
            "six variants; per-hit damage, Strength scaling, block, Weak/Vulnerable, "
            "exhaust, lethal interruption and selected on-hit interactions",
        )
        root = _claim(
            "native_fixture", "attack-sequence-card-recipes-v1",
            "fixed Cultist roots before/after play; hidden-order invariance and Frozen Eye; "
            "not natural trajectories or all-interaction verification",
        )
        quality = None
    if identifier in {"sword_boomerang", "whirlwind"}:
        native = _claim(
            "native_fixture", "random-x-attack-recipes-v1",
            "four variants; random live-target reselection, per-hit versus matrix timing, "
            "X payment/free play/Chemical X, modifiers, lifecycle and lethal interruption",
        )
        root = _claim(
            "native_fixture", "random-x-attack-recipes-v1",
            "Small Slimes fixed roots; 256 paired worlds ignore source hidden RNG/pile order; "
            "Frozen Eye and post-play sampling; not run-level calibration or quality evidence",
        )
        quality = None
    if identifier in {"bandage_up", "bite", "reaper", "feed", "hand_of_greed"}:
        native = _claim(
            "native_fixture", "healing-fatal-card-recipes-v1",
            "ten variants; heal caps/block/overkill/Bloom/Flower, minion/blocked fatal "
            "rewards/Ectoplasm, terminal heals and patched Greed attack triggers",
        )
        root = _claim(
            "native_fixture", "healing-fatal-card-recipes-v1",
            "Small Slimes pre/post-play roots, hidden-order invariance and Frozen Eye; "
            "not all revival/relic interactions or rollout quality",
        )
        quality = None

    if identifier in {"barricade", "entrench", "metallicize", "flame_barrier"}:
        native = _claim(
            "native_fixture", "persistent-defense-card-recipes-v1",
            "eight variants; retention/doubling/999 cap, non-card block modifiers, "
            "end-turn timing, blocked/multi-hit retaliation/expiry and selected combinations",
        )
        root = _claim(
            "native_fixture", "persistent-defense-card-recipes-v1",
            "Cultist pre/post-play sampling, paired unknown order and Frozen Eye; "
            "not all relic/power ordering or rollout quality",
        )
        quality = None

    claims = {
        EvidenceLayer.STATIC_METADATA: _CARD_STATIC,
        EvidenceLayer.NATIVE_BEHAVIOR: native,
        EvidenceLayer.ROOT_ACCEPTANCE: root,
    }
    if _recipe_derives(recipe, EvidenceLayer.PUBLIC_HISTORY):
        claims[EvidenceLayer.PUBLIC_HISTORY] = _CARD_HISTORY
    if _recipe_derives(recipe, EvidenceLayer.RULE_POLICY):
        claims[EvidenceLayer.RULE_POLICY] = _CARD_RULE
    if _recipe_derives(recipe, EvidenceLayer.RESAMPLING):
        claims[EvidenceLayer.RESAMPLING] = _CARD_RESAMPLING
    if quality is not None:
        claims[EvidenceLayer.ROLLOUT_QUALITY] = quality
    return MechanismRecord(
        "card",
        recipe.identifier,
        recipe.mechanisms,
        claims,
        recipe.signature,
        recipe.composition,
    )


_ENCOUNTER_NATIVE = _claim(
    "source_review", "combat-public-resample-v11-approx", "declared encounter recipe only",
)
_ENCOUNTER_STATIC = _claim(
    "source_generated", "rule-catalog-v1", "encounter composition definition",
)
_ENCOUNTER_RESAMPLING = _claim(
    "native_fixture", "public-combat-resampling-v1", "public-root invariance and branching",
)
_ENCOUNTER_ROOT = _claim(
    "measured", "combat-survival-benchmark-v1", "accepted fixed development root",
)
_ENCOUNTER_QUALITY = _claim(
    "scoped_measurement", "combat-survival-benchmark-v1", "paired first-combat fixture",
)
_SMALL_SLIMES_NATIVE = _claim(
    "native_fixture",
    "small-slimes-encounter-v1",
    "both legal A0 compositions and all eight base-ascension move effects",
)
_SMALL_SLIMES_RESAMPLING = _claim(
    "native_fixture",
    "small-slimes-encounter-v1",
    "both public A0 compositions are invariant; malformed composition/history rejects",
)
_SMALL_SLIMES_ROOT = _claim(
    "native_fixture",
    "small-slimes-encounter-v1",
    "two executable A0 PLAYER_NORMAL roots, one per legal composition",
)
_ACT1_ELITES_NATIVE = _claim(
    "native_fixture",
    "act1-elite-encounters-v1",
    "A0 Nob, Lagavulin and Sentries move effects and public state transitions",
)
_ACT1_ELITES_RESAMPLING = _claim(
    "native_fixture",
    "act1-elite-encounters-v1",
    "A0 elite public-root invariance plus malformed recipe/history refusal",
)
_ACT1_ELITES_ROOT = _claim(
    "native_fixture",
    "act1-elite-encounters-v1",
    "one executable A0 PLAYER_NORMAL root for each Act 1 elite encounter",
)


def _recipe_encounter(recipe: EncounterRecipe) -> MechanismRecord:
    native = _ENCOUNTER_NATIVE
    resampling = _ENCOUNTER_RESAMPLING
    root = _ENCOUNTER_ROOT
    quality: EvidenceClaim | None = _ENCOUNTER_QUALITY
    if recipe.identifier in {
        "gremlin_leader",
        "automaton",
        "collector",
        "champ",
        "three_darklings",
        "orb_walker",
        "three_shapes",
        "spire_growth",
        "transient",
        "four_shapes",
        "maw",
        "sphere_and_two_shapes",
        "jaw_worm_horde",
        "writhing_mass",
        "giant_head",
        "nemesis",
        "reptomancer",
        "awakened_one",
        "time_eater",
        "donu_and_deca",
        "shield_and_spear",
        "the_heart",
        "colosseum_event_slavers",
        "colosseum_event_nobs",
        "masked_bandits_event",
        "mysterious_sphere_event",
    }:
        native = _claim("native_fixture", "remaining-encounters-history-v1",
                        "A0 bounded lifecycle, summon/reuse, revival, phases and public counters; not exhaustive parity")
        resampling = _claim("native_fixture", "remaining-encounters-history-v1",
                           "same-public hidden-cache/RNG twins, independent Darkling damage prior and child history")
        root = _claim("native_fixture", "remaining-encounters-history-v1",
                      "26 encounter identities with full combat-entry histories; no policy quality claim")
        quality = None
    elif recipe.identifier in {"slavers", "book_of_stabbing"}:
        native = _claim("native_fixture", "act2-slavers-book-history-v1",
                        "A0 multi-hit growth, HP-loss versus unconditional wounds, debuffs and survivor masks")
        resampling = _claim("native_fixture", "act2-slavers-book-history-v1",
                           "observed hit-count history, single-stab hidden-counter/RNG twins and child histories")
        root = _claim("native_fixture", "act2-slavers-book-history-v1",
                      "multi-turn and all trio survivor masks; contradictory counts refuse; no quality claim")
        quality = None
    elif recipe.identifier in {"centurion_and_healer", "snecko", "two_thieves"}:
        native = _claim("native_fixture", "act2-ordinary-history-v1",
                        "A0 ally-state/targeting, Confusion draw costs and independent thief lifecycles")
        resampling = _claim("native_fixture", "act2-ordinary-history-v1",
                           "observed history, capped ordered theft, preserved visible costs and hidden-state twins")
        root = _claim("native_fixture", "act2-ordinary-history-v1",
                      "multi-turn/survivor/escape roots; ambiguous first-Mug death refuses; no quality claim")
        quality = None
    elif recipe.identifier in {"chosen", "three_byrds", "chosen_and_byrds", "cultist_and_chosen"}:
        native = _claim("native_fixture", "act2-chosen-byrd-history-v1",
                        "A0 Flight/grounding/recovery and Hex play-trigger lifecycle; four compositions")
        resampling = _claim("native_fixture", "act2-chosen-byrd-history-v1",
                           "observed intent histories, autoplay interruption and hidden-cache/RNG twins")
        root = _claim("native_fixture", "act2-chosen-byrd-history-v1",
                      "multi-turn, mixed-survivor and Hex insertion roots; no quality claim")
        quality = None
    elif recipe.identifier in {"spheric_guardian", "shell_parasite", "snake_plant",
                             "sentry_and_sphere", "shelled_parasite_and_fungi", "three_cultist"}:
        native = _claim("native_fixture", "act2-reactive-history-v1",
                        "A0 cycle/armor/stun/Malleable/healing and six fixed compositions")
        resampling = _claim("native_fixture", "act2-reactive-history-v1",
                           "public history, transient stun recovery and autoplay interruption twins")
        root = _claim("native_fixture", "act2-reactive-history-v1",
                      "multi-turn, reactive power and surviving-group roots; no quality claim")
        quality = None
    elif recipe.identifier in {"blue_slaver", "red_slaver", "looter", "gremlin_gang", "exordium_thugs"}:
        native = _claim("native_fixture", "act1-humanoid-history-v1",
                        "A0 humanoid effects and373 legal compositions; separate Slaver rule regression")
        resampling = _claim("native_fixture", "act1-humanoid-history-v1",
                           "observed limited-use/charge/theft; future ally selection; ambiguity refusal")
        root = _claim("native_fixture", "act1-humanoid-history-v1",
                      "all declared compositions and multi-turn/death/escape roots")
        quality = None
    elif recipe.identifier == "small_slimes":
        native = _SMALL_SLIMES_NATIVE
        resampling = _SMALL_SLIMES_RESAMPLING
        root = _SMALL_SLIMES_ROOT
        quality = None
    elif recipe.identifier in {"gremlin_nob", "lagavulin", "3_sentries"}:
        native = _ACT1_ELITES_NATIVE
        resampling = _ACT1_ELITES_RESAMPLING
        root = _ACT1_ELITES_ROOT
        quality = None
    elif recipe.identifier in {"the_guardian", "hexaghost", "slime_boss"}:
        native = _claim("native_fixture", "act1-boss-public-history-v1",
                        "A0 boss cycles, phase interrupts and both slime split orders")
        resampling = _claim("native_fixture", "act1-boss-public-history-v1",
                           "observed-history reconstruction; hidden-cache/RNG independence")
        root = _claim("native_fixture", "act1-boss-public-history-v1",
                      "opening and multi-turn/multi-phase resolved player roots")
        quality = None
    elif recipe.identifier in {"large_slime", "lots_of_slimes", "3_louse", "2_louse",
                               "2_fungi_beasts", "exordium_wildlife", "lagavulin_event",
                               "mushrooms_event"}:
        native = _claim("native_fixture", "act1-composition-history-v1",
                        "A0 composition lifecycles; Louse damage memory/prior at A0-20")
        resampling = _claim("native_fixture", "act1-composition-history-v1",
                           "observed history and independent unrevealed damage; hidden-cache invariance")
        root = _claim("native_fixture", "act1-composition-history-v1",
                      "all declared A0 compositions, splits/deaths and multi-turn roots")
        quality = None
    claims = {
        EvidenceLayer.STATIC_METADATA: _ENCOUNTER_STATIC,
        EvidenceLayer.NATIVE_BEHAVIOR: native,
        EvidenceLayer.RESAMPLING: resampling,
        EvidenceLayer.ROOT_ACCEPTANCE: root,
    }
    if quality is not None:
        claims[EvidenceLayer.ROLLOUT_QUALITY] = quality
    return MechanismRecord(
        "encounter",
        recipe.identifier,
        recipe.mechanisms(DEFAULT_ENCOUNTER_RECIPES.monsters),
        claims,
        recipe.signature(DEFAULT_ENCOUNTER_RECIPES.monsters),
        recipe.composition.value,
    )


DEFAULT_MECHANISM_EVIDENCE = MechanismEvidenceRegistry((
    *(_recipe_card(recipe) for recipe in DEFAULT_CARD_RECIPES.recipes()),
    *(
        _recipe_encounter(recipe)
        for recipe in DEFAULT_ENCOUNTER_RECIPES.encounters.values()
    ),
))
