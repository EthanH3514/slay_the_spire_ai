from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations, product
from types import MappingProxyType
from typing import Iterable, Mapping

from .identifiers import canonical_identifier


ENCOUNTER_RECIPE_VERSION = "combat-encounter-recipes-v9"


class MonsterStatePrimitive(str, Enum):
    PUBLIC_CURRENT_INTENT = "public_current_intent"
    PUBLIC_INTENT_HISTORY = "public_intent_history"
    RANDOM_FUTURE_INTENT = "random_future_intent"
    FIXED_INTENT_CYCLE = "fixed_intent_cycle"
    HIDDEN_DAMAGE_REVEALED_BY_ATTACK = "hidden_damage_revealed_by_attack"
    REVEALED_DAMAGE_MEMORY = "revealed_damage_memory"
    UNREVEALED_DAMAGE_PRIOR = "unrevealed_damage_prior"
    DEATH_TRIGGER_DEBUFF = "death_trigger_debuff"
    REACTIVE_PUBLIC_POWER = "reactive_public_power"
    PUBLIC_PHASE = "public_phase"
    STATUS_CARD_INSERTION = "status_card_insertion"
    PLAYER_DEBUFFS = "player_debuffs"
    PUBLIC_ARTIFACT = "public_artifact"
    OBSERVED_PHASE_COUNTER = "observed_phase_counter"
    SPAWN_IDENTITY = "spawn_identity"
    LIMITED_USE_INTENT = "limited_use_intent"
    OBSERVED_GOLD_THEFT = "observed_gold_theft"
    ENEMY_ESCAPE = "enemy_escape"
    RANDOM_ALLY_TARGET = "random_ally_target"
    PUBLIC_ALLY_PRESENCE = "public_ally_presence"
    BLOCK_RETENTION = "block_retention"
    DAMAGE_BASED_HEALING = "damage_based_healing"
    STUN_INTERRUPT = "stun_interrupt"
    DAMAGE_REDUCTION = "damage_reduction"
    NON_ATTACK_PLAY_TRIGGER = "non_attack_play_trigger"
    PUBLIC_ALLY_HEALTH = "public_ally_health"
    DRAW_COST_RANDOMIZATION = "draw_cost_randomization"
    REVIVAL = "revival"
    PUBLIC_CARD_STASIS = "public_card_stasis"
    PUBLIC_FACING = "public_facing"
    FORCED_TURN_END = "forced_turn_end"


class EncounterCompositionPrimitive(str, Enum):
    SINGLE_ENEMY = "single_enemy"
    FIXED_MULTI_ENEMY = "fixed_multi_enemy"
    PUBLIC_COMPOSITION_VARIANT = "public_composition_variant"
    PUBLIC_SPLIT_SLOTS = "public_split_slots"
    PUBLIC_SUMMON_SLOTS = "public_summon_slots"


@dataclass(frozen=True, slots=True)
class MonsterBehaviorRecipe:
    identifier: str
    move_ids: tuple[str, ...]
    mechanisms: tuple[MonsterStatePrimitive, ...]
    history_constraint: str = "known_moves"

    def __post_init__(self) -> None:
        canonical = canonical_identifier(self.identifier, "monster")
        if canonical != self.identifier:
            raise ValueError("monster recipe identifiers must be canonical")
        if not self.move_ids or len(set(self.move_ids)) != len(self.move_ids):
            raise ValueError("monster recipes require unique move IDs")
        if not self.mechanisms or len(set(self.mechanisms)) != len(self.mechanisms):
            raise ValueError("monster recipes require unique mechanisms")
        if self.history_constraint not in {
            "known_moves", "alternating", "fixed_without_history_shift",
        }:
            raise ValueError("unsupported monster history constraint")

    @property
    def signature(self) -> str:
        moves = ",".join(self.move_ids)
        return f"{self.identifier}:{self.history_constraint}:{moves}"


@dataclass(frozen=True, slots=True)
class EncounterRecipe:
    identifier: str
    compositions: tuple[tuple[str, ...], ...]
    composition: EncounterCompositionPrimitive
    minimum_ascension: int
    maximum_ascension: int
    requires_enemy_history: bool = False

    def __post_init__(self) -> None:
        canonical = canonical_identifier(self.identifier, "monster")
        if canonical != self.identifier:
            raise ValueError("encounter recipe identifiers must be canonical")
        if not self.compositions or len(set(self.compositions)) != len(self.compositions):
            raise ValueError("encounter recipes require unique compositions")
        if any(not row or len(row) > 5 for row in self.compositions):
            raise ValueError("encounter compositions require one to five slots")
        if any("invalid" in row for row in self.compositions) and self.composition not in {
            EncounterCompositionPrimitive.PUBLIC_SPLIT_SLOTS,
            EncounterCompositionPrimitive.PUBLIC_SUMMON_SLOTS,
        }:
            raise ValueError("empty slots require a public split or summon recipe")
        if not 0 <= self.minimum_ascension <= self.maximum_ascension <= 20:
            raise ValueError("invalid encounter ascension range")
        single_composition = self.composition in {
            EncounterCompositionPrimitive.SINGLE_ENEMY,
            EncounterCompositionPrimitive.FIXED_MULTI_ENEMY,
        }
        if (single_composition and len(self.compositions) != 1) or (
            not single_composition and len(self.compositions) < 2
        ):
            raise ValueError("composition primitive does not match alternatives")
        if self.composition is EncounterCompositionPrimitive.SINGLE_ENEMY and any(
            len(row) != 1 for row in self.compositions
        ):
            raise ValueError("single-enemy recipes require one monster")
        if self.composition is EncounterCompositionPrimitive.FIXED_MULTI_ENEMY and any(
            len(row) < 2 for row in self.compositions
        ):
            raise ValueError("fixed multi-enemy recipes require multiple monsters")

    @property
    def monster_ids(self) -> frozenset[str]:
        return frozenset(monster for row in self.compositions for monster in row
                         if monster != "invalid")

    def mechanisms(
        self, monsters: Mapping[str, MonsterBehaviorRecipe],
    ) -> tuple[str, ...]:
        ordered = [self.composition.value]
        for monster_id in sorted(self.monster_ids):
            recipe = monsters[monster_id]
            ordered.extend(mechanism.value for mechanism in recipe.mechanisms)
        return tuple(dict.fromkeys(ordered))

    def signature(
        self, monsters: Mapping[str, MonsterBehaviorRecipe],
    ) -> tuple[str, ...]:
        compositions = tuple(
            "composition:" + "+".join(row) for row in self.compositions
        )
        monster_rows = tuple(
            "monster:" + monsters[identifier].signature
            for identifier in sorted(self.monster_ids)
        )
        return (
            f"ascension:{self.minimum_ascension}-{self.maximum_ascension}",
            *(("public_enemy_history:from_combat_entry",) if self.requires_enemy_history else ()),
            *compositions,
            *monster_rows,
        )


class EncounterRecipeCatalog:
    version = ENCOUNTER_RECIPE_VERSION

    def __init__(
        self,
        monsters: Iterable[MonsterBehaviorRecipe],
        encounters: Iterable[EncounterRecipe],
    ) -> None:
        monster_rows = tuple(monsters)
        encounter_rows = tuple(encounters)
        monster_index = {recipe.identifier: recipe for recipe in monster_rows}
        encounter_index = {recipe.identifier: recipe for recipe in encounter_rows}
        if len(monster_index) != len(monster_rows):
            raise ValueError("duplicate monster recipe")
        if len(encounter_index) != len(encounter_rows):
            raise ValueError("duplicate encounter recipe")
        missing = {
            monster
            for recipe in encounter_index.values()
            for monster in recipe.monster_ids
            if monster not in monster_index
        }
        if missing:
            raise ValueError(f"encounter recipes reference missing monsters: {missing}")
        self.monsters = MappingProxyType(monster_index)
        self.encounters = MappingProxyType(encounter_index)

    def encounter(self, value: object) -> EncounterRecipe | None:
        identifier = canonical_identifier(value, "monster")
        return self.encounters.get(identifier or "<missing>")

    def identifiers(self) -> frozenset[str]:
        return frozenset(self.encounters)


def _monster(
    identifier: str,
    moves: tuple[str, ...],
    *mechanisms: MonsterStatePrimitive,
    history: str = "known_moves",
) -> MonsterBehaviorRecipe:
    return MonsterBehaviorRecipe(identifier, moves, mechanisms, history)


M = MonsterStatePrimitive
_MONSTERS = (
    _monster(
        "awakened_one", ("AWAKENED_ONE_SLASH", "AWAKENED_ONE_SOUL_STRIKE", "AWAKENED_ONE_REBIRTH", "AWAKENED_ONE_DARK_ECHO", "AWAKENED_ONE_SLUDGE", "AWAKENED_ONE_TACKLE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.REVIVAL, M.REACTIVE_PUBLIC_POWER, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "bear", ("BEAR_BEAR_HUG", "BEAR_LUNGE", "BEAR_MAUL"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PLAYER_DEBUFFS, M.FIXED_INTENT_CYCLE,
    ),
    _monster(
        "bronze_automaton", ("BRONZE_AUTOMATON_BOOST", "BRONZE_AUTOMATON_FLAIL", "BRONZE_AUTOMATON_HYPER_BEAM", "BRONZE_AUTOMATON_SPAWN_ORBS", "BRONZE_AUTOMATON_STUNNED"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.OBSERVED_PHASE_COUNTER, M.SPAWN_IDENTITY, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "bronze_orb", ("BRONZE_ORB_BEAM", "BRONZE_ORB_STASIS", "BRONZE_ORB_SUPPORT_BEAM"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.LIMITED_USE_INTENT, M.PUBLIC_CARD_STASIS, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "corrupt_heart", ("CORRUPT_HEART_DEBILITATE", "CORRUPT_HEART_BLOOD_SHOTS", "CORRUPT_HEART_ECHO", "CORRUPT_HEART_BUFF"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.REACTIVE_PUBLIC_POWER, M.DAMAGE_REDUCTION, M.STATUS_CARD_INSERTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "dagger", ("DAGGER_STAB", "DAGGER_EXPLODE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.SPAWN_IDENTITY, M.FIXED_INTENT_CYCLE, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "darkling", ("DARKLING_NIP", "DARKLING_CHOMP", "DARKLING_HARDEN", "DARKLING_REINCARNATE", "DARKLING_REGROW"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REVEALED_DAMAGE_MEMORY, M.UNREVEALED_DAMAGE_PRIOR, M.REVIVAL, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "deca", ("DECA_SQUARE_OF_PROTECTION", "DECA_BEAM"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.FIXED_INTENT_CYCLE, M.PUBLIC_ARTIFACT, M.STATUS_CARD_INSERTION, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "donu", ("DONU_CIRCLE_OF_POWER", "DONU_BEAM"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.FIXED_INTENT_CYCLE, M.PUBLIC_ARTIFACT, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "exploder", ("EXPLODER_SLAM", "EXPLODER_EXPLODE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.FIXED_INTENT_CYCLE,
    ),
    _monster(
        "giant_head", ("GIANT_HEAD_COUNT", "GIANT_HEAD_GLARE", "GIANT_HEAD_IT_IS_TIME"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.REACTIVE_PUBLIC_POWER, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "gremlin_leader", ("GREMLIN_LEADER_ENCOURAGE", "GREMLIN_LEADER_RALLY", "GREMLIN_LEADER_STAB"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.SPAWN_IDENTITY, M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "nemesis", ("NEMESIS_DEBUFF", "NEMESIS_ATTACK", "NEMESIS_SCYTHE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.DAMAGE_REDUCTION, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "orb_walker", ("ORB_WALKER_LASER", "ORB_WALKER_CLAW"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.STATUS_CARD_INSERTION, M.REACTIVE_PUBLIC_POWER,
    ),
    _monster(
        "pointy", ("POINTY_ATTACK",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE,

        history="fixed_without_history_shift",
    ),
    _monster(
        "reptomancer", ("REPTOMANCER_SUMMON", "REPTOMANCER_SNAKE_STRIKE", "REPTOMANCER_BIG_BITE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.SPAWN_IDENTITY, M.PUBLIC_ALLY_PRESENCE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "repulsor", ("REPULSOR_BASH", "REPULSOR_REPULSE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "romeo", ("ROMEO_MOCK", "ROMEO_AGONIZING_SLASH", "ROMEO_CROSS_SLASH"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.FIXED_INTENT_CYCLE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "spiker", ("SPIKER_CUT", "SPIKER_SPIKE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.OBSERVED_PHASE_COUNTER, M.REACTIVE_PUBLIC_POWER, M.LIMITED_USE_INTENT,
    ),
    _monster(
        "spire_growth", ("SPIRE_GROWTH_QUICK_TACKLE", "SPIRE_GROWTH_SMASH", "SPIRE_GROWTH_CONSTRICT"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PLAYER_DEBUFFS,
    ),
    _monster(
        "spire_shield", ("SPIRE_SHIELD_BASH", "SPIRE_SHIELD_FORTIFY", "SPIRE_SHIELD_SMASH"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_FACING, M.PUBLIC_ALLY_PRESENCE, M.PLAYER_DEBUFFS, M.PUBLIC_ARTIFACT,
    ),
    _monster(
        "spire_spear", ("SPIRE_SPEAR_BURN_STRIKE", "SPIRE_SPEAR_PIERCER", "SPIRE_SPEAR_SKEWER"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_FACING, M.PUBLIC_ALLY_PRESENCE, M.STATUS_CARD_INSERTION, M.PUBLIC_ARTIFACT,
    ),
    _monster(
        "the_champ", ("THE_CHAMP_DEFENSIVE_STANCE", "THE_CHAMP_FACE_SLAP", "THE_CHAMP_TAUNT", "THE_CHAMP_HEAVY_SLASH", "THE_CHAMP_GLOAT", "THE_CHAMP_EXECUTE", "THE_CHAMP_ANGER"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.OBSERVED_PHASE_COUNTER, M.LIMITED_USE_INTENT, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "the_collector", ("THE_COLLECTOR_BUFF", "THE_COLLECTOR_FIREBALL", "THE_COLLECTOR_MEGA_DEBUFF", "THE_COLLECTOR_SPAWN"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.SPAWN_IDENTITY, M.PUBLIC_PHASE, M.PUBLIC_ALLY_PRESENCE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "the_maw", ("THE_MAW_ROAR", "THE_MAW_DROOL", "THE_MAW_SLAM", "THE_MAW_NOM"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "time_eater", ("TIME_EATER_REVERBERATE", "TIME_EATER_HEAD_SLAM", "TIME_EATER_RIPPLE", "TIME_EATER_HASTE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.LIMITED_USE_INTENT, M.FORCED_TURN_END, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "torch_head", ("TORCH_HEAD_TACKLE",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE,
        M.SPAWN_IDENTITY,
        history="fixed_without_history_shift",
    ),
    _monster(
        "transient", ("TRANSIENT_ATTACK",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE,
        M.PUBLIC_PHASE, M.REACTIVE_PUBLIC_POWER,
        history="fixed_without_history_shift",
    ),
    _monster(
        "writhing_mass", ("WRITHING_MASS_IMPLANT", "WRITHING_MASS_FLAIL", "WRITHING_MASS_WITHER", "WRITHING_MASS_MULTI_STRIKE", "WRITHING_MASS_STRONG_STRIKE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.LIMITED_USE_INTENT, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "book_of_stabbing", ("BOOK_OF_STABBING_MULTI_STAB", "BOOK_OF_STABBING_SINGLE_STAB"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.OBSERVED_PHASE_COUNTER, M.REACTIVE_PUBLIC_POWER, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "taskmaster", ("TASKMASTER_SCOURING_WHIP",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, M.STATUS_CARD_INSERTION,
        history="fixed_without_history_shift",
    ),
    _monster(
        "centurion", ("CENTURION_DEFEND", "CENTURION_FURY", "CENTURION_SLASH"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_ALLY_PRESENCE, M.RANDOM_ALLY_TARGET,
    ),
    _monster(
        "mystic", ("MYSTIC_ATTACK_DEBUFF", "MYSTIC_BUFF", "MYSTIC_HEAL"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_ALLY_HEALTH, M.PUBLIC_ALLY_PRESENCE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "snecko", ("SNECKO_PERPLEXING_GLARE", "SNECKO_BITE", "SNECKO_TAIL_WHIP"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.LIMITED_USE_INTENT, M.PLAYER_DEBUFFS, M.DRAW_COST_RANDOMIZATION,
    ),
    _monster(
        "mugger", ("MUGGER_MUG", "MUGGER_LUNGE", "MUGGER_SMOKE_BOMB", "MUGGER_ESCAPE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.OBSERVED_GOLD_THEFT, M.ENEMY_ESCAPE,
    ),
    _monster(
        "byrd", ("BYRD_CAW", "BYRD_FLY", "BYRD_HEADBUTT", "BYRD_PECK", "BYRD_STUNNED", "BYRD_SWOOP"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.DAMAGE_REDUCTION, M.STUN_INTERRUPT, M.PUBLIC_PHASE,
    ),
    _monster(
        "chosen", ("CHOSEN_DEBILITATE", "CHOSEN_DRAIN", "CHOSEN_HEX", "CHOSEN_POKE", "CHOSEN_ZAP"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.LIMITED_USE_INTENT, M.PLAYER_DEBUFFS, M.STATUS_CARD_INSERTION, M.NON_ATTACK_PLAY_TRIGGER,
    ),
    _monster(
        "spheric_guardian", ("SPHERIC_GUARDIAN_ACTIVATE", "SPHERIC_GUARDIAN_ATTACK_DEBUFF",
                             "SPHERIC_GUARDIAN_SLAM", "SPHERIC_GUARDIAN_HARDEN"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.FIXED_INTENT_CYCLE,
        M.PUBLIC_ARTIFACT, M.BLOCK_RETENTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "shelled_parasite", ("SHELLED_PARASITE_DOUBLE_STRIKE", "SHELLED_PARASITE_FELL",
                             "SHELLED_PARASITE_SUCK", "SHELLED_PARASITE_STUNNED"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.STUN_INTERRUPT, M.DAMAGE_BASED_HEALING, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "snake_plant", ("SNAKE_PLANT_CHOMP", "SNAKE_PLANT_ENFEEBLING_SPORES"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "blue_slaver", ("BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PLAYER_DEBUFFS,
    ),
    _monster(
        "red_slaver", ("RED_SLAVER_STAB", "RED_SLAVER_SCRAPE", "RED_SLAVER_ENTANGLE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PLAYER_DEBUFFS, M.LIMITED_USE_INTENT,
    ),
    _monster(
        "looter", ("LOOTER_MUG", "LOOTER_LUNGE", "LOOTER_SMOKE_BOMB", "LOOTER_ESCAPE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.OBSERVED_GOLD_THEFT, M.ENEMY_ESCAPE,
    ),
    _monster(
        "mad_gremlin", ("MAD_GREMLIN_SCRATCH",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, M.REACTIVE_PUBLIC_POWER,
        history="fixed_without_history_shift",
    ),
    _monster(
        "sneaky_gremlin", ("SNEAKY_GREMLIN_PUNCTURE",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, history="fixed_without_history_shift",
    ),
    _monster(
        "fat_gremlin", ("FAT_GREMLIN_SMASH",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, M.PLAYER_DEBUFFS,
        history="fixed_without_history_shift",
    ),
    _monster(
        "shield_gremlin", ("SHIELD_GREMLIN_PROTECT", "SHIELD_GREMLIN_SHIELD_BASH"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_ALLY_TARGET,
        M.PUBLIC_ALLY_PRESENCE,
    ),
    _monster(
        "gremlin_wizard", ("GREMLIN_WIZARD_CHARGING", "GREMLIN_WIZARD_ULTIMATE_BLAST"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.FIXED_INTENT_CYCLE,
        M.OBSERVED_PHASE_COUNTER,
    ),
    _monster(
        "fungi_beast", ("FUNGI_BEAST_BITE", "FUNGI_BEAST_GROW"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.DEATH_TRIGGER_DEBUFF, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "the_guardian",
        ("THE_GUARDIAN_CHARGING_UP", "THE_GUARDIAN_FIERCE_BASH",
         "THE_GUARDIAN_VENT_STEAM", "THE_GUARDIAN_WHIRLWIND",
         "THE_GUARDIAN_DEFENSIVE_MODE", "THE_GUARDIAN_ROLL_ATTACK",
         "THE_GUARDIAN_TWIN_SLAM"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.FIXED_INTENT_CYCLE,
        M.REACTIVE_PUBLIC_POWER, M.PUBLIC_PHASE, M.OBSERVED_PHASE_COUNTER,
        M.PLAYER_DEBUFFS,
    ),
    _monster(
        "hexaghost",
        ("HEXAGHOST_ACTIVATE", "HEXAGHOST_DIVIDER", "HEXAGHOST_SEAR",
         "HEXAGHOST_TACKLE", "HEXAGHOST_INFLAME", "HEXAGHOST_INFERNO"),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, M.OBSERVED_PHASE_COUNTER,
        M.HIDDEN_DAMAGE_REVEALED_BY_ATTACK, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "slime_boss",
        ("SLIME_BOSS_GOOP_SPRAY", "SLIME_BOSS_PREPARING", "SLIME_BOSS_SLAM",
         "SLIME_BOSS_SPLIT"),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE, M.PUBLIC_PHASE,
        M.SPAWN_IDENTITY, M.STATUS_CARD_INSERTION,
    ),
    _monster(
        "acid_slime_l",
        ("ACID_SLIME_L_CORROSIVE_SPIT", "ACID_SLIME_L_TACKLE",
         "ACID_SLIME_L_LICK", "ACID_SLIME_L_SPLIT"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.SPAWN_IDENTITY, M.STATUS_CARD_INSERTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "spike_slime_l",
        ("SPIKE_SLIME_L_FLAME_TACKLE", "SPIKE_SLIME_L_LICK", "SPIKE_SLIME_L_SPLIT"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.PUBLIC_PHASE, M.SPAWN_IDENTITY, M.STATUS_CARD_INSERTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "cultist", ("CULTIST_INCANTATION", "CULTIST_DARK_STRIKE"),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE,
    ),
    _monster(
        "jaw_worm", ("JAW_WORM_CHOMP", "JAW_WORM_THRASH", "JAW_WORM_BELLOW"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
    ),
    _monster(
        "green_louse", ("GREEN_LOUSE_BITE", "GREEN_LOUSE_SPIT_WEB"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.HIDDEN_DAMAGE_REVEALED_BY_ATTACK, M.REVEALED_DAMAGE_MEMORY,
        M.UNREVEALED_DAMAGE_PRIOR,
    ),
    _monster(
        "red_louse", ("RED_LOUSE_BITE", "RED_LOUSE_GROW"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.HIDDEN_DAMAGE_REVEALED_BY_ATTACK, M.REVEALED_DAMAGE_MEMORY,
        M.UNREVEALED_DAMAGE_PRIOR,
    ),
    _monster(
        "acid_slime_s", ("ACID_SLIME_S_LICK", "ACID_SLIME_S_TACKLE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.FIXED_INTENT_CYCLE,
        M.PLAYER_DEBUFFS, history="alternating",
    ),
    _monster(
        "acid_slime_m",
        ("ACID_SLIME_M_CORROSIVE_SPIT", "ACID_SLIME_M_LICK", "ACID_SLIME_M_TACKLE"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.STATUS_CARD_INSERTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "spike_slime_s", ("SPIKE_SLIME_S_TACKLE",),
        M.PUBLIC_CURRENT_INTENT, M.FIXED_INTENT_CYCLE,
        history="fixed_without_history_shift",
    ),
    _monster(
        "spike_slime_m", ("SPIKE_SLIME_M_FLAME_TACKLE", "SPIKE_SLIME_M_LICK"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.STATUS_CARD_INSERTION, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "gremlin_nob",
        ("GREMLIN_NOB_BELLOW", "GREMLIN_NOB_RUSH", "GREMLIN_NOB_SKULL_BASH"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.RANDOM_FUTURE_INTENT,
        M.REACTIVE_PUBLIC_POWER, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "lagavulin", ("LAGAVULIN_SLEEP", "LAGAVULIN_ATTACK", "LAGAVULIN_SIPHON_SOUL"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.PUBLIC_PHASE,
        M.FIXED_INTENT_CYCLE, M.PLAYER_DEBUFFS,
    ),
    _monster(
        "sentry", ("SENTRY_BEAM", "SENTRY_BOLT"),
        M.PUBLIC_CURRENT_INTENT, M.PUBLIC_INTENT_HISTORY, M.FIXED_INTENT_CYCLE,
        M.STATUS_CARD_INSERTION, M.PUBLIC_ARTIFACT, history="alternating",
    ),
)

E = EncounterCompositionPrimitive
_ENCOUNTERS = (
    EncounterRecipe("champ", (("the_champ",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("three_darklings", (("darkling", "darkling", "darkling"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("orb_walker", (("orb_walker",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("spire_growth", (("spire_growth",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("transient", (("transient",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("maw", (("the_maw",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("jaw_worm_horde", (("jaw_worm", "jaw_worm", "jaw_worm"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("writhing_mass", (("writhing_mass",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("giant_head", (("giant_head",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("nemesis", (("nemesis",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("awakened_one", (("cultist", "cultist", "awakened_one"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("time_eater", (("time_eater",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("donu_and_deca", (("deca", "donu"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("shield_and_spear", (("spire_shield", "spire_spear"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("the_heart", (("corrupt_heart",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("colosseum_event_slavers", (("blue_slaver", "red_slaver"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("colosseum_event_nobs", (("taskmaster", "gremlin_nob"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("masked_bandits_event", (("pointy", "romeo", "bear"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("mysterious_sphere_event", (("orb_walker", "orb_walker"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("gremlin_leader",
                    tuple(row + ("gremlin_leader",) for row in product(
                        ("invalid", "mad_gremlin", "sneaky_gremlin", "fat_gremlin",
                         "shield_gremlin", "gremlin_wizard"), repeat=3)),
                    E.PUBLIC_SUMMON_SLOTS, 0, 0, requires_enemy_history=True),
    EncounterRecipe("automaton",
                    tuple((a, "bronze_automaton", b)
                          for a, b in product(("invalid", "bronze_orb"), repeat=2)),
                    E.PUBLIC_SUMMON_SLOTS, 0, 0, requires_enemy_history=True),
    EncounterRecipe("collector",
                    tuple((a, b, "the_collector")
                          for a, b in product(("invalid", "torch_head"), repeat=2)),
                    E.PUBLIC_SUMMON_SLOTS, 0, 0, requires_enemy_history=True),
    EncounterRecipe("reptomancer",
                    tuple((a, b, "reptomancer", c, d)
                          for a, b, c, d in product(("invalid", "dagger"), repeat=4)),
                    E.PUBLIC_SUMMON_SLOTS, 0, 0, requires_enemy_history=True),
    EncounterRecipe("sphere_and_two_shapes",
                    tuple((a, b, "spheric_guardian")
                          for a, b in product(("repulsor", "exploder", "spiker"), repeat=2)),
                    E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True),
    *(EncounterRecipe(name,
                      tuple(row for row in product(("repulsor", "exploder", "spiker"), repeat=size)
                            if max(row.count(x) for x in set(row)) <= 2),
                      E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True)
      for name, size in (("three_shapes", 3), ("four_shapes", 4))),
    EncounterRecipe("slavers", (("blue_slaver", "taskmaster", "red_slaver"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("book_of_stabbing", (("book_of_stabbing",),),
                    E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("centurion_and_healer", (("centurion", "mystic"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("snecko", (("snecko",),), E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("two_thieves", (("looter", "mugger"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("chosen", (("chosen",),), E.SINGLE_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("three_byrds", (("byrd", "byrd", "byrd"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("chosen_and_byrds", (("byrd", "chosen"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("cultist_and_chosen", (("cultist", "chosen"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("spheric_guardian", (("spheric_guardian",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("shell_parasite", (("shelled_parasite",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("snake_plant", (("snake_plant",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("sentry_and_sphere", (("sentry", "spheric_guardian"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("shelled_parasite_and_fungi", (("shelled_parasite", "fungi_beast"),),
                    E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True),
    EncounterRecipe("three_cultist", (("cultist", "cultist", "cultist"),), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("blue_slaver", (("blue_slaver",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("red_slaver", (("red_slaver",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("looter", (("looter",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe(
        "gremlin_gang",
        tuple(sorted(set(permutations(("mad_gremlin",) * 2 + ("sneaky_gremlin",) * 2 +
                                      ("fat_gremlin",) * 2 + ("shield_gremlin", "gremlin_wizard"), 4)))),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "exordium_thugs",
        tuple(product(("green_louse", "red_louse", "acid_slime_m", "spike_slime_m"),
                      ("cultist", "blue_slaver", "red_slaver", "looter"))),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "large_slime",
        (("acid_slime_l",), ("spike_slime_l",),
         ("acid_slime_m", "acid_slime_m"), ("spike_slime_m", "spike_slime_m")),
        E.PUBLIC_SPLIT_SLOTS, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "lots_of_slimes",
        tuple(sorted(set(permutations(("acid_slime_s",) * 2 + ("spike_slime_s",) * 3)))),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "3_louse", tuple(product(("green_louse", "red_louse"), repeat=3)),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "2_fungi_beasts", (("fungi_beast", "fungi_beast"),),
        E.FIXED_MULTI_ENEMY, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe(
        "exordium_wildlife",
        tuple(product(("fungi_beast", "jaw_worm"),
                      ("green_louse", "red_louse", "acid_slime_m", "spike_slime_m"))),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe("lagavulin_event", (("lagavulin",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("mushrooms_event", (("fungi_beast",) * 3,), E.FIXED_MULTI_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("the_guardian", (("the_guardian",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe("hexaghost", (("hexaghost",),), E.SINGLE_ENEMY, 0, 0,
                    requires_enemy_history=True),
    EncounterRecipe(
        "slime_boss",
        (("slime_boss",),
         ("spike_slime_l", "invalid", "acid_slime_l"),
         ("spike_slime_m", "spike_slime_m", "acid_slime_l", "invalid"),
         ("spike_slime_l", "invalid", "acid_slime_m", "acid_slime_m"),
         ("spike_slime_m", "spike_slime_m", "acid_slime_m", "acid_slime_m")),
        E.PUBLIC_SPLIT_SLOTS, 0, 0, requires_enemy_history=True,
    ),
    EncounterRecipe("cultist", (("cultist",),), E.SINGLE_ENEMY, 0, 20),
    EncounterRecipe("jaw_worm", (("jaw_worm",),), E.SINGLE_ENEMY, 0, 20),
    EncounterRecipe(
        "2_louse",
        (
            ("green_louse", "green_louse"),
            ("green_louse", "red_louse"),
            ("red_louse", "green_louse"),
            ("red_louse", "red_louse"),
        ),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 20,
    ),
    EncounterRecipe(
        "small_slimes",
        (
            ("acid_slime_s", "spike_slime_m"),
            ("spike_slime_s", "acid_slime_m"),
        ),
        E.PUBLIC_COMPOSITION_VARIANT, 0, 0,
    ),
    EncounterRecipe("gremlin_nob", (("gremlin_nob",),), E.SINGLE_ENEMY, 0, 0),
    EncounterRecipe("lagavulin", (("lagavulin",),), E.SINGLE_ENEMY, 0, 0),
    EncounterRecipe(
        "3_sentries", (("sentry", "sentry", "sentry"),),
        E.FIXED_MULTI_ENEMY, 0, 0,
    ),
)

DEFAULT_ENCOUNTER_RECIPES = EncounterRecipeCatalog(_MONSTERS, _ENCOUNTERS)
