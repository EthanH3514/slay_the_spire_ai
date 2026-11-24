from __future__ import annotations

import re


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_POSSESSIVE = re.compile(r"['’]s\b", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_ALIASES: dict[str, dict[str, str]] = {
    "power": {
        "intangible_player": "intangible",
        "no_block_power": "no_block",
    },
    "card": {
        "ghostly": "apparition",
        "master_reality_power": "master_reality",
        "j_a_x": "jax",
        "underhanded_strike": "sneaky_strike",
        "strike_r": "strike_red",
        "defend_r": "defend_red",
        "strike_g": "strike_green",
        "defend_g": "defend_green",
        "strike_b": "strike_blue",
        "defend_b": "defend_blue",
        "strike_p": "strike_purple",
        "defend_p": "defend_purple",
    },
    "potion": {
        "potion_slot": "empty_potion_slot",
        "empty_potion_id": "empty_potion_slot",
    },
    "monster": {
        "two_louse": "2_louse",
        "three_louse": "3_louse",
        "two_fungi_beasts": "2_fungi_beasts",
        "three_sentries": "3_sentries",
        "fuzzy_louse_normal": "red_louse",
        "fuzzy_louse_defensive": "green_louse",
    },
    "relic": {
        "boot": "the_boot",
        "paper_frog": "paper_phrog",
        "toxic_egg_2": "toxic_egg",
        "frozen_egg_2": "frozen_egg",
        "molten_egg_2": "molten_egg",
        "winged_greaves": "wing_boots",
        "sling": "sling_of_courage",
        "neows_blessing": "neows_lament",
    },
    "screen": {
        "combat_reward": "rewards",
        "grid": "card_select",
    },
}


def canonical_identifier(value: object, namespace: str = "generic") -> str | None:
    """Return a stable cross-backend identifier without using localized names."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _CAMEL_BOUNDARY.sub("_", text).lower()
    text = _POSSESSIVE.sub("s", text)
    token = _NON_ALNUM.sub("_", text).strip("_")
    return _ALIASES.get(namespace, {}).get(token, token)
