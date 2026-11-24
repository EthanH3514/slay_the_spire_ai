#pragma once
#include <stdexcept>

// Policy choices only; historical game actions retain their original legality.
namespace spire_potion_replacement {
inline constexpr auto version = "immediate-potion-replacement-v1";
inline thread_local bool enabled = false;
inline thread_local unsigned long long menus = 0, excluded = 0;

inline bool setEnabled(bool value) {
    const bool old = enabled;
    if (value && !old) menus = excluded = 0;
    enabled = value;
    return old;
}

template<class Slots, class Potion, class CanUse>
int source(const Slots& slots, int capacity, int discard, Potion empty,
           Potion refill, bool blocked, CanUse canUse) {
    if (blocked || discard < 0 || discard >= capacity || slots[discard] == empty) return -1;
    for (int i = 0; i < capacity; ++i) if (slots[i] == empty) return -1;
    // A fixed public order resolves multiple currently usable refill bottles.
    // The discarded bottle can never serve as its own replacement source.
    for (int i = 0; i < capacity; ++i)
        if (i != discard && slots[i] == refill && canUse(i)) return i;
    return -1;
}
} // namespace spire_potion_replacement
