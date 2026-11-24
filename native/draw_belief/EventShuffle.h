#pragma once
#include "draw_belief/SimulatorTrace.h"
#include "draw_belief/Exact.h"
#include <map>
#include <functional>
#include <numeric>
#include <optional>

namespace spire_belief::trace {
// Explicit alternative probability model, NOT native-PRNG seed conditioning.
// Only a historical empty-deck shuffle on an owned replay can install this.
// The already observed prefix is conditioned; every eligible physical copy
// and every permutation of the remaining copies have their proper probability.
struct EventShuffle {
    static constexpr auto version = "public-conditional-shuffle-event-v1";
    const sts::CardManager* target;
    Inventory input, prefix;
    std::mt19937_64 rng;
    Budget& budget;
    bool used = false, applied = false;
    std::function<bool(sts::CardManager&,std::mt19937_64&,Budget&)> conditionalDraws;

    EventShuffle(const sts::CardManager& cards, Inventory from, Inventory seen,
                 std::uint64_t seed, Budget& work)
        : target(&cards), input(std::move(from)), prefix(std::move(seen)),
          rng(seed ^ 0x53485546464c4531ULL), budget(work) {
        if (input.size() > 4096 || prefix.empty() || prefix.size() > input.size())
            throw Failure("event_shuffle_inventory");
    }
    std::size_t pick(std::size_t count) {
        if (!count) throw Failure("event_shuffle_empty_choice");
        // Existing exact bounded categorical primitive charges all random words
        // and limits rejection attempts, including integer sampling overhead.
        return randomBelow(UInt(count), rng, budget).limb[0];
    }
    bool apply(sts::CardManager& cards) {
        if (&cards != target || used) return false;
        used = true; // Never carry a missed historical condition into the future.
        budget.tick(1 + cards.discardPile.size());
        Inventory actual;
        for (const auto& c : cards.discardPile) actual.push_back(project(c));
        if (actual != input) return false; // Full joint replay will reject it.
        if (conditionalDraws) return applied = conditionalDraws(cards,rng,budget);
        std::map<Card, std::vector<std::size_t>> copies;
        // Charge a conservative log-sized bound per ordered-map operation.
        std::size_t mapCost = 2;
        for (auto n = input.size(); n > 1; n >>= 1) ++mapCost;
        for (std::size_t i = 0; i < input.size(); ++i) {
            budget.tick(mapCost); copies[input[i]].push_back(i);
        }
        std::vector<bool> selected(input.size(), false);
        std::vector<std::size_t> top, order;
        for (const auto& label : prefix) {
            budget.tick(mapCost);
            auto found = copies.find(label);
            if (found == copies.end() || found->second.empty())
                throw Failure("event_shuffle_impossible_prefix");
            auto& group = found->second;
            auto k = pick(group.size()), index = group[k];
            selected[index] = true; top.push_back(index);
            group[k] = group.back(); group.pop_back();
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            budget.tick(); if (!selected[i]) order.push_back(i);
        }
        for (std::size_t n = order.size(); n > 1; --n) {
            budget.tick(); std::swap(order[n - 1], order[pick(n)]);
        }
        // Native draw pile is bottom-to-top; the next draw is at the back.
        order.insert(order.end(), top.rbegin(), top.rend());
        auto original = cards.discardPile;
        for (std::size_t i = 0; i < order.size(); ++i) {
            budget.tick(); cards.discardPile[i] = original[order[i]];
        }
        applied = true;
        return true;
    }
};
inline thread_local EventShuffle* activeEventShuffle = nullptr;
struct EventShuffleScope {
    EventShuffle* previous;
    explicit EventShuffleScope(EventShuffle* current) : previous(activeEventShuffle) {
        if (previous) throw Failure("nested_event_shuffle");
        activeEventShuffle = current;
    }
    ~EventShuffleScope() { activeEventShuffle = previous; }
    EventShuffleScope(const EventShuffleScope&) = delete;
};
inline bool conditionalShuffleEvent(sts::CardManager& cards) {
    return activeEventShuffle && activeEventShuffle->apply(cards);
}
}
