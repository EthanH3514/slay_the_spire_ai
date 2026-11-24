#pragma once
#include "Exact.h"
#include <algorithm>
#include <type_traits>
#include <vector>

namespace spire_belief {
inline constexpr auto rankedBagSampleVersion = "exact-ranked-bag-sampling-v1";

// Sample a uniform bag conditioned on the label at an eligible occurrence rank.
// Keep the old sorted-label categorical draws, including their RNG consumption.
// Erasure has two latent states: the selected occurrence is still pending, or
// has already disappeared between emitted cards. Their exact masses replace
// the old enumeration/caching of every derivative remainder.
template<class Engine, class Node, class Choose>
auto sampleRankedBag(Engine& engine, const Node& node, bool erase, Choose&& choose) {
    using Label = typename std::decay_t<decltype(node.cards)>::value_type;
    using Cards = std::pmr::vector<Label>;
    auto& budget = engine.budget;
    auto matched = [&](Label c) {
        return std::binary_search(node.order.begin(), node.order.end(), c);
    };
    Cards remaining(node.cards, &engine.arena), output(&engine.arena);
    output.reserve(remaining.size());
    unsigned rank = node.markers;
    Rational pending(1), cleared;
    budget.tick(remaining.size() * (node.order.size() + 1) + 1);
    // Removing the only eligible occurrence reveals no order among the others.
    // Still use categorical prefix draws, not a different shuffle/RNG schedule.
    if (erase && std::none_of(remaining.begin(), remaining.end(), matched)) {
        pending = Rational(); cleared = Rational(1);
    }
    while (!remaining.empty()) {
        budget.tick(remaining.size() * (node.order.size() + 1) + 1);
        Cards options(remaining, &engine.arena);
        options.erase(std::unique(options.begin(), options.end()), options.end());
        std::pmr::vector<Rational> masses(&engine.arena), stays(&engine.arena), exits(&engine.arena);
        const auto length = remaining.size();
        const auto total = length + unsigned(erase);
        unsigned eligible = unsigned(erase), selected = unsigned(erase);
        for (auto c : remaining) { eligible += unsigned(matched(c)); selected += unsigned(c == node.label); }
        for (auto c : options) {
            budget.tick(remaining.size() + 1);
            const auto copies = std::count(remaining.begin(), remaining.end(), c);
            Rational stay, exit;
            if (!pending) {
                exit = Rational::ratio(copies, length, budget);
            } else if (rank) {
                // Before the selected eligible rank, all probability is pending.
                stay = Rational::ratio(copies + unsigned(erase && c == node.label), total, budget);
                if (matched(c)) {
                    auto future = Rational::ratio(selected - unsigned(c == node.label), eligible - 1, budget);
                    stay = Rational::divide(Rational::multiply(stay, future, budget),
                                            Rational::ratio(selected, eligible, budget), budget);
                }
            } else if (erase) {
                if (!matched(c)) stay = Rational::multiply(pending, Rational::ratio(copies, total, budget), budget);
                auto silent = Rational::multiply(pending, Rational::ratio(eligible, total, budget), budget);
                exit = Rational::multiply(Rational::add(cleared, silent, budget),
                                           Rational::ratio(copies, length, budget), budget);
            } else {
                if (c == node.label) exit = Rational::ratio(eligible, length, budget);
                else if (!matched(c)) stay = Rational::ratio(copies, length, budget);
            }
            stays.push_back(stay); exits.push_back(exit);
            masses.push_back(Rational::add(stay, exit, budget));
        }
        const auto chosen = choose(masses);
        const auto label = options.at(chosen);
        if (!masses.at(chosen)) throw std::logic_error("sampled zero-mass bag selection");
        pending = Rational::divide(stays[chosen], masses[chosen], budget);
        cleared = Rational::divide(exits[chosen], masses[chosen], budget);
        if (rank && matched(label)) --rank;
        output.push_back(label);
        remaining.erase(std::lower_bound(remaining.begin(), remaining.end(), label));
    }
    return output;
}
} // namespace spire_belief
