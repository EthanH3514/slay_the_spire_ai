#pragma once
#include "TurnShufflePrefix.h"

namespace spire_run_sampling {
// Extend the existing event law to the first observed empty-deck shuffle in a
// player action. The public ordered import, not the triggering card's name,
// defines the shuffled inventory. Draws before the import retain their existing
// weighted draw-prefix law. Complete joint replay still checks every callback.
struct ObservedShufflePrefix : TurnShufflePrefix {
    static constexpr auto actionVersion = "public-action-shuffle-prefix-v1";
    bool actionConditioned = false;

    ObservedShufflePrefix(const J& anchor, const std::vector<spire_belief::PublicStep>& tail)
        : TurnShufflePrefix(anchor, tail) {
        using namespace spire_belief;
        // Preserve all previously reviewed predicates and proposal sequences.
        if (!prefix.empty() || tail.empty()) return;
        const auto action = J::parse(tail.front().action).value("kind", "");
        if (action.empty() || action == "end_turn" || action == "opening_prelude") return;
        const auto& observations = tail.front().observations;
        std::size_t imported = observations.size(), work = 0;
        for (std::size_t i = 0; i < observations.size(); ++i) {
            work += observations[i].size() + 1;
            const auto event = J::parse(observations[i]);
            const int kind = event.value("kind", -1);
            if (kind == int(Mechanism::Import)) { imported = i; break; }
            // No earlier insertion, removal, shuffle, or unreviewed operation.
            if (kind != int(Mechanism::Map) && kind != int(Mechanism::Draw)
                    && kind != int(Mechanism::Modify)) return;
        }
        if (imported == observations.size()) return;
        const auto event = J::parse(observations[imported]);
        const auto side = J::parse(event.at("side").get<std::string>());
        if (!side.contains("discard_order") || side.at("turn") != anchor.at("turn")) return;
        auto input = side.at("discard_order").get<std::vector<Profile>>();
        auto sorted = input;
        std::sort(sorted.begin(), sorted.end());
        if (J(sorted) != event.at("cards")) throw Failure("public_shuffle_input_inventory");
        std::vector<Profile> observed;
        for (std::size_t i = imported + 1; i < observations.size() && observed.size() < 64; ++i) {
            work += observations[i].size() + 1;
            const auto draw = J::parse(observations[i]);
            if (draw.value("kind", -1) != int(Mechanism::Draw)) break;
            if (draw.at("confused") || draw.at("corruption")) break;
            if (draw.at("cards").size() != 1) throw Failure("public_shuffle_draw_contract");
            observed.push_back(draw.at("cards").at(0).get<Profile>());
        }
        if (observed.empty()) return;
        for (const auto& card : observed) {
            auto found = std::find(sorted.begin(), sorted.end(), card);
            if (found == sorted.end()) throw Failure("public_shuffle_prefix_inventory");
            sorted.erase(found);
        }
        // Failed optional probes change neither the old budget nor statistics.
        setupWork += work;
        discard = std::move(input);
        prefix = std::move(observed);
        eventConditioned = actionConditioned = true;
    }
};
} // namespace spire_run_sampling
