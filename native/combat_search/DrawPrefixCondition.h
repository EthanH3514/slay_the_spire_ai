#pragma once
#include "Engine.h"
#include "../draw_belief/SimulatorMechanisms.h"

namespace spire_session {
// Exact necessary-event preconditioning, NOT a replacement for joint replay.
// If each initial observed draw has ONE possible pre-draw public profile, its
// original top-first prefix is known. Sample P(order | that prefix), keep the
// original independent RNG-seed law, then reject on ALL original observations.
// Ambiguous cost preimages are never guessed or collapsed: stop the prefix.
struct DrawPrefixCondition {
    using J = spire_search::J;
    using Profile = spire_belief::trace::Card;
    bool initialized = false;
    std::vector<Profile> prefix;
    std::unique_ptr<spire_belief::trace::MechanismReplay> order;

    void prepare(spire_belief::trace::MechanismReplay& prior, const J& inventory,
                 const std::vector<std::string>& observations, spire_belief::Budget& work) {
        if (initialized) return;
        initialized = true;
        if (observations.empty()) return;
        using namespace spire_belief;
        std::map<Profile, unsigned> remaining;
        for (const auto& card : inventory) { work.tick(); ++remaining[card.get<Profile>()]; }
        for (const auto& encoded : observations) {
            work.tick(encoded.size() + 1);
            auto event = J::parse(encoded);
            if (event.value("kind", -1) != int(Mechanism::Draw) || prefix.size() == 64) break;
            if (event.at("cards").size() != 1) throw Failure("public_draw_prefix_contract");
            const auto observed = event.at("cards").at(0).get<Profile>();
            const bool confused = event.at("confused"), corruption = event.at("corruption");
            std::optional<Profile> unique;
            bool ambiguous = false;
            for (const auto& [before, count] : remaining) {
                if (!count) continue;
                work.tick(); bool possible = false;
                const int rolls = confused && before[2] >= 0 ? 4 : 1;
                for (int roll = 0; roll < rolls; ++roll) {
                    work.tick();
                    possible |= trace::drawProfile(before, roll, confused, corruption) == observed;
                }
                if (!possible) continue;
                if (unique) { ambiguous = true; break; }
                unique = before;
            }
            if (ambiguous) break;
            if (!unique) throw Failure("impossible_public_draw_prefix");
            prefix.push_back(*unique); --remaining[*unique];
        }
        if (prefix.empty()) return;
        prior.belief.engine.budget.begin(work.deadline);
        try { order = prior.compact(); }
        catch (...) { work.tick(prior.belief.engine.budget.work); throw; }
        work.tick(prior.belief.engine.budget.work);
        auto& belief = order->belief;
        belief.engine.budget.begin(work.deadline);
        try {
            const auto first = belief.freshHandles(unsigned(prefix.size()));
            for (std::size_t i = 0; i < prefix.size(); ++i) {
                Evidence draw(Mechanism::Draw, &belief.engine.arena);
                draw.cards.push_back(order->publicLabel(prefix[i])); draw.handle = first + i;
                belief.apply(draw);
            }
            // Restore original profiles, not their randomized observed costs.
            // This keeps the factory anchor inventory and all original checks.
            for (std::size_t i = prefix.size(); i > 0; --i) {
                Evidence top(Mechanism::Top, &belief.engine.arena);
                top.cards.push_back(order->publicLabel(prefix[i - 1])); top.handle = first + i - 1;
                belief.apply(top);
                Evidence release(Mechanism::Release, &belief.engine.arena); release.handle = first + i - 1;
                belief.apply(release);
            }
        } catch (...) { work.tick(belief.engine.budget.work); throw; }
        work.tick(belief.engine.budget.work);
    }
};
} // namespace spire_session
