#pragma once
#include "DrawPrefixCondition.h"

namespace spire_session {
// Exact necessary-event conditioning across deterministic, order-preserving
// public attribute maps. Keep ORIGINAL profile buckets distinct: two different
// cards that map to the same observed class are not interchangeable preimages.
// Actual worlds still replay every map/callback and pass full joint equality.
struct MappedDrawPrefixCondition : DrawPrefixCondition {
    static constexpr auto version = "public-mapped-draw-prefix-v1";
    std::size_t mappedEvents = 0;
    void prepare(spire_belief::trace::MechanismReplay& prior, const J& inventory,
                 const std::vector<std::string>& observations, spire_belief::Budget& work) {
        if (initialized) return;
        using namespace spire_belief;
        std::vector<J> events;
        bool hasMap = false;
        for (const auto& encoded : observations) {
            work.tick(encoded.size()+1);
            auto event = J::parse(encoded);
            const int kind = event.value("kind", -1);
            if (kind != int(Mechanism::Map) && kind != int(Mechanism::Draw)) break;
            hasMap |= kind == int(Mechanism::Map);
            events.push_back(std::move(event));
            if (events.size() == 4096) break;
        }
        if (!hasMap) { DrawPrefixCondition::prepare(prior, inventory, observations, work); return; }
        std::map<Profile, unsigned> remaining;
        std::map<Profile, Profile> transformed;
        for (const auto& card : inventory) {
            work.tick(); auto p=card.get<Profile>(); ++remaining[p]; transformed[p]=p;
        }
        std::vector<std::string> originalDraws;
        for (const auto& event : events) {
            if (originalDraws.size() == 64) break;
            if (event.at("kind") == int(Mechanism::Map)) {
                const int attribute=event.at("attribute").get<int>();
                if (attribute < int(trace::AttributeRule::UpgradeEligible) ||
                    attribute > int(trace::AttributeRule::CorruptionCosts))
                    throw Failure("public_draw_map_contract");
                if (!event.at("cards").empty()) throw Failure("public_draw_map_contract");
                for (auto& [original, current] : transformed) if (remaining.at(original)) {
                    work.tick(); current=trace::mappedProfile(current,static_cast<trace::AttributeRule>(attribute));
                }
                ++mappedEvents; continue;
            }
            if (event.at("cards").size()!=1) throw Failure("public_draw_prefix_contract");
            const auto observed=event.at("cards").at(0).get<Profile>();
            const bool confused=event.at("confused"),corruption=event.at("corruption");
            std::optional<Profile> unique;
            bool ambiguous=false;
            for (const auto& [original,count] : remaining) if (count) {
                const auto& current=transformed.at(original);bool possible=false;
                for (int roll=0;roll<(confused&&current[2]>=0?4:1);++roll) {
                    work.tick();possible |= trace::drawProfile(current,roll,confused,corruption)==observed;
                }
                if (!possible) continue;
                if (unique) { ambiguous=true;break; }
                unique=original;
            }
            // Ambiguity is left to full joint replay, never assigned an arbitrary
            // identity or equalized probability. Earlier unique prefix is valid.
            if (ambiguous) break;
            if (!unique) throw Failure("impossible_public_draw_prefix");
            --remaining.at(*unique);
            originalDraws.push_back(J{{"kind",int(Mechanism::Draw)},{"cards",{*unique}},
                {"confused",false},{"corruption",false}}.dump());
        }
        // Condition the ORIGINAL anchor order, then put the original profiles
        // back. Never put post-damage/upgraded costs into the pre-action world.
        DrawPrefixCondition::prepare(prior,inventory,originalDraws,work);
    }
};
} // namespace spire_session
