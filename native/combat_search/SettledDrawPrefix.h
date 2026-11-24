#pragma once
#include "MappedDrawPrefix.h"

namespace spire_session {
// A resolved anchor has no pending per-copy effects. Before the first pile
// insertion/import/removal, a newly discarded card cannot be in the remaining
// anchor draw pile. Its delayed discard-cost settlement therefore leaves that
// pile unchanged. Remove ONLY these callbacks from the necessary draw-prefix
// condition. The original callbacks and all their side evidence are still
// checked by complete joint replay; this never edits the observed history.
struct SettledDrawPrefixCondition : MappedDrawPrefixCondition {
    static constexpr auto settlementVersion = "public-settled-draw-prefix-v1";
    std::size_t settledEvents = 0;
    void prepare(spire_belief::trace::MechanismReplay& prior, const J& inventory,
                 const std::vector<std::string>& observations, spire_belief::Budget& work) {
        if (initialized) return;
        using namespace spire_belief;
        if (prior.belief.hasActiveReferences() || prior.belief.pendingRandomizedView())
            throw Failure("unresolved_settlement_prefix_anchor");
        std::vector<std::string> filtered;
        for (const auto& encoded : observations) {
            work.tick(encoded.size()+1);
            const auto event=J::parse(encoded); const int kind=event.value("kind",-1);
            if (kind==int(Mechanism::Modify) &&
                event.value("copy",-1)==int(trace::CopyRule::DiscardReset) &&
                event.value("amount",-1)==0 && event.at("cards").empty()) {
                ++settledEvents; continue;
            }
            filtered.push_back(encoded);
            if (kind!=int(Mechanism::Map) && kind!=int(Mechanism::Draw)) break;
        }
        MappedDrawPrefixCondition::prepare(prior,inventory,filtered,work);
    }
};
} // namespace spire_session
