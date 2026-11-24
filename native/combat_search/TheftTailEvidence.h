#pragma once
#include "sim/PublicTheftEvents.h"
namespace spire_session {
inline constexpr auto theftTailVersion="public-theft-tail-v1";
inline spire_search::J theftSnapshot(spire_search::J snapshot,
        const std::vector<sts::public_sampling::PublicTheftEvent>* events) {
    if(events && !events->empty()) {
        auto rows=spire_search::J::array();
        for(const auto& e:*events) rows.push_back({e.slot,e.generation,e.turn,int(e.id),int(e.intent),e.thievery,e.goldBefore,e.goldAfter});
        snapshot["public_theft_events"]=std::move(rows);
    }
    return snapshot;
}
}
