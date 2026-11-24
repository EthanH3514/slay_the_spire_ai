#pragma once
#include "Engine.h"

namespace spire_session {
inline constexpr auto boundaryVersion="public-session-boundaries-v1";
inline constexpr auto discoveryEventVersion="public-event-offer-v2-discovery";

// Explicit complete-round contract. No candidate pruning or rule fallback.
// Large CARD_SELECT roots get two H48 rounds; all other roots retain flat512.
inline std::size_t boundaryWork(const spire_search::J& view, std::size_t count,
                                const spire_search::J& budget) {
    using spire_search::J;
    if(budget.value("session_boundaries","")!=boundaryVersion ||
       budget.value("root_model","")!=discoveryEventVersion)
        throw std::runtime_error("session_boundary_version");
    if(count<2||count>1024)throw std::runtime_error("session_boundary_action_limit");
    const bool large=view.value("input","")=="CARD_SELECT"&&count>128;
    const int samples=large?int(2*count):256;
    const int steps=large?int(96*count):512;
    if(budget.at("samples_per_action")!=2||budget.at("rollout_horizon")!=48||
       budget.at("max_samples")!=samples||budget.at("max_value_evaluations")!=samples||
       budget.at("max_branch_steps")!=steps)
        throw std::runtime_error("session_boundary_budget");
    return large?std::max<std::size_t>(8000000,128000*count):8000000;
}
} // namespace spire_session
