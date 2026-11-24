#pragma once
#include "Exact.h"
#include <memory>
#include <vector>

namespace spire_belief {

struct PublicStep {
    std::string action;
    // Complete public observations at their declared callback/settled point.
    // A sequence is compared as one joint observation, not multiplied as
    // independent card/HP/energy/power/relic likelihoods.
    std::vector<std::string> observations;
};
struct ReplayLimits {
    std::size_t attempts = 256, steps = 256, observationBytes = 1024 * 1024;
};
struct ReplayStats { std::size_t proposals = 0, rejected = 0, actions = 0; };

// Reconstruct paused queues by replaying from a RESOLVED public anchor. Do not
// resample a live opaque std::function queue or infer captured objects from
// current pile indices. Each accepted world owns its real simulator queue and
// any card references created during its own replay.
//
// Factory must sample the anchor law with independent proposal RNG; Advance
// executes one public action and returns ALL public observations produced by
// it, including the selection window and side-state boundaries. Neither gets
// the real environment or its audit state. This is bounded rejection sampling,
// not a finite retained-particle approximation. Limit exhaustion is explicit.
template<class World, class Factory, class Advance>
std::unique_ptr<World> replayContinuation(const std::vector<PublicStep>& history,
                                         Factory factory, Advance advance,
                                         Budget& budget, ReplayStats& stats,
                                         ReplayLimits limits = {}) {
    if (!limits.attempts || !limits.steps || !limits.observationBytes || history.size() > limits.steps)
        throw Failure("continuation_limit");
    std::size_t bytes = 0;
    for (const auto& step : history) {
        if (step.action.size() > limits.observationBytes - bytes) throw Failure("continuation_limit");
        bytes += step.action.size();
        for (const auto& observation : step.observations) {
            if (observation.size() > limits.observationBytes - bytes) throw Failure("continuation_limit");
            bytes += observation.size();
        }
    }
    for (std::size_t attempt = 0; attempt < limits.attempts; ++attempt) {
        budget.tick(); ++stats.proposals;
        auto world = factory(); if (!world) throw Failure("invalid_continuation_anchor");
        bool match = true;
        for (const auto& step : history) {
            budget.tick(); ++stats.actions;
            auto actual = advance(*world, step.action);
            std::size_t actualBytes = 0;
            for (const auto& observation : actual) {
                if (observation.size() > limits.observationBytes - actualBytes) throw Failure("continuation_limit");
                actualBytes += observation.size();
            }
            budget.tick(actualBytes + 1);
            if (actual != step.observations) { match = false; break; }
        }
        if (match) return world;
        ++stats.rejected;
    }
    throw Failure("continuation_proposal_limit");
}
} // namespace spire_belief
