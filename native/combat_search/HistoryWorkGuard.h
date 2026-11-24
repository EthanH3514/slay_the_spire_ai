#pragma once
#include "draw_belief/Continuation.h"
#include <algorithm>

namespace spire_session {
// A deterministic construction safety allowance, NOT extra tree/search work.
// This bounds retries too; it does not promise that every posterior can finish.
struct HistoryWorkGuard {
    static constexpr auto version = "public-history-byte-work-guard-v1";
    static constexpr std::size_t minimum = 8000000, maximum = 128000000;
    static constexpr std::size_t perWorld = 4096, replayAllowance = 4;
    std::size_t bytes = 0, work = minimum;

    template<class J>
    HistoryWorkGuard(const std::vector<spire_belief::PublicStep>& history, const J& request) {
        using spire_belief::Failure;
        if (request.value("history_work_guard", "") != version ||
            request.at("max_samples") != 1024 || request.at("max_value_evaluations") != 1024 ||
            request.at("max_branch_steps") != 16384)
            throw Failure("history_work_guard_contract");
        const spire_belief::ReplayLimits limits;
        if (history.size() > limits.steps) throw Failure("continuation_limit");
        auto add = [&](std::size_t n) {
            if (n > limits.observationBytes - bytes) throw Failure("continuation_limit");
            bytes += n;
        };
        for (const auto& step : history) {
            add(step.action.size());
            for (const auto& observation : step.observations) add(observation.size());
        }
        // Checked byte bound above makes the arithmetic safe even on 32-bit.
        const auto allowance = perWorld + replayAllowance * bytes;
        work = allowance > maximum / 1024 ? maximum :
            std::max(minimum, allowance * 1024);
    }
};
} // namespace spire_session
