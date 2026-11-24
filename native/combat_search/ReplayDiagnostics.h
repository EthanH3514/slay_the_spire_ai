#pragma once
#include "Engine.h"
#include "../draw_belief/Continuation.h"

namespace spire_session {
// Diagnostic-only comparisons of PUBLIC observations. Never captures source
// RNG, hidden card order, native queues or sampled RNG states. Disabled unless
// the caller explicitly requests a trace; cannot influence acceptance.
struct ReplayDiagnostics {
    using J = spire_search::J;
    const std::vector<spire_belief::PublicStep>* history = nullptr;
    std::size_t cursor = 0, builds = 0, mismatches = 0;
    J examples = J::array();
    explicit ReplayDiagnostics(const std::vector<spire_belief::PublicStep>* h) : history(h) {}
    void begin() { if (history) { cursor = 0; ++builds; } }
    void observe(const std::vector<std::string>& actual) {
        if (!history || cursor >= history->size()) return;
        const auto index = cursor++;
        const auto& expected = history->at(index).observations;
        if (actual == expected) return;
        ++mismatches;
        if (examples.size() == 8) return;
        J differences = J::array();
        for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
            if (actual[i] == expected[i]) continue;
            // Bound diagnostic parsing; normal replay independently validates
            // its own observation quota and complete byte-for-byte equality.
            if (actual[i].size() + expected[i].size() > 1024 * 1024) break;
            auto diff = J::diff(J::parse(expected[i]), J::parse(actual[i]));
            if (diff.size() > 16) diff.erase(diff.begin() + 16, diff.end());
            differences.push_back({{"observation", i}, {"diff", diff}});
            if (differences.size() == 4) break;
        }
        examples.push_back({{"step", index}, {"expected_count", expected.size()},
            {"actual_count", actual.size()}, {"differences", differences}});
    }
    J result() const {
        J steps = J::array();
        if (history) for (const auto& step : *history) {
            J observations = J::array();
            for (const auto& item : step.observations) observations.push_back(J::parse(item));
            steps.push_back({{"action", J::parse(step.action)}, {"observations", observations}});
        }
        return {{"schema", "public-replay-diagnostics-v1"}, {"diagnostic_only", true},
            {"builds", builds}, {"mismatches", mismatches}, {"examples", examples}, {"history", steps}};
    }
};
} // namespace spire_session
