#pragma once
#include <nlohmann/json.hpp>

// Observational only: copies an existing root prior once. No inference, RNG,
// tree allocation, action filtering, pseudo-visits or search-work changes.
namespace spire_root_policy {
inline constexpr auto version = "public-root-policy-record-v1";
inline thread_local bool enabled = false;

class Record {
    const bool enabled_ = enabled;
    nlohmann::json rows_ = nlohmann::json::array();
public:
    bool active() const { return enabled_; }
    template<class Weights> void observe(unsigned depth, const Weights& weights) {
        if (!enabled_ || depth != 0 || !rows_.empty()) return;
        for (const auto& item : weights)
            rows_.push_back({{"action", nlohmann::json::parse(item.first->action)},
                             {"probability", item.second}});
    }
    nlohmann::json diagnostic() const {
        return {{"version", version}, {"rows", rows_},
                {"origin", "existing-native-root-prior-cache"},
                {"additional_inference_calls", 0}, {"additional_simulator_steps", 0}};
    }
};
}
