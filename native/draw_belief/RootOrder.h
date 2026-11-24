#pragma once
#include "PublicHistory.h"

namespace spire_belief {

// Search owns a compact checkpoint, never the live consumer or its transport
// cursor. Resolved order sampling erases causal roles only when there is no live
// effect/view that could distinguish them. Paused roots use continuation replay
// from an earlier resolved anchor instead of assigning cards to opaque queues.
class RootOrder {
    std::unique_ptr<MechanismBelief> checkpoint_;
    Id root_;
    std::string certificate_;
public:
    RootOrder(PublicHistory& history, const CacheContext& context) {
        if (history.window().open() || history.belief().pendingRandomizedView() ||
            history.belief().hasActiveReferences()) throw Failure("unresolved_order_anchor");
        certificate_ = history.certificate(context);
        checkpoint_ = history.belief().compact();
        root_ = checkpoint_->publicRoot(); // Requires reconciled inventory.
    }
    const std::string& certificate() const { return certificate_; }
    std::vector<Label> sample(std::uint64_t seed, Budget& request) {
        request.tick();
        auto& e = checkpoint_->engine;
        e.budget.begin(request.deadline);
        // Independent request namespaces may change, but graph caching must
        // never turn the posterior into a retained finite particle reservoir.
        std::mt19937_64 random(seed);
        auto sample = e.sample(root_, random);
        std::vector<Label> result;
        result.reserve(sample.size());
        for (auto symbol : sample) result.push_back(checkpoint_->profile(symbol));
        if(e.budget.randomWords>request.limits.randomWords-request.randomWords)throw Failure("random_limit");
        request.randomWords+=e.budget.randomWords;
        request.tick(e.budget.work);
        return result; // Public profiles in TOP-FIRST sampled order; no UIDs.
    }
};
} // namespace spire_belief
