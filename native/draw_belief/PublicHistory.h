#pragma once
#include "Selection.h"
#include <deque>

namespace spire_belief {

struct PublicRecord {
    std::uint64_t sequence;
    std::string payload; // Canonical public bytes; transport/audit fields excluded.
};
struct CacheContext {
    std::string profileDictionary;
    std::string continuation;
    std::string request; // Search version/config/seed namespace, when used.
};

// A bounded, transactional consumer for overlapping non-destructive snapshots.
// Acknowledgement is separate from reads, so a repeated get-state cannot lose
// or apply evidence twice. This does not substitute for producer completeness.
class PublicHistory {
    std::uint64_t epoch_, next_ = 0;
    bool valid_ = true;
    std::unique_ptr<MechanismBelief> belief_;
    SelectionWindow window_;
    std::deque<PublicRecord> retained_;
    std::size_t retainedBytes_ = 0, byteLimit_, recordLimit_;
    std::string boundary_; // Full public state incl. side effects / pending input.
public:
    PublicHistory(std::uint64_t epoch, std::unique_ptr<MechanismBelief> belief,
                  std::size_t recordLimit = 1024, std::size_t byteLimit = 1024 * 1024)
        : epoch_(epoch), belief_(std::move(belief)), byteLimit_(byteLimit), recordLimit_(recordLimit) {
        if (!epoch || !belief_ || !byteLimit || !recordLimit) throw Failure("history_contract");
    }
    bool valid() const { return valid_; }
    std::uint64_t epoch() const { return epoch_; }
    std::uint64_t nextSequence() const { return next_; }
    std::uint64_t retainedFrom() const { return retained_.empty() ? next_ : retained_.front().sequence; }
    std::size_t retainedBytes() const { return retainedBytes_; }
    MechanismBelief& belief() { if (!valid_) throw Failure("stale_history"); return *belief_; }
    const SelectionWindow& window() const { if (!valid_) throw Failure("stale_history"); return window_; }
    const std::string& boundary() const { if (!valid_) throw Failure("stale_history"); return boundary_; }

    template<class Apply>
    Rational drain(std::uint64_t epoch, const std::vector<PublicRecord>& input,
                   const std::string& publicBoundary, Apply apply) {
        if (!valid_) throw Failure("stale_history");
        try {
            if (epoch != epoch_ || input.size() > recordLimit_ || publicBoundary.size() > byteLimit_)
                throw Failure("history_contract");
            std::size_t bytes = publicBoundary.size(), begin = input.size();
            std::uint64_t expected = next_;
            for (std::size_t i = 0; i < input.size(); ++i) {
                const auto& r = input[i];
                if (r.payload.size() > byteLimit_ - bytes) throw Failure("journal_limit");
                bytes += r.payload.size();
                if (i && (input[i - 1].sequence == UINT64_MAX || r.sequence != input[i - 1].sequence + 1)) throw Failure("event_gap");
                if (r.sequence < next_) {
                    if (r.sequence < retainedFrom()) throw Failure("replay_before_retained_checkpoint");
                    const auto& old = retained_.at(std::size_t(r.sequence - retainedFrom()));
                    if (r.payload != old.payload) throw Failure("conflicting_replay");
                } else {
                    if (r.sequence != expected || expected == UINT64_MAX) throw Failure("event_gap");
                    if (begin == input.size()) begin = i;
                    ++expected;
                }
            }
            if (begin == input.size()) {
                if (publicBoundary != boundary_) throw Failure("unrecorded_public_change");
                return Rational(1); // Already consumed: no second likelihood.
            }
            belief_->engine.budget.begin(belief_->engine.budget.deadline);
            auto candidate = belief_->compact(); auto window = window_;
            Rational z(1); // Copy + update share this one transition work cap.
            for (std::size_t i = begin; i < input.size(); ++i) {
                auto likelihood = apply(*candidate, window, input[i]);
                z = Rational::multiply(z, likelihood, candidate->engine.budget);
            }
            // Complete the transaction before publishing anything. A failure
            // below invalidates the cursor, never exposes a prefix posterior.
            auto retained = retained_; auto keptBytes = retainedBytes_;
            for (std::size_t i = begin; i < input.size(); ++i) {
                retained.push_back(input[i]); keptBytes += input[i].payload.size();
                while (retained.size() > recordLimit_ || keptBytes > byteLimit_) {
                    keptBytes -= retained.front().payload.size(); retained.pop_front();
                }
            }
            belief_ = std::move(candidate); window_ = std::move(window);
            retained_ = std::move(retained); retainedBytes_ = keptBytes; next_ = expected; boundary_ = publicBoundary;
            return z;
        } catch (...) { valid_ = false; throw; }
    }
    // For an explicit full acknowledgement only; ordinary reads never erase.
    void acknowledge(std::uint64_t epoch, std::uint64_t through) {
        if (!valid_ || epoch != epoch_ || through > next_) throw Failure("ack_contract");
        while (!retained_.empty() && retained_.front().sequence < through) {
            retainedBytes_ -= retained_.front().payload.size(); retained_.pop_front();
        }
    }
    std::string certificate(const CacheContext& context) {
        if (context.profileDictionary.empty() || context.continuation.empty() || context.request.empty())
            throw Failure("incomplete_cache_context");
        if (context.profileDictionary.size() > byteLimit_ || context.continuation.size() > byteLimit_ || context.request.size() > byteLimit_)
            throw Failure("cache_context_limit");
        auto base = belief().certificate(); std::string out(base.begin(), base.end());
        auto add = [&](std::uint64_t v) { for (unsigned i = 0; i < 8; ++i) out.push_back(char(v >> (i * 8))); };
        add(1); add(epoch_); add(next_); add(boundary_.size()); out += boundary_;
        auto window = window_.certificate(); add(window.size()); out += window;
        for (const auto* field : {&context.profileDictionary, &context.continuation, &context.request}) {
            add(field->size()); out += *field;
        }
        return out; // Digest may index a cache, but compare these bytes on hits.
    }
};
} // namespace spire_belief
