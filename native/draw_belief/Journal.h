#pragma once

#include "Order.h"

namespace spire_belief {

// This is an input contract, not a claim that either game adapter already
// supplies these events. Producers must certify public observability, ordering,
// selector semantics and completeness before attaching this to live search.
enum class EventKind {
    DrawOrdered, DrawUnordered, InsertBelowTop, PutTop, PutBottom,
    EraseUnique, EraseUniformMatching, Shuffle, Map, RevealOrder,
    PrependShuffled,
    MissingEvidence, UnspecifiedSelector
};

struct Event {
    std::uint64_t epoch, sequence;
    EventKind kind;
    Cards cards;
    Vec<std::pair<Label, Label>> mapping;
    Event(std::uint64_t e, std::uint64_t s, EventKind k, std::pmr::memory_resource* resource)
        : epoch(e), sequence(s), kind(k), cards(resource), mapping(resource) {}
};

class Journal {
    Engine& engine_;
    Id root_;
    std::uint64_t epoch_, nextSequence_;
    bool valid_ = true;
public:
    Journal(Engine& engine, Id initial, std::uint64_t epoch, std::uint64_t nextSequence = 0)
        : engine_(engine), root_(initial), epoch_(epoch), nextSequence_(nextSequence) { engine_.node(initial); }
    bool valid() const { return valid_; }
    std::uint64_t nextSequence() const { return nextSequence_; }
    Id root() const {
        if (!valid_) throw Failure("stale_belief");
        return root_;
    }
    // A completed public transition publishes exactly one immutable new root.
    // Failed transitions invalidate the cursor, even when the old root survives.
    // Work/deadline starts outside this method, so event construction/validation
    // can share one caller-owned transition budget rather than resetting it here.
    Rational apply(const Vec<Event>& events, const Cards& expectedAfter) {
        if (!valid_) throw Failure("stale_belief");
        Id candidate = root_;
        auto sequence = nextSequence_;
        Rational likelihood(1);
        try {
            for (const auto& event : events) {
                engine_.budget.tick(event.cards.size() + event.mapping.size() + 1);
                if (event.epoch != epoch_ || event.sequence != sequence) throw Failure("event_gap");
                if (sequence == std::numeric_limits<std::uint64_t>::max()) throw Failure("event_gap");
                ++sequence;
                Result result{Rational(1), candidate};
                auto single = [&]() {
                    if (event.cards.size() != 1) throw Failure("event_contract");
                    return event.cards.front();
                };
                if (event.kind != EventKind::Map && !event.mapping.empty()) throw Failure("event_contract");
                switch (event.kind) {
                    case EventKind::DrawOrdered:
                        for (auto c : event.cards) {
                            auto one = engine_.pop(result.root, c);
                            result.likelihood = Rational::multiply(result.likelihood, one.likelihood, engine_.budget);
                            result.root = one.root;
                            if (!result.likelihood) break;
                        }
                        break;
                    case EventKind::DrawUnordered: result = engine_.unorderedDraw(candidate, event.cards); break;
                    case EventKind::InsertBelowTop: result.root = engine_.belowTop(candidate, single()); break;
                    case EventKind::PutTop: result.root = engine_.concat(engine_.word(event.cards), candidate); break;
                    case EventKind::PutBottom: result.root = engine_.concat(candidate, engine_.word(event.cards)); break;
                    case EventKind::EraseUnique: case EventKind::EraseUniformMatching: {
                        Label c = single(); const auto& cards = engine_.node(candidate).cards;
                        auto copies = std::count(cards.begin(), cards.end(), c);
                        if (!copies) throw Failure("event_contract");
                        if (event.kind == EventKind::EraseUnique && copies != 1) throw Failure("missing_selector_contract");
                        result.root = engine_.erase(candidate, c); break;
                    }
                    case EventKind::Shuffle: {
                        Cards inventory(event.cards, &engine_.arena); std::sort(inventory.begin(), inventory.end());
                        if (inventory != engine_.node(candidate).cards) throw Failure("inventory_mismatch");
                        result.root = engine_.bag(inventory); break;
                    }
                    case EventKind::PrependShuffled:
                        // A genuinely shuffled external pile is added above the
                        // old one. It does NOT reshuffle the old conditional law.
                        result.root = engine_.concat(engine_.bag(event.cards), candidate); break;
                    case EventKind::Map:
                        if (!event.cards.empty()) throw Failure("event_contract");
                        result.root = engine_.mapped(candidate, event.mapping); break;
                    case EventKind::RevealOrder: result = engine_.reveal(candidate, event.cards); break;
                    case EventKind::MissingEvidence: throw Failure("missing_public_evidence");
                    case EventKind::UnspecifiedSelector: throw Failure("missing_selector_contract");
                    default: throw Failure("event_contract");
                }
                if (!result.likelihood) throw Failure("impossible_evidence");
                likelihood = Rational::multiply(likelihood, result.likelihood, engine_.budget);
                candidate = result.root;
            }
            Cards after(expectedAfter, &engine_.arena); std::sort(after.begin(), after.end());
            engine_.budget.tick(after.size() + 1);
            if (after != engine_.node(candidate).cards) throw Failure("inventory_mismatch");
            root_ = candidate; nextSequence_ = sequence;
            return likelihood;
        } catch (...) {
            valid_ = false;
            throw;
        }
    }
};

} // namespace spire_belief
