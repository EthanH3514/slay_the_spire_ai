#pragma once
#include "SimulatorTrace.h"
#include "Journal.h"
#include <map>

namespace spire_belief::trace {

// Diagnostic adapter only. Labels are assigned from sorted public inventories
// and subsequent public events, never from iteration over a private pile.
class Labels {
    std::map<Card, Label> entries_;
public:
    Label encode(const Card& card) {
        auto it = entries_.find(card);
        if (it != entries_.end()) return it->second;
        if (entries_.size() >= 4096) throw Failure("public_label_limit");
        auto label = Label(entries_.size()); entries_.emplace(card, label); return label;
    }
    Cards encode(Engine& engine, const Inventory& cards) {
        Cards out(&engine.arena);
        for (const auto& c : cards) out.push_back(encode(c));
        return out;
    }
    const auto& entries() const { return entries_; }
};

inline Event convert(Engine& engine, Labels& labels, const Record& record) {
    EventKind kind;
    switch (record.kind) {
        case Kind::Draw: kind = EventKind::DrawOrdered; break;
        case Kind::InsertBelowTop: kind = EventKind::InsertBelowTop; break;
        case Kind::PutTop: kind = EventKind::PutTop; break;
        case Kind::PutBottom: kind = EventKind::PutBottom; break;
        case Kind::Shuffle: kind = EventKind::Shuffle; break;
        case Kind::ImportShuffledTop: kind = EventKind::PrependShuffled; break;
        case Kind::Map: kind = EventKind::Map; break;
        default: throw Failure("event_contract");
    }
    if (kind == EventKind::Map) {
        if (!record.cards.empty() || record.mapping.empty() || !record.rule) throw Failure("event_contract");
        Event event(record.epoch, record.sequence, kind, &engine.arena);
        for (std::size_t i = 0; i < record.mapping.size(); ++i) {
            const auto& [before, after] = record.mapping[i];
            if (before == after || (i && !(record.mapping[i - 1].first < before))) throw Failure("event_contract");
            try {
                if (mappedProfile(before, *record.rule) != after) throw Failure("event_contract");
            } catch (const std::invalid_argument&) { throw Failure("event_contract"); }
            auto from = labels.encode(before);
            auto to = labels.encode(after);
            event.mapping.emplace_back(from, to);
        }
        return event;
    }
    if (!record.mapping.empty() || record.rule) throw Failure("event_contract");
    if (kind != EventKind::Shuffle && kind != EventKind::PrependShuffled && record.cards.size() != 1)
        throw Failure("event_contract");
    Event event(record.epoch, record.sequence, kind, &engine.arena);
    event.cards = labels.encode(engine, record.cards);
    return event;
}

class Replay {
    bool valid_ = true;
    std::uint64_t epoch_;
    static Id initial(Engine& engine, Labels& labels, const Recorder& trace) {
        if (!trace.gap.empty()) throw Failure(trace.gap.c_str());
        if (!trace.anchored) throw Failure("missing_shuffle_anchor");
        return engine.bag(labels.encode(engine, trace.initial));
    }
public:
    Engine engine;
    Labels labels;
private:
    Journal journal;
public:
    explicit Replay(const Recorder& trace, Limits limits = {})
        : epoch_(trace.epoch), engine(limits), journal(engine, initial(engine, labels, trace), trace.epoch) {}
    Id root() const {
        if (!valid_) throw Failure("stale_belief");
        return journal.root();
    }
    Rational drain(const Recorder& trace, const Inventory& actualAfter) {
        if (!valid_) throw Failure("stale_belief");
        engine.budget.begin();
        try {
            if (!trace.gap.empty()) throw Failure(trace.gap.c_str());
            if (!trace.anchored || epoch_ != trace.epoch || journal.nextSequence() > trace.events.size())
                throw Failure("event_gap");
            Vec<Event> events(&engine.arena);
            for (std::size_t i = journal.nextSequence(); i < trace.events.size(); ++i)
                events.push_back(convert(engine, labels, trace.events[i]));
            auto after = labels.encode(engine, actualAfter);
            return journal.apply(events, after);
        } catch (...) { valid_ = false; throw; }
    }
};

} // namespace spire_belief::trace
