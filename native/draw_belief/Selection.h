#pragma once
#include "Mechanisms.h"

namespace spire_belief {

enum class ViewSource { Hand, Discard, Exhaust, Generated, DrawStable, DrawRandomized };
struct ViewCard {
    Label profile;
    Handle source = 0; // Previously observed public occurrence, never a UUID.
    unsigned tie = 0;  // Public comparator-equivalence bucket, in UI order.
};
struct ViewSpec {
    Handle id = 0;
    ViewSource source = ViewSource::Hand;
    std::vector<ViewCard> cards;
    std::vector<Label> eligible;
    unsigned minimum = 1, maximum = 1;
    bool bottomFirst = true;
    bool selectionOrderMatters = false;
};

// UI contracts are separate from card effects. Confirmation returns the chosen
// public occurrences in the action's specified order; the simulator continuation
// then executes the actual effects. No private-pile lookup is possible here.
class SelectionWindow {
    ViewSpec spec_;
    Handle lastId_ = 0;
    bool open_ = false;
    std::vector<unsigned> chosen_;
public:
    bool open() const { return open_; }
    const ViewSpec& spec() const { return spec_; }
    const auto& chosen() const { return chosen_; }
    Rational begin(MechanismBelief& b, const ViewSpec& spec) {
        if (open_ || !spec.id || spec.id <= lastId_ || spec.cards.size() > b.engine.budget.limits.cards ||
            spec.eligible.size() > b.engine.budget.limits.cards ||
            spec.minimum > spec.maximum || spec.maximum > spec.cards.size()) throw Failure("selector_contract");
        auto& e = b.engine; Cards displayed(&e.arena), eligible(spec.eligible.begin(), spec.eligible.end(), &e.arena);
        for (const auto& c : spec.cards) displayed.push_back(c.profile);
        Rational z(1);
        if (spec.source == ViewSource::DrawRandomized) {
            if (spec.minimum != 1 || spec.maximum != 1) throw Failure("selector_contract");
            z = b.openRandomizedView(spec.id, displayed, eligible);
        } else if (spec.source == ViewSource::DrawStable) {
            z = b.reconcileEligible(eligible, displayed);
            std::map<Label, unsigned> buckets;
            for (std::size_t i = 0; i < spec.cards.size(); ++i) {
                const auto& c = spec.cards[i];
                if (c.source || (i && c.tie < spec.cards[i - 1].tie)) throw Failure("selector_contract");
                auto inserted = buckets.emplace(c.profile, c.tie);
                if (!inserted.second && inserted.first->second != c.tie) throw Failure("selector_contract");
            }
            for (std::size_t i = 0; i < spec.cards.size(); ++i) {
                Cards group(&e.arena); unsigned rank = 0;
                for (const auto& p : buckets) if (p.second == spec.cards[i].tie) group.push_back(p.first);
                for (std::size_t j = 0; j < i; ++j) rank += spec.cards[j].tie == spec.cards[i].tie;
                z = Rational::multiply(z, b.observeRank(group, rank, spec.cards[i].profile, spec.bottomFirst), e.budget);
            }
        } else {
            std::set<Handle> seen;
            for (const auto& c : spec.cards) {
                if (spec.source == ViewSource::Generated ? c.source != 0 : (!c.source || !seen.insert(c.source).second))
                    throw Failure("selector_contract");
            }
        }
        spec_ = spec; lastId_ = spec.id; chosen_.clear(); open_ = true; return z;
    }
    void select(Handle id, unsigned index) {
        if (!open_ || id != spec_.id || index >= spec_.cards.size() || chosen_.size() >= spec_.maximum ||
            std::find(chosen_.begin(), chosen_.end(), index) != chosen_.end()) throw Failure("selector_contract");
        chosen_.push_back(index);
    }
    void unselect(Handle id, unsigned index) {
        if (!open_ || id != spec_.id) throw Failure("selector_contract");
        auto found = std::find(chosen_.begin(), chosen_.end(), index);
        if (found == chosen_.end()) throw Failure("selector_contract");
        chosen_.erase(found);
    }
    std::vector<ViewCard> confirm(MechanismBelief& b, Handle id, const std::vector<Handle>& outputs) {
        if (!open_ || id != spec_.id || chosen_.size() < spec_.minimum || chosen_.size() > spec_.maximum ||
            outputs.size() != chosen_.size()) throw Failure("selector_contract");
        auto& e = b.engine;
        auto order = chosen_; if (!spec_.selectionOrderMatters) std::sort(order.begin(), order.end());
        std::vector<unsigned> removed; std::vector<ViewCard> result;
        for (std::size_t i = 0; i < order.size(); ++i) {
            unsigned index = order[i]; const auto& card = spec_.cards[index]; Rational z(1);
            if (spec_.source == ViewSource::DrawRandomized) z = b.chooseRandomizedView(id, index, outputs[i]);
            else if (spec_.source == ViewSource::DrawStable) {
                Evidence event(Mechanism::RankedRemove, &e.arena);
                event.cards.push_back(card.profile); event.handle = outputs[i]; event.fromBottom = spec_.bottomFirst;
                for (unsigned j = 0; j < spec_.cards.size(); ++j) if (spec_.cards[j].tie == card.tie) {
                    event.eligible.push_back(spec_.cards[j].profile);
                    if (j < index && std::find(removed.begin(), removed.end(), j) == removed.end()) ++event.rank;
                }
                std::sort(event.eligible.begin(), event.eligible.end());
                event.eligible.erase(std::unique(event.eligible.begin(), event.eligible.end()), event.eligible.end());
                z = b.apply(event);
            } else {
                Evidence event(card.source ? Mechanism::Alias : Mechanism::Expose, &e.arena);
                event.handle = outputs[i];
                if (card.source) event.sources.push_back(card.source); else event.cards.push_back(card.profile);
                z = b.apply(event);
            }
            if (!(z == Rational(1))) throw Failure("selector_contract");
            removed.push_back(index); result.push_back({card.profile, outputs[i], card.tie});
        }
        open_ = false; chosen_.clear(); return result;
    }
    std::string certificate() const {
        std::string key;
        auto add = [&](std::uint64_t v) { for (unsigned i = 0; i < 8; ++i) key.push_back(char(v >> (i * 8))); };
        add(lastId_); add(open_);
        if (!open_) return key;
        add(spec_.id); add(unsigned(spec_.source)); add(spec_.minimum); add(spec_.maximum);
        add(spec_.bottomFirst); add(spec_.selectionOrderMatters); add(spec_.cards.size());
        for (const auto& c : spec_.cards) { add(c.profile); add(c.source); add(c.tie); }
        add(spec_.eligible.size()); for (auto c : spec_.eligible) add(c);
        add(chosen_.size()); for (auto i : chosen_) add(i); return key;
    }
};
} // namespace spire_belief
