#pragma once
#include "SimulatorMechanisms.h"
#include "ObservedDrawCosts.h"

namespace spire_belief::trace {
inline std::array<sts::CardId,3> discoveryEvent(sts::BattleContext& b,sts::CardType type) {
    auto* recorder=mechanismRecorder(b.cards,true);
    if(recorder&&recorder->eventOffer) {
        auto cards=recorder->eventOffer->consume(b.player.cc,type);
        recorder->eventOffer.reset(); // One historical event; future offers are not forced.
        return cards;
    }
    return sts::generateDiscoveryCards(b.cardRandomRng,b.player.cc,type);
}
inline void bothGap(const sts::CardManager& cards, const char* reason) {
    gap(cards, reason); mechanismHook(cards, [&](auto& r) { r.fail(reason); });
}
inline void bothCard(const sts::CardManager& cards, Kind kind, const sts::CardInstance& c, bool created = false) {
    card(cards, kind, c);
    mechanismHook(cards, [&](auto& r) { if (kind == Kind::Draw) r.drawn(c); else r.placed(kind, c, created); });
}
struct BothOperation {
    Operation legacy;
    MechanismOperation modern;
    BothOperation(const sts::CardManager& cards, Kind kind, const sts::CardInstance* c = nullptr, bool created = false)
        : legacy(cards, kind, [&] {
            if (kind == Kind::Shuffle) return inventory(cards.drawPile);
            if (kind == Kind::ImportShuffledTop) return inventory(cards.discardPile);
            return Inventory{project(*c)};
        }), modern(cards, kind, c, created) {}
};
struct MechanismAttribute {
    const sts::CardManager& cards;
    MechanismRecorder* recorder;
    Inventory expected;
    std::size_t cursor = 0;
    int exceptions = std::uncaught_exceptions();
    MechanismAttribute(const sts::CardManager& c, AttributeRule rule) : cards(c), recorder(mechanismRecorder(c)) {
        if (!recorder) return;
        try {
            for (const auto& card : inventory(c.drawPile)) expected.push_back(mappedProfile(card, rule));
            std::sort(expected.begin(), expected.end());
            MechanismRecord r{Mechanism::Map}; r.attribute = rule; recorder->emit(std::move(r)); cursor = recorder->records.size();
        } catch (...) { recorder->fail("producer_contract_failed"); throw; }
    }
    ~MechanismAttribute() {
        if (!recorder || !recorder->gap.empty()) return;
        if (std::uncaught_exceptions() > exceptions) { recorder->fail("producer_action_failed"); return; }
        try {
            if (cursor != recorder->records.size()) recorder->fail("attribute_map_interleaved_event");
            else if (inventory(cards.drawPile) != expected) recorder->fail("attribute_map_postcondition");
        } catch (...) { recorder->fail("producer_contract_failed"); }
    }
};
struct BothAttribute {
    AttributeOperation legacy;
    MechanismAttribute modern;
    BothAttribute(const sts::CardManager& cards, AttributeRule rule) : legacy(cards, rule), modern(cards, rule) {}
};
struct MechanismDraw : MechanismGroup {
    using MechanismGroup::MechanismGroup;
    void finish(const sts::CardInstance& c, const sts::BattleContext& b) {
        if (!recorder) return;
        ::spire_draw_costs::drawn(b.cards,c,b.player.hasStatus<PlayerStatus::CONFUSED>(),b.player.hasStatus<PlayerStatus::CORRUPTION>());
        recorder->drawn(c, b.player.hasStatus<PlayerStatus::CONFUSED>(), b.player.hasStatus<PlayerStatus::CORRUPTION>());
        --recorder->muted; recorder = nullptr;
    }
};
inline void publicDrawSelection(const sts::CardManager& cards, const sts::CardInstance& card) {
    mechanismHook(cards, [&](auto& r) {
        if (r.selection) {
            if (r.selection->first != project(card)) throw Failure("public_selector_mismatch");
            // A selector describes ONE transfer, not the rest of executeActions.
            // Duplication can queue another automatic fetch immediately.
            const auto rank = r.selection->second;
            r.selection.reset();
            r.removed(card, Selector::PublicBottomRank, rank);
        } else {
            unsigned count = 0; for (const auto& other : cards.drawPile) if (project(other) == project(card)) ++count;
            if (count != 1) throw Failure("missing_selector_contract");
            r.removed(card, Selector::PublicBottomRank, 0);
        }
    }, true);
}
inline Handle publicReference(const sts::CardManager& cards, const sts::CardInstance& c) {
    Handle result = 0; mechanismHook(cards, [&](auto& r) { result = r.reference(c); }); return result;
}
inline Handle publicTrack(const sts::CardManager& cards, Handle source) {
    Handle result = 0; mechanismHook(cards, [&](auto& r) { result = r.track(source); }); return result;
}
} // namespace spire_belief::trace

#undef SPIRE_DRAW_GAP
#undef SPIRE_DRAW_CARD
#undef SPIRE_DRAW_OPERATION
#undef SPIRE_DRAW_SHUFFLE
#undef SPIRE_DRAW_IMPORT
#undef SPIRE_DRAW_ATTRIBUTE_MAP
#define SPIRE_DRAW_GAP(cards, why) ::spire_belief::trace::bothGap(cards, why)
#define SPIRE_DRAW_LEGACY_GAP(cards, why) ::spire_belief::trace::gap(cards, why)
#define SPIRE_DRAW_CARD(cards, kind, c) ::spire_belief::trace::bothCard(cards, ::spire_belief::trace::Kind::kind, c)
#define SPIRE_DRAW_NEW_CARD(cards, kind, c) ::spire_belief::trace::bothCard(cards, ::spire_belief::trace::Kind::kind, c, true)
#define SPIRE_DRAW_OPERATION(cards, kind, c) ::spire_belief::trace::BothOperation spireDrawOperation(cards, ::spire_belief::trace::Kind::kind, &c)
#define SPIRE_DRAW_NEW_OPERATION(cards, kind, c) ::spire_belief::trace::BothOperation spireDrawOperation(cards, ::spire_belief::trace::Kind::kind, &c, true)
#define SPIRE_DRAW_SHUFFLE(cards) ::spire_belief::trace::BothOperation spireDrawOperation(cards, ::spire_belief::trace::Kind::Shuffle)
#define SPIRE_DRAW_IMPORT(cards) ::spire_belief::trace::BothOperation spireDrawOperation(cards, ::spire_belief::trace::Kind::ImportShuffledTop)
#define SPIRE_DRAW_ATTRIBUTE_MAP(cards, rule) ::spire_belief::trace::BothAttribute spireDrawAttributeOperation(cards, ::spire_belief::trace::AttributeRule::rule)
#define SPIRE_DRAW_OPENING(cards, game) ::spire_belief::trace::MechanismOpening spireDrawOpening(cards, game)
#define SPIRE_DRAW_GROUP(cards) ::spire_belief::trace::MechanismGroup spireDrawGroup(cards)
#define SPIRE_DRAW_BEGIN_OBSERVATION(cards) ::spire_belief::trace::MechanismDraw spireDrawObservation(cards)
#define SPIRE_DRAW_END_OBSERVATION(c, battle) spireDrawObservation.finish(c, battle)
#define SPIRE_DRAW_EXPOSE(cards, c) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.expose(c); })
#define SPIRE_DRAW_DISCARD(cards, c) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.discard(c); }, true)
#define SPIRE_DRAW_SETTLE(cards) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.settle(); })
#define SPIRE_DRAW_REMOVED(cards, c, selector) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.removed(c, ::spire_belief::trace::Selector::selector); }, true)
#define SPIRE_DRAW_SELECTED(cards, c) ::spire_belief::trace::publicDrawSelection(cards, c)
#define SPIRE_DRAW_REFERENCE(cards, c) ::spire_belief::trace::publicReference(cards, c)
#define SPIRE_DRAW_TRACK(cards, ref) ::spire_belief::trace::publicTrack(cards, ref)
#define SPIRE_DRAW_FORGET(cards, ref) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.forget(ref); })
#define SPIRE_DRAW_COPY_CHANGE(cards, ref, amount) ::spire_belief::trace::mechanismHook(cards, [&](auto& r) { r.modify(ref, ::spire_belief::trace::CopyRule::AddDamage, amount); })
#define SPIRE_DRAW_PRELUDE(battle, game) ::spire_belief::trace::mechanismHook((battle).cards, [&](auto& r) { if(r.openingPrelude)r.openingPrelude(battle,game); })
#define SPIRE_DISCOVERY_EVENT(battle, type) ::spire_belief::trace::discoveryEvent(battle, type)
