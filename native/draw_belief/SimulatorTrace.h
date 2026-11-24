#pragma once

// Opt-in, diagnostic producer. This is deliberately not a search input API.
// The audited overlay includes these hooks; the production extension does not.
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "combat/CardManager.h"

namespace spire_belief::trace {

// Observable equivalence class, NEVER an instance ID or draw-pile position.
// [card enum, upgrade count, combat cost, turn cost, free once, base damage]
using Card = std::array<int, 6>;
using Inventory = std::vector<Card>;
using Mapping = std::vector<std::pair<Card, Card>>;
inline Card project(const sts::CardInstance& c) {
    // Do not silently merge an unexported per-copy attribute into another class.
    if (c.retain || (c.usesSpecialData() && c.id != sts::CardId::SEARING_BLOW &&
                    c.id != sts::CardId::RAMPAGE && c.id != sts::CardId::RITUAL_DAGGER))
        throw std::invalid_argument("unverified_public_card_attribute");
    int damage = 0;
    if (c.id == sts::CardId::RAMPAGE) damage = 8 + c.specialData;
    if (c.id == sts::CardId::RITUAL_DAGGER) damage = c.specialData;
    return {int(c.id), c.getUpgradeCount(), int(c.cost), int(c.costForTurn),
            int(c.freeToPlayOnce), damage};
}
template<class Pile> Inventory inventory(const Pile& pile) {
    Inventory out;
    out.reserve(pile.size());
    for (const auto& c : pile) out.push_back(project(c));
    std::sort(out.begin(), out.end());
    return out;
}

// Each rule is a deterministic function of a public equivalence class. It is
// NOT a before/after match by private UID, index, or guessed copy lineage.
enum class AttributeRule { UpgradeEligible, DamageCosts, ResetTurnCosts, CorruptionCosts };
inline const char* name(AttributeRule rule) {
    switch (rule) {
        case AttributeRule::UpgradeEligible: return "upgrade_eligible";
        case AttributeRule::DamageCosts: return "damage_costs";
        case AttributeRule::ResetTurnCosts: return "reset_turn_costs";
        case AttributeRule::CorruptionCosts: return "corruption_costs";
    }
    throw std::invalid_argument("invalid_attribute_rule");
}
inline Card mappedProfile(const Card& before, AttributeRule rule) {
    if (before[0] <= int(sts::CardId::INVALID) || before[0] > int(sts::CardId::ZAP) ||
        before[1] < 0 || before[1] > 32766 || before[2] < -128 || before[2] > 127 ||
        before[3] < -128 || before[3] > 127 || (before[4] != 0 && before[4] != 1))
        throw std::invalid_argument("invalid_public_card_attribute");
    // Reuse the simulator's single-card rules on a sanitized representative.
    // There is no original CardInstance (or private pile) available here.
    sts::CardInstance c(static_cast<sts::CardId>(before[0]));
    c.upgraded = before[1] != 0;
    c.cost = static_cast<std::int8_t>(before[2]);
    c.costForTurn = static_cast<std::int8_t>(before[3]);
    c.freeToPlayOnce = before[4] != 0;
    if (c.id == sts::CardId::SEARING_BLOW) c.specialData = static_cast<std::int16_t>(before[1]);
    if (c.id == sts::CardId::RAMPAGE || c.id == sts::CardId::RITUAL_DAGGER) {
        const auto special = std::int64_t(before[5]) - (c.id == sts::CardId::RAMPAGE ? 8 : 0);
        if (special < -32768 || special > 32767) throw std::invalid_argument("invalid_public_card_attribute");
        c.specialData = static_cast<std::int16_t>(special);
    }
    if (project(c) != before) throw std::invalid_argument("invalid_public_card_attribute");
    switch (rule) {
        case AttributeRule::UpgradeEligible: if (c.canUpgrade()) c.upgrade(); break;
        case AttributeRule::DamageCosts: if (c.isBloodCard()) c.tookDamage(); break;
        case AttributeRule::ResetTurnCosts: c.setCostForTurn(c.cost); break;
        case AttributeRule::CorruptionCosts:
            // ApplyPowerAction's modifyCostForCombat(-9), including already
            // discounted skills. This must not become an unconditional zero.
            if (c.getType() == sts::CardType::SKILL) {
                if (c.costForTurn > 0) c.cost = c.costForTurn = std::max(0, int(c.costForTurn) - 9);
                else if (c.cost >= 0) { c.cost = std::max(0, int(c.cost) - 9); c.costForTurn = 0; }
            }
            break;
        default: throw std::invalid_argument("invalid_attribute_rule");
    }
    return project(c);
}

enum class Kind { Draw, InsertBelowTop, PutTop, PutBottom, Shuffle, ImportShuffledTop, Map };
inline const char* name(Kind kind) {
    switch (kind) {
        case Kind::Draw: return "draw";
        case Kind::InsertBelowTop: return "insert";
        case Kind::PutTop: return "top";
        case Kind::PutBottom: return "bottom";
        case Kind::Shuffle: return "shuffle";
        case Kind::ImportShuffledTop: return "import_shuffled_top";
        case Kind::Map: return "map";
    }
    throw std::logic_error("invalid trace event");
}
struct Record {
    std::uint64_t epoch, sequence;
    Kind kind;
    Inventory cards;
    Mapping mapping;
    std::optional<AttributeRule> rule;
};

class Recorder;
inline thread_local Recorder* active = nullptr;
class Recorder {
    const sts::CardManager* target_;
    Recorder* previous_;
    std::size_t maxEvents_, maxCards_, payloadCards_ = 0;
    int muted_ = 0;
public:
    const std::uint64_t epoch;
    Inventory initial;
    std::vector<Record> events;
    std::string gap;
    bool anchored = false;
    explicit Recorder(const sts::CardManager& target, std::uint64_t e = 1,
                      std::size_t maxEvents = 4096, std::size_t maxCards = 65536)
        : target_(&target), previous_(active), maxEvents_(maxEvents), maxCards_(maxCards), epoch(e) {
        if (active) throw std::logic_error("nested public recorder");
        active = this;
    }
    ~Recorder() { active = previous_; }
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;
    bool accepts(const sts::CardManager& target) const {
        return target_ == &target && muted_ == 0 && gap.empty();
    }
    void fail(const char* reason) { if (gap.empty()) gap = reason; }
    void mute() { ++muted_; }
    void unmute() { --muted_; }
    void emit(Kind kind, Inventory cards) {
        if (!gap.empty()) return;
        if (kind == Kind::Map) { fail("event_contract"); return; }
        if (cards.size() > maxCards_ - payloadCards_ || events.size() >= maxEvents_) {
            fail("producer_limit"); return;
        }
        payloadCards_ += cards.size();
        if (!anchored) {
            // Only a real full shuffle establishes a fresh unknown-order law.
            // Merely attaching to an arbitrary mid-combat multiset does not.
            if (kind != Kind::Shuffle) { fail("missing_shuffle_anchor"); return; }
            initial = std::move(cards); anchored = true; return;
        }
        events.push_back({epoch, events.size(), kind, std::move(cards), {}, std::nullopt});
    }
    void emitMap(AttributeRule rule, Mapping mapping) {
        if (!gap.empty()) return;
        if (!anchored) { fail("missing_shuffle_anchor"); return; }
        // Two public descriptors per entry; no-op rules do not grow history.
        if (mapping.empty()) return;
        if (mapping.size() > (maxCards_ - payloadCards_) / 2 || events.size() >= maxEvents_) {
            fail("producer_limit"); return;
        }
        payloadCards_ += 2 * mapping.size();
        events.push_back({epoch, events.size(), Kind::Map, {}, std::move(mapping), rule});
    }
};
inline Recorder* observing(const sts::CardManager& cards) {
    return active && active->accepts(cards) ? active : nullptr;
}
inline void gap(const sts::CardManager& cards, const char* reason) {
    if (auto* r = observing(cards)) r->fail(reason);
}
inline void card(const sts::CardManager& cards, Kind kind, const sts::CardInstance& c) {
    if (auto* r = observing(cards)) {
        try { r->emit(kind, {project(c)}); }
        catch (...) { r->fail("producer_payload_failed"); throw; }
    }
}

// Semantic operation emits once and suppresses its lower-level implementation
// hooks. Neither an insertion index nor RNG is accepted by this interface.
class Operation {
    Recorder* recorder_ = nullptr;
    int exceptions_ = std::uncaught_exceptions();
public:
    template<class Payload> Operation(const sts::CardManager& cards, Kind kind, Payload payload) {
        recorder_ = observing(cards);
        if (recorder_) {
            try { recorder_->emit(kind, payload()); }
            catch (...) { recorder_->fail("producer_payload_failed"); throw; }
            recorder_->mute();
        }
    }
    Operation(const sts::CardManager& cards, Kind kind, Inventory payload)
        : Operation(cards, kind, [&] { return std::move(payload); }) {}
    ~Operation() {
        if (recorder_) {
            recorder_->unmute();
            if (std::uncaught_exceptions() > exceptions_) recorder_->fail("producer_action_failed");
        }
    }
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
};

// Audited whole-pile rules neither move cards nor trigger other draw operations.
// Validate the actual post-inventory against the public rule, then emit one Map.
// Unlike movement groups this does not mute hooks: interleaved events are a gap,
// not something a deterministic map is allowed to swallow.
class AttributeOperation {
    const sts::CardManager& cards_;
    Recorder* recorder_ = nullptr;
    AttributeRule rule_;
    Mapping mapping_;
    Inventory expected_;
    std::size_t cursor_ = 0;
    int exceptions_ = std::uncaught_exceptions();
public:
    AttributeOperation(const sts::CardManager& cards, AttributeRule rule) : cards_(cards), rule_(rule) {
        recorder_ = observing(cards);
        if (!recorder_) return;
        try {
            cursor_ = recorder_->events.size();
            auto before = inventory(cards.drawPile);
            for (const auto& card : before) {
                auto after = mappedProfile(card, rule);
                expected_.push_back(after);
                if (after != card && (mapping_.empty() || mapping_.back().first != card))
                    mapping_.emplace_back(card, after);
            }
            std::sort(expected_.begin(), expected_.end());
        } catch (...) { recorder_->fail("producer_payload_failed"); throw; }
    }
    ~AttributeOperation() {
        if (!recorder_ || !recorder_->gap.empty()) return;
        if (std::uncaught_exceptions() > exceptions_) { recorder_->fail("producer_action_failed"); return; }
        if (cursor_ != recorder_->events.size()) { recorder_->fail("attribute_map_interleaved_event"); return; }
        try {
            if (inventory(cards_.drawPile) != expected_) { recorder_->fail("attribute_map_postcondition"); return; }
            recorder_->emitMap(rule_, std::move(mapping_));
        } catch (...) { recorder_->fail("producer_payload_failed"); }
    }
    AttributeOperation(const AttributeOperation&) = delete;
    AttributeOperation& operator=(const AttributeOperation&) = delete;
};

// Only actual shuffle operations call this. The shuffled realization is never
// serialized. A discard transfer onto a nonempty pile shuffles ONLY its imports.
inline Kind importKind(const sts::CardManager& cards) {
    (void)cards;
    return Kind::ImportShuffledTop;
}

} // namespace spire_belief::trace

#define SPIRE_DRAW_GAP(cards, reason) ::spire_belief::trace::gap(cards, reason)
#define SPIRE_DRAW_CARD(cards, kind, c) ::spire_belief::trace::card(cards, ::spire_belief::trace::Kind::kind, c)
#define SPIRE_DRAW_OPERATION(cards, kind, c) \
    ::spire_belief::trace::Operation spireDrawOperation(cards, ::spire_belief::trace::Kind::kind, [&] { return ::spire_belief::trace::Inventory{::spire_belief::trace::project(c)}; })
#define SPIRE_DRAW_SHUFFLE(cards) \
    ::spire_belief::trace::Operation spireDrawOperation(cards, ::spire_belief::trace::Kind::Shuffle, [&] { return ::spire_belief::trace::inventory((cards).drawPile); })
#define SPIRE_DRAW_IMPORT(cards) \
    ::spire_belief::trace::Operation spireDrawOperation(cards, ::spire_belief::trace::importKind(cards), [&] { return ::spire_belief::trace::inventory((cards).discardPile); })
#define SPIRE_DRAW_ATTRIBUTE_MAP(cards, rule) \
    ::spire_belief::trace::AttributeOperation spireDrawAttributeOperation(cards, ::spire_belief::trace::AttributeRule::rule)
