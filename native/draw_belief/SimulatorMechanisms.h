#pragma once
#include "SimulatorTrace.h"
#include "Mechanisms.h"
#include "EventOffer.h"
#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include <map>

namespace spire_belief::trace {

// Shared public draw channel, including the same-cost temporary-cost case.
// This describes observed profiles; it never reads or overrides simulator RNG.
inline Card drawProfile(Card before, int roll, bool confused, bool corruption) {
    if (confused && before[2] >= 0) {
        if (roll < 0 || roll > 3) throw Failure("public_draw_roll");
        if (roll != before[2]) before[2] = before[3] = roll;
        before[4] = 0;
    }
    if (corruption && sts::getCardType(sts::CardId(before[0])) == sts::CardType::SKILL && before[3] >= 0)
        before[3] = 0;
    return before;
}

enum class Selector { All, Attack, HighestRarity, PublicBottomRank };
enum class CopyRule { DiscardReset, AddDamage };
struct MechanismRecord {
    Mechanism kind;
    Inventory cards;
    std::vector<Handle> sources;
    Handle handle = 0;
    AttributeRule attribute = AttributeRule::ResetTurnCosts;
    CopyRule copy = CopyRule::DiscardReset;
    Selector selector = Selector::All;
    unsigned rank = 0;
    int amount = 0;
    bool confused = false, corruption = false;
    std::uint64_t epoch = 1, sequence = 0;
    std::string side; // Optional ordered PUBLIC callback evidence, never RNG.
};

class MechanismRecorder;
inline thread_local MechanismRecorder* mechanismActive = nullptr;
class MechanismRecorder {
    const sts::CardManager* target_;
    MechanismRecorder* previous_;
    bool ownsActivation_;
    std::map<int, Handle> visible_; // Adapter-local object references; never serialized.
    std::map<int, Handle> pendingDiscard_; // Captured at the PUBLIC discard, not late hidden lookup.
    Handle nextHandle_ = 1;
    bool managedLifetimes_ = false;
    std::set<Handle> liveHandles_;
    std::size_t payload_ = 0;
    std::size_t sideBytes_ = 0;
public:
    unsigned muted = 0;
    bool anchored = false;
    Inventory priority, other;
    std::vector<MechanismRecord> records;
    std::uint64_t firstSequence = 0;
    std::string gap;
    std::optional<std::pair<Card, unsigned>> selection;
    // Owned counterfactual replayers only. Never installed on the real source.
    std::optional<EventOffer> eventOffer;
    std::function<std::string()> sideProjector;
    std::function<void(const sts::BattleContext&,const sts::GameContext&)> openingPrelude;
    explicit MechanismRecorder(const sts::CardManager& target,bool activate=true,bool managedLifetimes=false)
        : target_(&target), previous_(mechanismActive), ownsActivation_(activate),managedLifetimes_(managedLifetimes) {
        if(activate) {
            if (mechanismActive) throw std::logic_error("nested mechanism recorder");
            mechanismActive = this;
        }
    }
    ~MechanismRecorder() { if(ownsActivation_)mechanismActive = previous_; }
    struct Scope {
        MechanismRecorder* previous;
        explicit Scope(MechanismRecorder& current):previous(mechanismActive) {
            if(previous&&previous!=&current)throw std::logic_error("nested mechanism recorder");
            mechanismActive=&current;
        }
        ~Scope(){mechanismActive=previous;}
        Scope(const Scope&)=delete;
    };
    MechanismRecorder(const MechanismRecorder&) = delete;
    MechanismRecorder(const sts::CardManager& target,const MechanismRecorder& other)
        :target_(&target),previous_(nullptr),ownsActivation_(false),visible_(other.visible_),
         pendingDiscard_(other.pendingDiscard_),nextHandle_(other.nextHandle_),managedLifetimes_(other.managedLifetimes_),
         liveHandles_(other.liveHandles_),payload_(other.payload_),
         sideBytes_(other.sideBytes_),muted(other.muted),anchored(other.anchored),priority(other.priority),
         other(other.other),records(other.records),firstSequence(other.firstSequence),gap(other.gap),selection(other.selection) {
        if(muted||selection||other.eventOffer)throw Failure("clone_during_public_operation");
        // Caller installs a projector bound to the NEW owner, never copies a
        // closure referencing another world.
    }
    bool targets(const sts::CardManager& cards) const { return target_ == &cards && gap.empty(); }
    void fail(const char* why) { if (gap.empty()) gap = why; }
    Handle fresh() {
        if(nextHandle_==std::numeric_limits<Handle>::max()||
           (managedLifetimes_?liveHandles_.size()>=4096:nextHandle_>4096))throw Failure("reference_limit");
        auto result=nextHandle_++;if(managedLifetimes_)liveHandles_.insert(result);return result;
    }
    void release(Handle handle) {
        if(!managedLifetimes_||!handle)return;
        if(liveHandles_.erase(handle)!=1)throw Failure("reference_contract");
        MechanismRecord record{Mechanism::Release};record.handle=handle;emit(std::move(record));
    }
    void pruneVisible(const std::set<int>& publicCards) {
        if(!managedLifetimes_)return;
        for(auto it=visible_.begin();it!=visible_.end();) {
            if(publicCards.count(it->first)){++it;continue;}
            release(it->second);it=visible_.erase(it);
        }
    }
    std::size_t liveReferences() const {return liveHandles_.size();}
    void emit(MechanismRecord record) {
        if (!gap.empty()) return;
        if (!anchored) { fail("missing_anchor"); return; }
        auto size = record.cards.size() + record.sources.size() + 1;
        if (records.size() >= 4096 || size > 65536 - payload_) { fail("producer_limit"); return; }
        record.sequence = firstSequence + records.size();
        if(sideProjector)record.side=sideProjector();
        if(record.side.size()>4*1024*1024-sideBytes_)throw Failure("producer_side_limit");
        sideBytes_+=record.side.size();
        payload_ += size; records.push_back(std::move(record));
    }
    void acknowledge(std::uint64_t through) {
        if(through<firstSequence||through>firstSequence+records.size())throw Failure("producer_ack_contract");
        auto count=std::size_t(through-firstSequence);
        for(std::size_t i=0;i<count;++i){payload_-=records[i].cards.size()+records[i].sources.size()+1;sideBytes_-=records[i].side.size();}
        records.erase(records.begin(),records.begin()+count);firstSequence=through;
    }
    void anchor(Inventory top, Inventory rest) {
        if (anchored) { fail("duplicate_anchor"); return; }
        if (top.size() + rest.size() > 4096) { fail("producer_limit"); return; }
        std::sort(top.begin(), top.end()); std::sort(rest.begin(), rest.end());
        priority = std::move(top); other = std::move(rest); anchored = true;
    }
    Handle expose(const sts::CardInstance& card, bool newlyObserved = false) {
        auto old = visible_.find(card.uniqueId);
        if (!newlyObserved && old != visible_.end()) return old->second;
        if(old!=visible_.end())release(old->second);
        auto h = fresh(); visible_[card.uniqueId] = h;
        MechanismRecord r{Mechanism::Expose}; r.cards = {project(card)}; r.handle = h; emit(std::move(r)); return h;
    }
    Handle reference(const sts::CardInstance& card) {
        auto old = visible_.find(card.uniqueId);
        if (old != visible_.end()) return old->second;
        // A fresh generated/visible external card may be introduced, but never
        // invent public lineage by recognizing a card still hidden in the pile.
        for (const auto& c : target_->drawPile) if (c.uniqueId == card.uniqueId) throw Failure("missing_public_reference");
        return expose(card);
    }
    Handle track(Handle source) {
        if (!source || source >= nextHandle_) throw Failure("missing_public_reference");
        auto h = fresh(); MechanismRecord alias{Mechanism::Alias}; alias.handle = h; alias.sources = {source}; emit(std::move(alias));
        MechanismRecord record{Mechanism::Track}; record.handle = h; emit(std::move(record)); return h;
    }
    void forget(Handle handle) { if (handle) { MechanismRecord r{Mechanism::Forget}; r.handle = handle; emit(std::move(r)); release(handle); } }
    void modify(Handle handle, CopyRule rule, int amount = 0) {
        if (!handle) throw Failure("missing_public_reference");
        MechanismRecord r{Mechanism::Modify}; r.handle = handle; r.copy = rule; r.amount = amount; emit(std::move(r));
    }
    void discard(const sts::CardInstance& card) {
        auto source = reference(card);
        if (!pendingDiscard_.count(card.uniqueId)) pendingDiscard_[card.uniqueId] = track(source);
    }
    void settle() {
        if (pendingDiscard_.size() != target_->pendingDiscardCostResets.size()) throw Failure("missing_discard_lineage");
        for (auto uid : target_->pendingDiscardCostResets) if (!pendingDiscard_.count(uid)) throw Failure("missing_discard_lineage");
        // Event order is by public reference, not native UID/map ordering.
        std::vector<Handle> pending; for (const auto& p : pendingDiscard_) pending.push_back(p.second);
        std::sort(pending.begin(), pending.end());
        for (auto h : pending) { modify(h, CopyRule::DiscardReset); forget(h); }
        pendingDiscard_.clear();
    }
    void drawn(const sts::CardInstance& card, bool confused = false, bool corruption = false) {
        if(auto old=visible_.find(card.uniqueId);old!=visible_.end())release(old->second);
        auto h = fresh(); visible_[card.uniqueId] = h;
        MechanismRecord r{Mechanism::Draw}; r.cards = {project(card)}; r.handle = h;
        r.confused = confused; r.corruption = corruption; emit(std::move(r));
    }
    void placed(Kind kind, const sts::CardInstance& card, bool created = false) {
        MechanismRecord r{kind == Kind::PutTop ? Mechanism::Top : kind == Kind::PutBottom ? Mechanism::Bottom : Mechanism::Insert};
        r.cards = {project(card)}; r.handle = created ? 0 : reference(card);
        const auto handle=r.handle;emit(std::move(r));
        if (!created) {visible_.erase(card.uniqueId);release(handle);}
    }
    void imported() {
        std::vector<std::pair<Card, Handle>> inputs;
        for (const auto& c : target_->discardPile) inputs.emplace_back(project(c), reference(c));
        std::sort(inputs.begin(), inputs.end());
        MechanismRecord r{Mechanism::Import};
        for (const auto& p : inputs) { r.cards.push_back(p.first); r.sources.push_back(p.second); }
        emit(std::move(r));for(const auto& p:inputs)release(p.second);
        for (const auto& c : target_->discardPile) visible_.erase(c.uniqueId);
    }
    void removed(const sts::CardInstance& card, Selector selector, unsigned rank = 0) {
        if(auto old=visible_.find(card.uniqueId);old!=visible_.end())release(old->second);
        auto h = fresh(); visible_[card.uniqueId] = h;
        MechanismRecord r{selector == Selector::PublicBottomRank ? Mechanism::RankedRemove : Mechanism::RandomRemove};
        r.cards = {project(card)}; r.handle = h; r.selector = selector; r.rank = rank; emit(std::move(r));
    }
    void reveal(const sts::BattleContext& battle) {
        if (!targets(battle.cards)) throw Failure("recorder_target_mismatch");
        if (!battle.player.hasRelic<sts::RelicId::FROZEN_EYE>()) return;
        MechanismRecord r{Mechanism::Reveal};
        for (auto it = battle.cards.drawPile.rbegin(); it != battle.cards.drawPile.rend(); ++it) r.cards.push_back(project(*it));
        emit(std::move(r));
    }
};
inline MechanismRecorder* mechanismRecorder(const sts::CardManager& cards, bool explicitOperation = false) {
    auto* r = mechanismActive;
    return r && r->targets(cards) && (explicitOperation || !r->muted) ? r : nullptr;
}
template<class F> void mechanismHook(const sts::CardManager& cards, F f, bool explicitOperation = false) {
    if (auto* r = mechanismRecorder(cards, explicitOperation)) {
        try { f(*r); } catch (...) { r->fail("producer_contract_failed"); throw; }
    }
}
struct MechanismGroup {
    MechanismRecorder* recorder;
    int exceptions = std::uncaught_exceptions();
    explicit MechanismGroup(const sts::CardManager& cards) : recorder(mechanismRecorder(cards)) { if (recorder) ++recorder->muted; }
    ~MechanismGroup() {
        if (recorder) { --recorder->muted; if (std::uncaught_exceptions() > exceptions) recorder->fail("producer_action_failed"); }
    }
};
struct MechanismOpening : MechanismGroup {
    MechanismOpening(const sts::CardManager& cards, const sts::GameContext& game) : MechanismGroup(cards) {
        if (!recorder) return;
        Inventory priority, rest;
        for (int i = 0; i < game.deck.size(); ++i) {
            const auto& card = game.deck.cards[i];
            bool bottled = std::find(game.deck.bottleIdxs.begin(), game.deck.bottleIdxs.end(), i) != game.deck.bottleIdxs.end();
            (card.isInnate() || bottled ? priority : rest).push_back(project(sts::CardInstance(card)));
        }
        recorder->anchor(std::move(priority), std::move(rest));
    }
};
struct MechanismOperation : MechanismGroup {
    MechanismOperation(const sts::CardManager& cards, Kind kind, const sts::CardInstance* card = nullptr, bool created = false)
        : MechanismGroup(cards) {
        if (!recorder) return;
        try {
            if (kind == Kind::Shuffle) {
                if (!recorder->anchored) {
                    if (!cards.pendingDiscardCostResets.empty()) throw Failure("missing_discard_lineage");
                    recorder->anchor({}, inventory(cards.drawPile));
                }
                else recorder->emit(MechanismRecord{Mechanism::Shuffle});
            } else if (kind == Kind::ImportShuffledTop) recorder->imported();
            else if (card) recorder->placed(kind, *card, created);
            else throw Failure("event_contract");
        } catch (...) { recorder->fail("producer_contract_failed"); throw; }
    }
};
// Explicit public native-canonical selector. The ordinal comes from the caller's
// public candidate choice, NEVER from reverse-inference of a private index.
inline void choosePublicDrawToHand(sts::BattleContext& battle, const Card& card, unsigned ordinalFromBottom) {
    auto* r = mechanismRecorder(battle.cards);
    if (!r) throw Failure("missing_public_recorder");
    if (r->selection) throw Failure("nested_public_selector");
    int selected = -1; unsigned rank = 0;
    for (int i = 0; i < int(battle.cards.drawPile.size()); ++i) if (project(battle.cards.drawPile[i]) == card) {
        if (rank++ == ordinalFromBottom) { selected = i; break; }
    }
    if (selected < 0) throw Failure("illegal_public_choice");
    r->selection = std::make_pair(card, ordinalFromBottom);
    try { battle.chooseDrawToHandCards(&selected, 1); r->selection.reset(); }
    catch (...) { r->selection.reset(); r->fail("producer_action_failed"); throw; }
}

class MechanismReplay {
    std::size_t cursor_ = 0;
    std::map<Card, Label> labels_;
    std::vector<Card> cards_;
    std::unique_ptr<MechanismBelief> owned_;
    MechanismReplay(std::unique_ptr<MechanismBelief> b,const MechanismReplay& prior)
        :cursor_(prior.cursor_),labels_(prior.labels_),cards_(prior.cards_),owned_(std::move(b)),belief(*owned_) {}
    Label encode(const Card& card) {
        auto found = labels_.find(card); if (found != labels_.end()) return found->second;
        if (cards_.size() >= 4096) throw Failure("public_label_limit");
        // Validate all fields on a sanitized representative before any enum
        // table lookup. Its result is intentionally not used as an observation.
        (void)mappedProfile(card, AttributeRule::ResetTurnCosts);
        auto id = Label(cards_.size()); cards_.push_back(card); labels_.emplace(card, id); return id;
    }
    Cards encode(const Inventory& input) { Cards out(&belief.engine.arena); for (const auto& c : input) out.push_back(encode(c)); return out; }
public:
    MechanismBelief& belief;
    explicit MechanismReplay(const MechanismRecorder& trace, Limits limits = {})
        :owned_(std::make_unique<MechanismBelief>(limits)),belief(*owned_) {
        if (!trace.gap.empty()) throw Failure(trace.gap.c_str());
        if (!trace.anchored) throw Failure("missing_anchor");
        auto priority = encode(trace.priority), other = encode(trace.other);
        belief.anchor(priority, other);
    }
    const Card& card(Label internalSymbol) const { return cards_.at(belief.profile(internalSymbol)); }
    const Card& publicCard(Label profile) const { return cards_.at(profile); }
    Label publicLabel(const Card& profile) {return encode(profile);}
    std::unique_ptr<MechanismReplay> compact() {
        return std::unique_ptr<MechanismReplay>(new MechanismReplay(belief.compact(),*this));
    }
    std::uint64_t nextSequence() const {return cursor_;}
    Rational drain(const MechanismRecorder& trace, const Inventory& after, bool reconcile=true) {
        belief.engine.budget.begin(); auto& budget = belief.engine.budget;
        Rational likelihood(1);
        try {
            if (!trace.gap.empty()) throw Failure(trace.gap.c_str());
            if (cursor_ < trace.firstSequence || cursor_ > trace.firstSequence+trace.records.size()) throw Failure("event_gap");
            for (; cursor_ < trace.firstSequence+trace.records.size(); ++cursor_) {
                const auto& r = trace.records[cursor_-trace.firstSequence]; Evidence event(r.kind, &belief.engine.arena);
                if (r.epoch != 1 || r.sequence != cursor_) throw Failure("event_gap");
                if (r.selector < Selector::All || r.selector > Selector::PublicBottomRank ||
                    r.copy < CopyRule::DiscardReset || r.copy > CopyRule::AddDamage)
                    throw Failure("event_contract");
                event.cards = encode(r.cards); event.handle = r.handle; event.sources.assign(r.sources.begin(), r.sources.end()); event.rank = r.rank;
                // Snapshot the public dictionary size; callbacks may intern new
                // transformed profiles, never inspect the simulator's pile.
                auto size = cards_.size(); int bestRarity = -1;
                if (r.kind == Mechanism::RandomRemove || r.kind == Mechanism::RankedRemove) {
                    // Highest-rarity eligibility is resolved per public inventory
                    // below; card type/rarity do not depend on hidden costs.
                    if (r.selector == Selector::HighestRarity) {
                        // Current inventory is needed, not all historical labels.
                        for (auto profile : belief.profiles()) {
                            auto rarity = sts::getCardRarity(sts::CardId(cards_.at(profile)[0]));
                            if (rarity == sts::CardRarity::RARE) bestRarity = 2;
                            else if (rarity == sts::CardRarity::UNCOMMON) bestRarity = std::max(bestRarity, 1);
                            else if (rarity == sts::CardRarity::COMMON) bestRarity = std::max(bestRarity, 0);
                        }
                    }
                    for (Label i = 0; i < size; ++i) {
                        auto id = sts::CardId(cards_[i][0]); bool eligible = true;
                        if (r.selector == Selector::Attack) eligible = sts::getCardType(id) == sts::CardType::ATTACK;
                        if (r.selector == Selector::PublicBottomRank) eligible = cards_[i] == r.cards.at(0);
                        if (r.selector == Selector::HighestRarity && bestRarity >= 0)
                            eligible = int(sts::getCardRarity(id)) == bestRarity;
                        if (eligible) event.eligible.push_back(i);
                    }
                    if (r.selector == Selector::PublicBottomRank) {
                        event.fromBottom = true;
                    }
                }
                Channel channel;
                if (r.kind == Mechanism::Draw) channel = [&](Label old, Label observed) {
                    Card before = cards_.at(old); const Card seen = cards_.at(observed);
                    int matches = 0, count = r.confused && before[2] >= 0 ? 4 : 1;
                    for (int roll = 0; roll < count; ++roll) {
                        auto next = drawProfile(before, roll, r.confused, r.corruption);
                        if (next == seen) ++matches;
                    }
                    return Rational::ratio(matches, count, budget);
                };
                Transform change;
                if (r.kind == Mechanism::Map) change = [&](Label old) { return encode(mappedProfile(cards_.at(old), r.attribute)); };
                if (r.kind == Mechanism::Modify) change = [&](Label old) {
                    auto next = cards_.at(old);
                    if (r.copy == CopyRule::DiscardReset) next[3] = next[2];
                    else {
                        if (next[0] != int(sts::CardId::RAMPAGE) && next[0] != int(sts::CardId::RITUAL_DAGGER)) throw Failure("event_contract");
                        next[5] += r.amount;
                    }
                    return encode(next);
                };
                likelihood = Rational::multiply(likelihood, belief.apply(event, channel, change), budget);
            }
            if(reconcile)likelihood = Rational::multiply(likelihood, belief.reconcile(encode(after)), budget);
            return likelihood;
        } catch (...) { belief.invalidate(); throw; }
    }
};

} // namespace spire_belief::trace
