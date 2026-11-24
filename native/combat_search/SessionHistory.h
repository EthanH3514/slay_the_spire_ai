#pragma once
#include "HistoryWorkGuard.h"
#include "SessionBoundaryLimits.h"
#include "ConditionalRoots.h"
#include "TreeSearch.h"
#include "StandaloneTree.h"
#include "FrozenTreeAudit.h"
#include "../diagnostics/ConfirmationAllocation.h"
#include "ContinuationAudit.h"
#include "FrozenRootSearch.h"
#include "RootRoundSearch.h"
#include "PublicSessionState.h"
#include "ReplayDiagnostics.h"
#include "DrawPrefixCondition.h"
#include "SettledDrawPrefix.h"
#include "AutoplayTargetPrefix.h"
#include "ShufflePrefixProbe.h"
#include "ObservedShufflePrefix.h"
#include "ConditionalDrawShuffle.h"
#include "ObservedDrawPrefix.h"
#include "draw_belief/EventShuffle.h"
#include "EnemyDeathHistory.h"
#include "TheftTailEvidence.h"
#include "../draw_belief/SimulatorHooks.h"
#include "game/Game.h"

#include "RunSampling.h"
#include "OfferReplay.h"

namespace spire_session {
using namespace spire_belief;
using spire_search::SearchRoot;
using spire_search::RootSeeds;
inline constexpr auto sessionVersion="public-session-search-v2";

inline void advanceObserved(BattleContext& b,const J& action,int& top,trace::MechanismRecorder& recorder) {
    trace::MechanismRecorder::Scope scoped(recorder);
    if(b.inputState==InputState::CARD_SELECT&&action.at("kind")=="choose"&&
        (b.cardSelectInfo.cardSelectTask==CardSelectTask::SECRET_WEAPON||
         b.cardSelectInfo.cardSelectTask==CardSelectTask::SECRET_TECHNIQUE)) {
        const auto group=spire_search::selectionMap(b,true).at(action.at("choice_index").get<int>());
        const int index=group.at(0);auto profile=trace::project(b.cards.drawPile.at(index));unsigned rank=0;
        for(int i=0;i<index;++i)if(trace::project(b.cards.drawPile[i])==profile)++rank;
        recorder.selection=std::make_pair(profile,rank);
    }
    try {spire_search::advance(b,action,top,true);recorder.selection.reset();}
    catch(...){recorder.selection.reset();recorder.fail("public_action_failed");throw;}
}

inline J side(const BattleContext& b) {
    spire_perf::Scope timing(spire_perf::Phase::Side);
    J out=J::object();out["player"]=player(b.player);out["monsters"]=J::array();
    out["discard_order"]=pile(b.cards.discardPile);
    out["turn"]=b.turn;out["outcome"]=int(b.outcome);
    for(int i=0;i<b.monsters.monsterCount;++i)out["monsters"].push_back(monster(b.monsters.arr[i],b));
    return out;
}
inline std::string encodeSide(const BattleContext& b,bool conditioningEvidence=true) {
    spire_perf::Scope timing(spire_perf::Phase::Side);
    // Preserve player validation before inspecting monsters, as in side().
    // Every event is encoded afresh; no cross-event/state/branch cache.
    const auto playerText=encodePlayer(b.player);
    PublicNumericJson out;
    if(conditioningEvidence){out.raw("{\"discard_order\":");
        out.raw(pile(b.cards.discardPile).dump());out.raw(",\"monsters\":[");}
    else out.raw("{\"monsters\":[");
    for(int i=0;i<b.monsters.monsterCount;++i){if(i)out.raw(",");encodeMonster(out,b.monsters.arr[i],b);}
    out.raw("],");out.field("outcome",int(b.outcome));out.raw(",\"player\":");out.raw(playerText);
    out.raw(",");out.field("turn",b.turn);out.raw("}");return std::move(out).finish();
}
inline std::vector<std::string> observations(const trace::MechanismRecorder& recorder,
                                            std::size_t begin,const BattleContext& b,const std::vector<public_sampling::PublicTheftEvent>* thefts=nullptr) {
    spire_perf::Scope timing(spire_perf::Phase::Observations);
    std::vector<std::string> out;
    if(!recorder.gap.empty())throw Failure(recorder.gap.c_str());
    for(std::size_t i=begin;i<recorder.records.size();++i) {
        const auto& r=recorder.records[i];
        // Alias lifetimes belong to each world's own adapter. Compare public
        // effects, not local alias numbers or incidental expose bookkeeping.
        if(r.kind==Mechanism::Expose||r.kind==Mechanism::Alias||r.kind==Mechanism::Track||
           r.kind==Mechanism::Forget||r.kind==Mechanism::Release)continue;
        J row=J::object();row["kind"]=int(r.kind);row["cards"]=r.cards;row["attribute"]=int(r.attribute);
        row["copy"]=int(r.copy);row["selector"]=int(r.selector);row["rank"]=r.rank;row["amount"]=r.amount;
        row["confused"]=r.confused;row["corruption"]=r.corruption;row["side"]=r.side;
        out.push_back(row.dump());
    }
    out.push_back(thefts&&!thefts->empty()?theftSnapshot(snapshot(b),thefts).dump():encodeSnapshot(b));
    return out;
}

// Public-only enemy memory, built by observations, never source miscInfo or an
// unobserved previous move. The reviewed native ledger handles variable damage,
// spawn identities and phases; simple encounters also need the last public move.
struct Enemies {
    public_sampling::PublicEnemyHistory reviewed;
    public_sampling::PublicEnemyFrame last{};
    std::array<MMID,5> previous{MMID::INVALID,MMID::INVALID,MMID::INVALID,MMID::INVALID,MMID::INVALID};
    bool initialized=false;
    void observe(const BattleContext& b, const std::vector<public_sampling::PublicTheftEvent>* events=nullptr) {
        auto next=public_sampling::publicEnemyFrame(b);
        if(initialized)for(int i=0;i<next.count;++i) {
            if(i>=last.count||next.slots[i].id!=last.slots[i].id||next.slots[i].generation!=last.slots[i].generation)
                previous[i]=MMID::INVALID;
            else if(publicEnemyMoveAdvanced(last.slots[i].gone,next.slots[i].gone,
                    next.turn!=last.turn,next.slots[i].intent!=last.slots[i].intent)) {
                const auto* recipe=public_sampling::findMonsterBehaviorRecipe(next.slots[i].id);
                if(!recipe||recipe->history!=public_sampling::IntentHistoryConstraint::FIXED_WITHOUT_HISTORY_SHIFT)
                    previous[i]=last.slots[i].intent;
            }
        }
        reviewed.observe(next,events);last=next;initialized=true;
    }
    void reconstruct(BattleContext& b,const RootSeeds& seeds) const {
        if(!initialized||!(public_sampling::publicEnemyFrame(b)==last))
            throw Failure("public_enemy_anchor_mismatch");
        for(int i=0;i<b.monsters.monsterCount;++i)b.monsters.arr[i].moveHistory[1]=previous[i];
        GameContext game{};game.cc=CharacterClass::IRONCLAD;game.screenState=ScreenState::BATTLE;
        if(b.player.hasRelic<RelicId::RUNIC_DOME>())throw Failure("hidden_enemy_intent");
        // Reuse only the reviewed enemy/RNG initialization. Order below is
        // replaced by the exact history sampler, NOT this independent shuffle.
        public_sampling::resampleCombatContinuation(game,b,seeds,-1,&reviewed);
    }
};

struct OwnedReplay {
    std::size_t conditioningSteps=0;
    bool conditioningEvidence=true;
    std::optional<trace::EventShuffle> eventShuffle;
    std::optional<spire_draw_costs::Program> drawCosts;
    std::size_t observedStep=0;
    trace::MechanismRecorder recorder;
    explicit OwnedReplay(BattleContext& b):recorder(b.cards,false,true) {
        recorder.anchor({},trace::inventory(b.cards.drawPile));
        // Only visible piles receive local aliases. No source IDs are imported.
        for(int i=0;i<b.cards.cardsInHand;++i)recorder.expose(b.cards.hand[i]);
        for(const auto* p:{&b.cards.discardPile,&b.cards.exhaustPile})for(const auto& c:*p)recorder.expose(c);
        recorder.sideProjector=[this,&b]{return encodeSide(b,conditioningEvidence);};
        recorder.acknowledge(recorder.firstSequence+recorder.records.size());
    }
};

class SessionHistory {
    spire_terminal_proof::Cache terminalProofCache_;
    BattleContext& source_;
    trace::MechanismRecorder recorder_;
    std::unique_ptr<trace::MechanismReplay> posterior_;
    Enemies enemies_;
    struct Anchor {
        J state;
        std::unique_ptr<trace::MechanismReplay> order;
        Enemies enemies;
        std::string graph;
        J opening;
        std::vector<Profile> openingPrefix;
        bool prefixOpen=true;
    };
    std::shared_ptr<Anchor> anchor_;
    std::shared_ptr<Anchor> opening_;
    std::vector<PublicStep> tail_;
    J last_,relics_;
    std::string failure_;
    std::uint64_t epoch_,revision_=0;
    double updateSeconds_=0;
    std::size_t consumed_=0;

    // Exact factorization for an observed immediate generation window: potion
    // use has no card-RNG consumer before Discovery (Toy Ornithopter only heals).
    // Condition THAT independent stream cheaply, then retain the full joint
    // replay check. Never replace an offer or copy the source's actual RNG.
    struct OfferCondition {
        CardType type=CardType::INVALID;
        std::array<CardId,3> observed{};
        std::map<std::uint64_t,std::uint64_t> accepted;
        std::size_t attempts=0,cacheHits=0;
        std::uint64_t sample(std::uint64_t namespaceSeed,Budget& budget) {
            if(auto i=accepted.find(namespaceSeed);i!=accepted.end()){++cacheHits;return i->second;}
            std::mt19937_64 proposals(namespaceSeed);
            while(true) {
                budget.tick();if(budget.randomWords==budget.limits.randomWords)throw Failure("random_limit");
                ++budget.randomWords;++attempts;auto proposal=proposals();Random rng(proposal);
                std::array<CardId,3> generated{};int count=0;bool possible=true;
                while(count<3&&possible) {
                    budget.tick();auto c=type==CardType::STATUS?getTrulyRandomColorlessCardInCombat(rng):
                        getTrulyRandomCardInCombat(rng,CharacterClass::IRONCLAD,type);
                    if(std::find(generated.begin(),generated.begin()+count,c)!=generated.begin()+count)continue;
                    generated[count]=c;possible=c==observed[count];++count;
                }
                if(possible&&count==3){accepted.emplace(namespaceSeed,proposal);return proposal;}
            }
        }
    };
    std::shared_ptr<OfferCondition> offerCondition(bool discoveryCard=false) const {
        if(tail_.empty())return {};
        const auto a=J::parse(tail_[0].action);
        const auto first=J::parse(tail_[0].observations.back());
        if(first.at("input")!=int(InputState::CARD_SELECT)||
           first.at("selection").value("task","")!="DISCOVERY")return {};
        auto condition=std::make_shared<OfferCondition>();
        for(std::size_t i=0;i<3;++i) {
            const auto name=first.at("selection").at("candidates").at(i).at("id");bool found=false;
            for(int c=1;c<=int(CardId::ZAP);++c)if(spire_search::key(getCardEnumName(static_cast<CardId>(c)))==name)
                {condition->observed[i]=static_cast<CardId>(c);found=true;break;}
            if(!found)throw Failure("unmapped_public_offer");
        }
        if(a.value("kind","")=="opening_prelude") {
            bool toolbox=false;
            for(const auto& r:anchor_->opening.at("relics")) {
                if(r[0]==int(RelicId::ENCHIRIDION))return {}; // consumes this stream first
                toolbox|=r[0]==int(RelicId::TOOLBOX);
            }
            if(!toolbox)return {};condition->type=CardType::STATUS;
            return condition;
        }
        if(discoveryCard&&a.value("kind","")=="play_card") {
            const auto index=a.at("card_index").get<std::size_t>();
            const auto& hand=anchor_->state.at("hand");
            if(index>=hand.size())throw Failure("discovery_public_card_index");
            if(hand.at(index).at(0)==int(CardId::DISCOVERY)) {
                // Event conditioning is applied at the actual generation hook;
                // earlier queued effects are still replayed and checked.
                condition->type=CardType::INVALID;return condition;
            }
            return {};
        }
        if(a.value("kind","")!="use_potion")return {};
        const auto id=static_cast<Potion>(anchor_->state.at("potions").at(a.at("potion_index").get<int>()).get<int>());
        switch(id){
            case Potion::ATTACK_POTION:condition->type=CardType::ATTACK;break;
            case Potion::SKILL_POTION:condition->type=CardType::SKILL;break;
            case Potion::POWER_POTION:condition->type=CardType::POWER;break;
            case Potion::COLORLESS_POTION:condition->type=CardType::STATUS;break;
            default:return {};
        }
        return condition;
    }

    static SearchRoot build(const std::shared_ptr<Anchor>& anchor,const RootSeeds& seeds,Budget& budget,
                            trace::MechanismReplay* conditionalOrder=nullptr, spire_run_sampling::Blueprint* blueprint=nullptr,
                            std::optional<Id> conditionedRoot=std::nullopt) {
        auto b=blueprint?blueprint->make(anchor->state,budget):resolved(anchor->state);
        if(!blueprint)budget.tick(anchor->state.dump().size()+1);
        // At this point no actual world, actual RNG, private queue or card ID
        // exists. Enemy latent data are reconstructed solely from public history.
        anchor->enemies.reconstruct(*b,seeds);
        auto& order=conditionalOrder?*conditionalOrder:*anchor->order;
        auto& belief=order.belief;
        if(belief.hasActiveReferences()||belief.pendingRandomizedView())throw Failure("unresolved_order_anchor");
        belief.engine.budget.begin(budget.deadline);
        auto root=conditionedRoot?*conditionedRoot:belief.publicRoot();std::mt19937_64 random(seeds[6]);
        auto sampled=blueprint?spire_run_sampling::sampleOrder(belief.engine,root,random):belief.engine.sample(root,random);
        if(belief.engine.budget.randomWords>budget.limits.randomWords-budget.randomWords)throw Failure("random_limit");
        budget.randomWords+=belief.engine.budget.randomWords;budget.tick(belief.engine.budget.work);
        // IDs are created from the sorted public multiset; order selects from
        // equal-profile buckets. Real latent copy identities never cross here.
        std::map<Profile,std::vector<CardInstance>> inventory;
        for(const auto& c:b->cards.drawPile)inventory[trace::project(c)].push_back(c);
        b->cards.drawPile.clear();
        for(auto it=sampled.rbegin();it!=sampled.rend();++it) {
            const auto& profile=order.card(*it);auto& copies=inventory[profile];
            if(copies.empty())throw Failure("sampled_inventory_mismatch");
            b->cards.drawPile.push_back(copies.back());copies.pop_back();
        }
        for(const auto& group:inventory)if(!group.second.empty())throw Failure("sampled_inventory_mismatch");
        restoreKnowledge(*b,anchor->state);
        if(snapshot(*b)!=anchor->state)throw Failure("complete_anchor_projection_mismatch");
        auto replay=std::make_shared<OwnedReplay>(*b);
        return {std::move(b),-1,std::move(replay)};
    }
    void checkpoint() {
        if(!public_state::effectsResolved(source_))return;
        if(posterior_->belief.hasActiveReferences()||posterior_->belief.pendingRandomizedView())
            throw Failure("unresolved_checkpoint_references");
        auto next=std::make_shared<Anchor>();
        next->state=last_;next->enemies=enemies_;next->order=posterior_->compact();
        auto bytes=next->order->belief.certificate();next->graph.assign(bytes.begin(),bytes.end());
        anchor_=std::move(next);tail_.clear();
    }
    void captureOpening(const BattleContext& b,const GameContext& game) {
        if(opening_||revision_)throw Failure("duplicate_opening_prelude");
        auto a=std::make_shared<Anchor>();a->state=snapshot(b);
        const auto extra=std::max(0,int(recorder_.priority.size())-b.player.cardDrawPerTurn);
        if(b.actionQueue.size!=(extra?1:0)||!b.cardQueue.isEmpty()||b.cards.cardsInHand)
            throw Failure("unexpected_opening_program");
        // A declared virtual resolved anchor BEFORE the initialization program.
        // Only the known extra-innate draw is pending here; no opaque queue is
        // copied or erased from a root. The owned replayer creates it below.
        a->state["input"]=int(InputState::PLAYER_NORMAL);a->state["resolved"]=true;
        a->enemies.observe(b);a->order=std::make_unique<trace::MechanismReplay>(recorder_);
        a->opening={{"room",int(game.curRoom)},{"last_room",int(game.lastRoom)},{"extra_draw",extra},{"relics",J::array()}};
        for(const auto& r:game.relics.relics)a->opening["relics"].push_back({int(r.id),r.data});
        opening_=std::move(a);
    }
    void conditionOpeningPrefix() {
        if(!anchor_||anchor_->opening.is_null()||!anchor_->prefixOpen)return;
        // Exact shuffle-stream preconditioning on an observed, untransformed
        // initial draw prefix. Stop at ANY map/insertion/shuffle/channel: later
        // effects are conditioned by the complete replay, never guessed away.
        auto& prefix=anchor_->openingPrefix;const auto before=prefix.size();
        for(const auto& r:recorder_.records) {
            if(r.kind==Mechanism::Expose||r.kind==Mechanism::Alias||r.kind==Mechanism::Release)continue;
            if(r.kind!=Mechanism::Draw||r.confused||r.corruption){anchor_->prefixOpen=false;break;}
            prefix.push_back(r.cards.at(0));
        }
        if(prefix.size()==before&& !anchor_->graph.empty())return;
        auto& p=*anchor_->order;auto& belief=p.belief;belief.engine.budget.begin();
        for(std::size_t i=0;i<prefix.size();++i) {
            Evidence e(Mechanism::Draw,&belief.engine.arena);e.cards.push_back(p.publicLabel(prefix[i]));e.handle=i+1;
            belief.apply(e);
        }
        for(std::size_t i=prefix.size();i>0;--i) {
            Evidence e(Mechanism::Top,&belief.engine.arena);e.cards.push_back(p.publicLabel(prefix[i-1]));e.handle=i;
            belief.apply(e);Evidence release(Mechanism::Release,&belief.engine.arena);release.handle=i;belief.apply(release);
        }
        anchor_->order=p.compact();auto bytes=anchor_->order->belief.certificate();
        anchor_->graph.assign(bytes.begin(),bytes.end());
    }
    void beginOpening(const std::vector<std::string>& observed) {
        anchor_=std::move(opening_);
        if(!anchor_)throw Failure("missing_opening_prelude");
        tail_.push_back({J{{"kind","opening_prelude"}}.dump(),observed});
    }
public:
    SessionHistory(BattleContext& source,std::uint64_t epoch,J relics)
        :source_(source),recorder_(source.cards,false,true),relics_(std::move(relics)),epoch_(epoch) {
        if(!epoch_)throw Failure("public_session_epoch");
        std::sort(relics_.begin(),relics_.end());
        recorder_.sideProjector=[this]{return encodeSide(source_);};
        recorder_.openingPrelude=[this](const auto& b,const auto& g){captureOpening(b,g);};
    }
    SessionHistory(BattleContext& source,SessionHistory& prior)
        :source_(source),recorder_(source.cards,prior.recorder_),enemies_(prior.enemies_),
         anchor_(prior.anchor_),tail_(prior.tail_),last_(prior.last_),relics_(prior.relics_),failure_(prior.failure_),
         epoch_(prior.epoch_),revision_(prior.revision_),updateSeconds_(prior.updateSeconds_),consumed_(prior.consumed_) {
        prior.fresh();if(prior.posterior_)posterior_=prior.posterior_->compact();
        recorder_.sideProjector=[this]{return encodeSide(source_);};
        // Anchors own mutable sampler scratch space. Do not share it across
        // cloned sessions that may execute on different threads.
        if(anchor_){auto a=std::make_shared<Anchor>();a->state=anchor_->state;a->enemies=anchor_->enemies;
            a->order=anchor_->order->compact();a->graph=anchor_->graph;a->opening=anchor_->opening;
            a->openingPrefix=anchor_->openingPrefix;a->prefixOpen=anchor_->prefixOpen;anchor_=std::move(a);}
        fresh();
    }
    void setRelics(J relics) {std::sort(relics.begin(),relics.end());relics_=std::move(relics);}
    J view() const {fresh();return policyView(source_,relics_);}
    trace::MechanismRecorder& recorder(){return recorder_;}
    bool anchored() const {return recorder_.anchored;}
    void commit(const J& action=J(),const std::vector<public_sampling::PublicTheftEvent>* events=nullptr) {
        const auto started=spire_search::Clock::now();
        if(!failure_.empty())throw Failure("stale_public_session");
        try {
            if(!recorder_.anchored)throw Failure("missing_battle_opening_journal");
            if(public_state::effectsResolved(source_)) {
                std::set<int> visible;
                for(int i=0;i<source_.cards.cardsInHand;++i)visible.insert(source_.cards.hand[i].uniqueId);
                for(const auto* p:{&source_.cards.discardPile,&source_.cards.exhaustPile})
                    for(const auto& c:*p)visible.insert(c.uniqueId);
                for(int i=0;i<source_.monsters.monsterCount;++i) {
                    const auto& m=source_.monsters.arr[i];
                    if(m.id==MonsterId::BRONZE_ORB&&!m.isDeadOrEscaped()&&m.hasStatus<MS::STASIS>())
                        visible.insert(source_.cards.stasisCards[std::min(1,i)].uniqueId);
                }
                recorder_.pruneVisible(visible);
            }
            recorder_.reveal(source_);
            if(!posterior_)posterior_=std::make_unique<trace::MechanismReplay>(recorder_);
            auto observed=observations(recorder_,0,source_,events);
            if(!revision_&&!public_state::effectsResolved(source_))beginOpening(observed);
            conditionOpeningPrefix();
            if(!action.is_null()&&anchor_)tail_.push_back({action.dump(),observed});
            posterior_->drain(recorder_,trace::inventory(source_.cards.drawPile));
            posterior_=posterior_->compact();
            last_=snapshot(source_);enemies_.observe(source_,events);++revision_;
            checkpoint();consumed_=posterior_->nextSequence();
            opening_.reset();
            recorder_.acknowledge(consumed_);
            updateSeconds_+=std::chrono::duration<double>(spire_search::Clock::now()-started).count();
        } catch(const std::exception& e){failure_=e.what();throw;}
    }
    void fresh() const {
        if(!failure_.empty())throw std::runtime_error("public_session_invalid:"+failure_);
        if(last_.is_null()||snapshot(source_)!=last_)throw Failure("stale_public_session");
    }
    J autoplayTargetScope() const {
        fresh();if(!anchor_)throw Failure("missing_resolved_session_anchor");
        return AutoplayTargetPrefix(J{{"autoplay_target_prefix",AutoplayTargetPrefix::version}},anchor_->state,tail_).diagnostic();
    }
    unsigned theftTailSteps() const {
        unsigned count=0;for(const auto& step:tail_)
            count+=J::parse(step.observations.back()).contains("public_theft_events");
        return count;
    }
    J status() const {
        return {{"version",sessionVersion},{"epoch",epoch_},{"revision",revision_},
            {"events",consumed_},{"anchor",bool(anchor_)},{"tail_actions",tail_.size()},
            {"failure",failure_},{"update_seconds",updateSeconds_},
            {"theft_tail_steps",theftTailSteps()},{"theft_events",enemies_.reviewed.publicTheftEvents()},
            {"theft_corrections",enemies_.reviewed.publicTheftCorrections()}};
    }
    J search(const J& expected,const J& candidates,const std::vector<RootSeeds>& seeds,
             const J& budget,const J& metadata,bool trace=false,bool historyTree=false,bool guardedTree=false,
             const J* overrideProbe=nullptr,bool pairedTree=false,const J* pairedAudit=nullptr,bool standalone=false,
             const J* continuationAudit=nullptr) {
        spire_perf::Scope timing(spire_perf::Phase::Search);
        const auto boundaryStarted=spire_search::Clock::now();
        if(budget.contains("continuation_audit"))throw Failure("continuation_audit_requires_separate_entry");
        if(standalone&&(historyTree||guardedTree||pairedTree||overrideProbe||pairedAudit))
            throw Failure("invalid_standalone_entry");
        const bool frozenRoot=budget.value("tree_algorithm","")==spire_tree::frozenRootVersion;
        const bool roundTree=budget.value("tree_algorithm","")==spire_tree::roundVersion;
        const bool deepQuarter=budget.value("tree_algorithm","")==spire_tree::deepQuarterVersion;
        const bool scoreOrder=budget.value("tree_algorithm","")==spire_tree::scoreOrderVersion;
        const bool rootObjective=budget.value("tree_algorithm","")==spire_tree::rootObjectiveVersion;
        const bool deepPrior=budget.value("tree_algorithm","")==spire_tree::deepPriorVersion;
        const bool rootRecency=budget.value("tree_algorithm","")==spire_tree::rootRecencyVersion;
        if(continuationAudit&&(!standalone||(!deepPrior&&budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)))
            throw Failure("continuation_audit_entry_contract");
        if(standalone!=(budget.value("tree_algorithm","")==spire_tree::standaloneVersion||frozenRoot||roundTree||deepQuarter||scoreOrder||rootObjective||deepPrior||rootRecency))
            throw Failure("standalone_entry_version_contract");
        if(pairedAudit&&(historyTree||guardedTree||pairedTree||overrideProbe))throw Failure("invalid_paired_audit_entry");
        if(pairedTree&&(!guardedTree||overrideProbe))throw Failure("invalid_paired_tree_entry");
        if(overrideProbe&&(!trace||!historyTree||!guardedTree))throw Failure("invalid_override_probe_entry");
        if(guardedTree&&!historyTree)throw Failure("guarded_tree_requires_tree_entry");
        const bool aligned=spire_objective::enabled(budget);
        if(aligned&&(pairedAudit||overrideProbe))throw Failure("objective_diagnostic_entry_not_reviewed");
        fresh();if(!anchor_)throw Failure("missing_resolved_session_anchor");
        const auto model=budget.value("root_model",std::string("native-seed-conditioned-v1"));
        const bool optimized=budget.value("run_sampling","")==spire_run_sampling::version;
        if(budget.contains("run_sampling")&&(!optimized||!standalone||model!=trace::eventOfferVersion))throw Failure("run_sampling_contract");
        const bool repaired=model==discoveryEventVersion;
        if(repaired&&(standalone||historyTree||continuationAudit||pairedAudit||overrideProbe))
            throw Failure("session_boundary_flat_entry_only");
        const bool eventModel=model==trace::eventOfferVersion||repaired;
        if(!eventModel&&model!="native-seed-conditioned-v1")throw Failure("unknown_root_probability_model");
        if(eventModel&&(overrideProbe||pairedAudit))throw Failure("event_model_diagnostic_entry_not_reviewed");
        auto current=policyView(source_,relics_);
        if(expected!=current||spire_search::actions(source_,true)!=candidates)throw Failure("stale_public_search_request");
        auto anchor=anchor_;
        const auto oldRepairWork=repaired?boundaryWork(current,candidates.size(),budget):8000000;
        std::optional<HistoryWorkGuard> historyGuard;
        if(budget.contains("history_work_guard")) {
            if(!standalone||!optimized||repaired)throw Failure("history_work_guard_entry");
            historyGuard.emplace(tail_,budget);
        }
        const auto repairWork=historyGuard?historyGuard->work:oldRepairWork;
        const bool offers=budget.value("offer_replay","")==spire_offer_replay::version;
        if(budget.contains("offer_replay")&&(!offers||!optimized))throw Failure("offer_replay_contract");
        auto program=offers?std::make_shared<spire_offer_replay::Program>(anchor->state,tail_):nullptr;
        auto offer=offers?std::shared_ptr<OfferCondition>{}:offerCondition(repaired||optimized);
        if(continuationAudit&&offer&&continuationAudit->value("version",std::string())!=spire_teacher::version)throw Failure("continuation_audit_pending_offer_unreviewed");
        std::optional<trace::EventOffer> eventOffer;
        if(eventModel&&offer)eventOffer.emplace(offer->type,offer->observed,repaired||optimized);
        auto diagnostic=std::make_shared<ReplayDiagnostics>(trace?&tail_:nullptr);
        if(budget.contains("autoplay_target_prefix")&&(!optimized||!standalone))throw Failure("autoplay_target_entry");
        auto autoplayTargets=std::make_shared<AutoplayTargetPrefix>(budget,anchor->state,tail_);
        auto drawPrefix=std::make_shared<SettledDrawPrefixCondition>();
        const bool boundaryTimed=budget.contains("boundary_value")&&!budget.at("max_seconds").is_null();
        auto boundaryOrder=std::make_shared<spire_boundary_value::OrderSnapshot<trace::MechanismReplay>>();
        auto blueprint=optimized?std::make_shared<spire_run_sampling::Blueprint>():nullptr;
        auto shufflePrefix=optimized?std::make_shared<spire_run_sampling::ObservedShufflePrefix>(anchor->state,tail_):nullptr;
        auto drawShuffle=optimized?std::make_shared<spire_draw_shuffle::Plan>(anchor->state,tail_):nullptr;
        auto observedPrefix=std::make_shared<spire_observed_prefix::Plan>(tail_,eventModel&&optimized&&anchor->opening.is_null());
        // Opening preludes have their own existing prefix factorization. Never
        // apply the same condition twice or infer across an unreviewed program.
        auto prefixEvents=anchor->opening.is_null()&&!tail_.empty()?tail_[0].observations:std::vector<std::string>{};
        bool costOpening=false;
        if(eventModel&&optimized&&!anchor->opening.is_null()&&!tail_.empty()) {
            for(const auto& encoded:tail_[0].observations) {
                const auto e=J::parse(encoded);
                if(e.value("kind",-1)==int(Mechanism::Draw)&&e.value("confused",false))costOpening=true;
            }
            if(costOpening)for(const auto& encoded:tail_[0].observations) {
                const auto e=J::parse(encoded);const int kind=e.value("kind",-1);
                if(kind==int(Mechanism::Expose)||kind==int(Mechanism::Alias)||kind==int(Mechanism::Release))continue;
                prefixEvents.push_back(encoded);
                if(kind!=int(Mechanism::Draw))break;
            }
        }
        auto observe=[anchor](const SearchRoot& r){return std::vector<std::string>{snapshot(*r.battle).dump(),anchor->opening.dump()};};
        auto advance=[anchor,diagnostic,program,historicalSteps=tail_.size()](SearchRoot& r,const std::string& encoded) {
            auto context=std::static_pointer_cast<OwnedReplay>(r.replayContext);
            if(!context)throw Failure("missing_owned_replay_context");
            context->conditioningEvidence=context->conditioningSteps++<historicalSteps;
            trace::EventShuffleScope eventShuffleScope(context->eventShuffle && context->observedStep==0 ? &*context->eventShuffle : nullptr);
            auto& recorder=context->recorder;const auto begin=recorder.records.size();
            std::vector<public_sampling::PublicTheftEvent> theftEvents;
            std::optional<public_sampling::PublicTheftScope> theftScope;
            if(context->conditioningEvidence)theftScope.emplace(*r.battle,theftEvents);
            const bool historical=program&&program->arm(context->observedStep,encoded,recorder);
            const auto action=J::parse(encoded);
            auto* costProgram=context->conditioningSteps==1&&context->drawCosts?&*context->drawCosts:nullptr;
            spire_draw_costs::Scope drawCostScope(costProgram,r.battle->cards);
            if(action.value("kind","")=="opening_prelude") {
                if(anchor->opening.is_null())throw Failure("missing_opening_program");
                auto& b=*r.battle;const auto& program=anchor->opening;
                GameContext game{};game.cc=CharacterClass::IRONCLAD;game.curHp=b.player.curHp;game.maxHp=b.player.maxHp;
                game.curRoom=static_cast<Room>(program.at("room").get<int>());
                game.lastRoom=static_cast<Room>(program.at("last_room").get<int>());
                for(const auto& relic:program.at("relics"))game.relics.add({static_cast<RelicId>(relic[0].get<int>()),relic[1].get<int>()});
                trace::MechanismRecorder::Scope scope(recorder);
                const int extra=program.at("extra_draw");if(extra)b.addToBot(Actions::DrawCards(extra));
                b.initRelics(game);b.player.energy+=b.player.energyPerTurn;
                b.inputState=InputState::EXECUTING_ACTIONS;b.executeActions();
            } else advanceObserved(*r.battle,action,r.knownTop,recorder);
            if(costProgram&&costProgram->active())throw Failure("unconsumed_observed_draw_cost_prefix");
            // A sampled target may end combat before the known window. Reset the
            // unused evidence and let full public replay reject that world.
            if(historical)recorder.eventOffer.reset();
            else if(recorder.eventOffer)throw Failure("unconsumed_public_offer_event");
            recorder.reveal(*r.battle);
            // Necessary final-state condition first: rejected worlds need no
            // serialization of callback history. Accepted worlds still
            // compare every ordered callback and final byte below.
            const bool finalMismatch=historical&&theftSnapshot(snapshot(*r.battle),&theftEvents)!=program->steps.at(context->observedStep-1).final;
            auto result=finalMismatch?std::vector<std::string>{}:observations(recorder,begin,*r.battle,&theftEvents);
            recorder.acknowledge(recorder.firstSequence+recorder.records.size());
            diagnostic->observe(result);
            return result;
        };
        spire_search::RootEvidence evidence{anchor->graph,eventModel?model:"public-session-anchor-v3-draw-prefix",stateVersion,
            "native-public-occurrences-v1",{anchor->state.dump(),anchor->opening.dump()},tail_,current,candidates,relics_,"IRONCLAD"};
        // The session must rebuild every candidate's owned pending program.
        // A 64-choice opening with two paired rounds exceeds the small 1M
        // diagnostic cap even at ~50 ms. Explicit fixed session-only caps:
        // 8M construction work / 1M random words, still under the controller's
        // unchanged deadline. Neither rollout steps nor sample counts change.
        Limits construction;construction.work=repairWork;construction.randomWords=1000000;
        ReplayLimits replayLimits;if(eventModel)replayLimits.attempts=trace::eventReplayAttempts;
        spire_search::ConditionalRoots roots(std::move(evidence),
            [anchor,offer,eventModel,eventOffer,diagnostic,drawPrefix,prefixEvents,blueprint,shufflePrefix,program,boundaryTimed,boundaryOrder,autoplayTargets,advance,costOpening,drawShuffle,observedPrefix](const RootSeeds& plan,Budget& work){auto conditioned=plan;
                diagnostic->begin();
                if(program)program->charge(work);
                if(drawShuffle)drawShuffle->prepare(work);
                auto& searchOrder=boundaryTimed?boundaryOrder->get(*anchor->order,work):*anchor->order;
                if(observedPrefix->eligible)observedPrefix->prepare(searchOrder,work);
                else drawPrefix->prepare(searchOrder,anchor->state.at("draw"),prefixEvents,work);
                if(offer&&!eventModel)conditioned[1]=offer->sample(plan[1],work);
                if(shufflePrefix&&!(drawShuffle&&drawShuffle->eligible)&&!shufflePrefix->eventConditioned)conditioned[5]=shufflePrefix->sample(plan[5],work);
                if(shufflePrefix&&shufflePrefix->eventConditioned&&!shufflePrefix->charged){
                    work.tick(shufflePrefix->setupWork);shufflePrefix->charged=true;}
                autoplayTargets->prepare(conditioned[1],drawPrefix->prefix.size(),work,[&](std::uint64_t cardSeed){
                    auto trial=conditioned;trial[1]=cardSeed;diagnostic->begin();
                    return build(anchor,trial,work,drawPrefix->order?drawPrefix->order.get():&searchOrder,blueprint.get());
                },advance);
                conditioned[1]=autoplayTargets->sample(conditioned[1],work);diagnostic->begin();
                auto root=build(anchor,conditioned,work,observedPrefix->eligible?observedPrefix->order.get():
                    (drawPrefix->order?drawPrefix->order.get():&searchOrder),blueprint.get(),observedPrefix->root);
                if(observedPrefix->eligible)
                    std::static_pointer_cast<OwnedReplay>(root.replayContext)->drawCosts=observedPrefix->costs(root.battle->cards,work);
                if(costOpening&&!drawPrefix->prefix.empty()) {
                    spire_draw_costs::Program costs;costs.budget=&work;
                    for(std::size_t i=0;i<drawPrefix->prefix.size();++i) {
                        const auto e=J::parse(prefixEvents.at(i));
                        spire_draw_costs::Evidence item{drawPrefix->prefix[i],e.at("cards").at(0).get<Profile>(),
                            e.at("confused").get<bool>(),e.at("corruption").get<bool>()};
                        spire_draw_costs::rolls(item);costs.events.push_back(item);
                    }
                    std::static_pointer_cast<OwnedReplay>(root.replayContext)->drawCosts=std::move(costs);
                }
                if(shufflePrefix&&!(drawShuffle&&drawShuffle->eligible)&&shufflePrefix->eventConditioned)
                    std::static_pointer_cast<OwnedReplay>(root.replayContext)->eventShuffle.emplace(
                        root.battle->cards,shufflePrefix->discard,shufflePrefix->prefix,plan[5],work);
                if(drawShuffle&&drawShuffle->eligible) {
                    auto owned=std::static_pointer_cast<OwnedReplay>(root.replayContext);
                    if(owned->eventShuffle||owned->drawCosts)throw Failure("overlapping_draw_shuffle_condition");
                    owned->drawCosts.emplace();auto* costs=&*owned->drawCosts;
                    std::vector<Profile> seen;for(const auto& e:drawShuffle->seen)seen.push_back(e.after);
                    owned->eventShuffle.emplace(root.battle->cards,drawShuffle->input,std::move(seen),plan[5],work);
                    owned->eventShuffle->conditionalDraws=[drawShuffle,costs](sts::CardManager& cards,std::mt19937_64& rng,Budget& budget) {
                        return drawShuffle->law->apply(cards,rng,budget,*costs);
                    };
                }
                if(eventOffer){work.tick(eventOffer->poolSize+3);
                    std::static_pointer_cast<OwnedReplay>(root.replayContext)->recorder.eventOffer=eventOffer;}
                return root;},observe,advance,construction,replayLimits,
            [](const BattleContext& b,const J& relics,int top,const std::string&){return policyView(b,relics,top);},offers);
        if(budget.contains("frozen_audit")&&!standalone)throw Failure("frozen_audit_entry_contract");
        J boundaryBudget=budget;
        if(budget.contains("boundary_value")&&!budget.at("max_seconds").is_null())
            boundaryBudget["max_seconds"]=spire_boundary_value::remainingSeconds(budget.at("max_seconds").get<double>(),
                std::chrono::duration<double>(spire_search::Clock::now()-boundaryStarted).count());
        spire_terminal_proof::CacheScope terminalProofCacheScope(terminalProofCache_);
        auto result=continuationAudit?spire_search::auditContinuations(roots,seeds,budget,metadata,*continuationAudit,trace):
            standalone?(rootRecency?spire_search::searchStandaloneTree<false,false,false,false,true>(roots,seeds,budget,metadata,trace):
            deepPrior?spire_search::searchStandaloneTree<false,false,false,true>(roots,seeds,budget,metadata,trace):
            rootObjective?spire_search::searchStandaloneTree<false,false,true>(roots,seeds,budget,metadata,trace):
            scoreOrder?spire_search::searchStandaloneTree<false,true>(roots,seeds,budget,metadata,trace):
            deepQuarter?spire_search::searchStandaloneTree<true>(roots,seeds,budget,metadata,trace):
            roundTree?spire_search::searchRootRounds(roots,seeds,budget,metadata,trace):
            frozenRoot?spire_search::searchFrozenRoot(roots,seeds,budget,metadata,trace):budget.contains("frozen_audit")?
            spire_search::allocateInternalNode(roots,seeds,budget,metadata,trace):
            spire_search::searchStandaloneTree(roots,seeds,boundaryBudget,metadata,trace)):
            pairedAudit?spire_search::auditPairedContinuation(roots,seeds,budget,metadata,*pairedAudit):
            historyTree?spire_search::searchHistoryTree(roots,seeds,budget,metadata,trace,guardedTree,overrideProbe,pairedTree):
            spire_search::searchConditioned(roots,seeds,budget,metadata,trace);
        if(aligned)result["combat_objective"]=spire_objective::name(budget);
        if(repaired)result["session_boundaries"]=boundaryVersion;
        if(historyGuard) {
            result["root_sampling"]["history_work_guard"]=HistoryWorkGuard::version;
            result["root_sampling"]["history_work_bytes"]=historyGuard->bytes;
            result["root_sampling"]["history_work_limit"]=historyGuard->work;
            result["root_sampling"]["history_work_hard_limit"]=HistoryWorkGuard::maximum;
        }
        result["root_sampling"]["offer_seed_attempts"]=offer?offer->attempts:0;
        result["root_sampling"]["offer_seed_cache_hits"]=offer?offer->cacheHits:0;
        result["root_sampling"]["conditioned_draw_prefix"]=drawPrefix->prefix.size();
        result["root_sampling"]["observed_draw_prefix"]={{"version",spire_observed_prefix::version},
            {"eligible",observedPrefix->eligible},{"prefix",observedPrefix->eligible?observedPrefix->seen.size():0},
            {"states",observedPrefix->states}};
        result["root_sampling"]["draw_shuffle_channel"]={{"version",spire_draw_shuffle::version},
            {"eligible",drawShuffle&&drawShuffle->eligible},{"prefix",drawShuffle?drawShuffle->seen.size():0},
            {"states",drawShuffle&&drawShuffle->law?drawShuffle->law->states():0}};
        result["root_sampling"]["observed_draw_costs"]={{"version",spire_draw_costs::version},{"eligible",costOpening},{"prefix",costOpening?drawPrefix->prefix.size():0}};
        if(budget.contains("autoplay_target_prefix"))result["root_sampling"]["autoplay_target_prefix"]=autoplayTargets->diagnostic();
        result["root_sampling"]["mapped_draw_events"]=drawPrefix->mappedEvents;
        result["root_sampling"]["settled_draw_events"]=drawPrefix->settledEvents;
        result["root_sampling"]["settled_draw_prefix"]=SettledDrawPrefixCondition::settlementVersion;
        result["root_sampling"]["mapped_draw_prefix"]=MappedDrawPrefixCondition::version;
        if(eventModel) {
            result["root_model"]=model;
            auto& sampling=result["root_sampling"];
            sampling["probability_model"]=model;
            sampling["replay_attempts_per_world"]=trace::eventReplayAttempts;
            sampling["observed_offer_conditioned"]=program?program->events!=0:bool(eventOffer);
            if(program)sampling["observed_offer_events"]=program->events;
            sampling["observed_offer_pool_size"]=program?program->firstPool:(eventOffer?eventOffer->poolSize:0);
            if(standalone||repaired) {
                // This result has root evidence, NOT rectangular rounds. Resource
                // failures never authorize the legacy event/rule fallback here.
                result["resource_fallback"]=nullptr;
            } else {
            // Typed opt-in resource result. Contracts/contradictions never get
            // this authorization, even if their error text contains "limit".
            std::string code=roots.failureCode();
            if(code.empty()&&result.value("unavailable","").empty()&&result.at("rounds").empty()) {
                const auto stop=result.value("stop","");
                if(stop=="deadline"&&(result.value("refusal",J()).is_null()||result["refusal"]=="tree_deadline"))code="deadline";
                if(stop=="step_budget"||stop=="sample_budget"||stop=="value_budget")code=stop;
            }
            const bool resource=code=="continuation_proposal_limit"||code=="work_limit"||code=="random_limit"||
                code=="deadline"||code=="step_budget"||code=="sample_budget"||code=="value_budget";
            if(resource&&!result.at("rounds").empty())throw Failure("partial_resource_result_contract");
            result["resource_fallback"]=resource?J{{"version","event-resource-fallback-v1"},{"reason",code}}:J(nullptr);
            }
        }
        if(offers)result["offer_replay"]=spire_offer_replay::version;
        if(optimized){result["run_sampling"]=spire_run_sampling::version;
            result["root_sampling"]["shuffle_prefix"]=shufflePrefix->prefix.size();
            result["root_sampling"]["shuffle_seed_attempts"]=shufflePrefix->attempts;
            result["root_sampling"]["shuffle_probe"]=spire_run_sampling::ShufflePrefixProbe::version;
            result["root_sampling"]["turn_shuffle"]=spire_run_sampling::TurnShufflePrefix::version;
            result["root_sampling"]["event_shuffle_conditioned"]=shufflePrefix->eventConditioned;
            result["root_sampling"]["action_shuffle_conditioned"]=shufflePrefix->actionConditioned;
            result["root_sampling"]["action_shuffle_prefix"]=spire_run_sampling::ObservedShufflePrefix::actionVersion;
            result["root_sampling"]["shuffle_probability_model"]=trace::EventShuffle::version;
            result["root_sampling"]["conditioning_evidence_isolation"]="public-shuffle-evidence-isolation-v1";}
        if(trace){result["order_graph"]=spire_run_sampling::graph(drawPrefix->order?*drawPrefix->order:*anchor->order);result["conditioning_trace"]=diagnostic->result();result["conditioning_trace"]["anchor"]=anchor->state;}
        result["session"]=status();return result;
    }
};
} // namespace spire_session
