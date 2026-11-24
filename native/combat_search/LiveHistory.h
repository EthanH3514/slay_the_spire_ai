#pragma once
#include "PublicSessionState.h"
#include "LiveCombat.h"
#include "../draw_belief/SimulatorMechanisms.h"
#include "../draw_belief/Selection.h"

namespace spire_session {
// Continuous Java semantic consumer. Input consists ONLY of the versioned
// public journal. Transport acknowledgement follows a successful semantic
// commit, never a successful read/JSON parse alone.
inline constexpr auto liveHistoryVersion="public-live-history-v1";
class LiveHistory {
    using Record=spire_belief::trace::MechanismRecord;
    using Mechanism=spire_belief::Mechanism;
    CardManager dummy_; // No game, private order or RNG exists in this adapter.
    spire_belief::trace::MechanismRecorder recorder_{dummy_,false};
    std::unique_ptr<spire_belief::trace::MechanismReplay> replay_;
    spire_belief::SelectionWindow window_;
    std::map<std::string,CardId> dictionary_;
    std::string session_,failure_;
    std::uint64_t epoch_=0,next_=0;
    bool ended_=false;
    bool viewClosed_=false;
    J boundary_,view_,retrieved_=J::array();
    J closedIndices_=J::array();
    LiveCombat combat_;
    Profile profile(const J& c) const {
        auto entry=dictionary_.find(c.at("id").get<std::string>());
        if(entry==dictionary_.end())throw std::runtime_error("unmapped_public_card");
        auto id=entry->second;
        if(c.at("type")!=cardTypeStrings[int(getCardType(id))])throw std::runtime_error("public_card_type_mismatch");
        Profile out={int(id),checked<int>(c.at("upgrades")),checked<int>(c.at("base_cost")),
            checked<int>(c.at("cost_for_turn")),checked<bool>(c.at("free_to_play_once"))?1:0,c.value("base_damage",0)};
        (void)spire_belief::trace::mappedProfile(out,spire_belief::trace::AttributeRule::ResetTurnCosts);return out;
    }
    spire_belief::trace::Inventory inventory(const J& cards) const {
        if(!cards.is_array()||cards.size()>4096)throw std::runtime_error("public_inventory_limit");
        spire_belief::trace::Inventory out;for(const auto& c:cards)out.push_back(profile(c));return out;
    }
    void drain() {
        replay_->drain(recorder_,{},false);
        recorder_.acknowledge(replay_->nextSequence());
    }
    void emit(Record r) {recorder_.emit(std::move(r));drain();}
    bool power(const J& side,const std::string& id) const {
        for(const auto& p:side.at("powers"))if(p.at("id")==id)return true;return false;
    }
    void settleView() {
        if(!viewClosed_)return;
        if(window_.open()) {
            if(retrieved_.empty())return; // UI closes before queued retrieval.
            if(retrieved_.size()!=1||closedIndices_.size()!=1)throw spire_belief::Failure("draw_view_retrieval_contract");
            auto& b=replay_->belief;b.engine.budget.begin();const auto& r=retrieved_[0];
            auto idx=r.at("index").get<unsigned>();
            if(closedIndices_[0]!=idx||profile(view_.at("cards").at(idx))!=profile(r.at("card")))
                throw spire_belief::Failure("selected_profile_mismatch");
            window_.select(view_.at("view"),idx);
            window_.confirm(b,view_.at("view"),{r.at("handle").get<spire_belief::Handle>()});
        }
        view_=J();retrieved_=J::array();closedIndices_=J::array();viewClosed_=false;
    }
    void apply(const J& event) {
        using namespace spire_belief;using namespace spire_belief::trace;
        auto kind=event.at("kind").get<std::string>();const auto& d=event.at("data");
        if(kind=="anchor") {
            if(replay_||next_)throw Failure("duplicate_anchor");
            recorder_.anchor(inventory(d.at("priority")),inventory(d.at("other")));
            replay_=std::make_unique<MechanismReplay>(recorder_);return;
        }
        if(!replay_)throw Failure("missing_anchor");
        if(kind=="checkpoint") {
            const auto& state=d.at("state");auto cards=inventory(state.at("draw_inventory"));
            replay_->drain(recorder_,cards);
            if(state.contains("revealed_order")){Record r{Mechanism::Reveal};r.cards=inventory(state["revealed_order"]);emit(std::move(r));}
            replay_=replay_->compact();boundary_=state;return;
        }
        if(kind=="selection_open") {
            if(!view_.is_null())throw Failure("nested_public_view");view_=d;retrieved_=J::array();viewClosed_=false;
            auto contract=d.at("contract").get<std::string>();
            if(contract=="draw_stable"||contract=="draw_randomized") {
                auto& b=replay_->belief;b.engine.budget.begin();ViewSpec spec;spec.id=d.at("view");
                spec.source=contract=="draw_stable"?ViewSource::DrawStable:ViewSource::DrawRandomized;
                spec.minimum=spec.maximum=1;spec.selectionOrderMatters=true;spec.bottomFirst=d.at("bottom_first");
                if(d.at("count")!=1||d.at("any_number")!=false||d.at("up_to")!=false)
                    throw Failure("draw_view_arity_contract");
                if(d.at("cards").size()!=d.at("tie_groups").size())throw Failure("selector_contract");
                for(std::size_t i=0;i<d.at("cards").size();++i){auto label=replay_->publicLabel(profile(d["cards"][i]));
                    spec.cards.push_back({label,0,d["tie_groups"][i].get<unsigned>()});spec.eligible.push_back(label);}
                std::sort(spec.eligible.begin(),spec.eligible.end());spec.eligible.erase(std::unique(spec.eligible.begin(),spec.eligible.end()),spec.eligible.end());
                window_.begin(b,spec);
            } else if(contract!="visible")throw Failure("unknown_view_contract");
            return;
        }
        if(kind=="selection_closed") {
            if(view_.is_null()||view_.at("view")!=d.at("view"))throw Failure("stale_selection_view");
            if(viewClosed_)throw Failure("duplicate_selection_close");
            viewClosed_=true;closedIndices_=d.at("indices");settleView();return;
        }
        if(kind=="selection_choice"||kind=="selection_state") {
            if(view_.is_null()||view_.at("view")!=d.at("view"))throw Failure("stale_selection_view");return;
        }
        if(kind=="command")return; // Preserved by the transport/continuation journal.
        if(kind=="card_use") {
            combat_.cardUse(card(J(profile(d.at("card"))),0),checked<int>(d.at("target")));return;
        }
        if(kind=="battle_end"){ended_=true;return;}
        Record r{Mechanism::Expose};
        if(kind=="expose"||kind=="top"||kind=="bottom"||kind=="insert_below_top") {
            r.kind=kind=="expose"?Mechanism::Expose:kind=="top"?Mechanism::Top:kind=="bottom"?Mechanism::Bottom:Mechanism::Insert;
            r.cards={profile(d.at("card"))};r.handle=d.at("handle");
        } else if(kind=="alias"){r.kind=Mechanism::Alias;r.handle=d.at("handle");r.sources={d.at("source").get<Handle>()};}
        else if(kind=="track"||kind=="forget"||kind=="release") {
            r.kind=kind=="track"?Mechanism::Track:kind=="forget"?Mechanism::Forget:Mechanism::Release;r.handle=d.at("handle");
        } else if(kind=="modify_damage"){r.kind=Mechanism::Modify;r.handle=d.at("handle");r.copy=CopyRule::AddDamage;r.amount=d.at("amount");}
        else if(kind=="draw_after_callbacks"||kind=="autoplay_top") {
            r.kind=Mechanism::Draw;r.cards={profile(d.at("card"))};r.handle=d.at("handle");
            if(kind=="draw_after_callbacks"){r.confused=power(d.at("side_state"),"Confusion");r.corruption=power(d.at("side_state"),"Corruption");}
        } else if(kind=="shuffle")r.kind=Mechanism::Shuffle;
        else if(kind=="import_shuffled") {
            r.kind=Mechanism::Import;for(const auto& c:d.at("cards")){r.cards.push_back(profile(c.at("card")));r.sources.push_back(c.at("handle"));}
        } else if(kind=="attribute_map") {
            r.kind=Mechanism::Map;auto rule=d.at("rule").get<std::string>();
            if(rule=="reset_turn_costs")r.attribute=AttributeRule::ResetTurnCosts;
            else if(rule=="upgrade_eligible")r.attribute=AttributeRule::UpgradeEligible;
            else if(rule=="corruption_costs")r.attribute=AttributeRule::CorruptionCosts;
            else if(rule=="damage_costs")r.attribute=AttributeRule::DamageCosts;
            else throw Failure("unknown_attribute_rule");
        } else if(kind=="retrieve") {
            auto selector=d.at("selector").get<std::string>();
            if(selector=="stable_view"||selector=="randomized_view") {
                if(!window_.open()||view_.at("view")!=d.at("view"))throw Failure("missing_public_view");
                retrieved_.push_back(d);settleView();return;
            }
            r.kind=Mechanism::RandomRemove;r.cards={profile(d.at("card"))};r.handle=d.at("handle");
            if(selector=="highest_rarity")r.selector=Selector::HighestRarity;
            else if(selector=="uniform_eligible")r.selector=Selector::Attack;
            else if(selector=="single_eligible") {
                // Original attack/skill selectors auto-pick only one eligible
                // card. Condition on that PUBLIC arity, never an occurrence ID.
                auto& b=replay_->belief;b.engine.budget.begin();
                Cards eligible(&b.engine.arena),shown(&b.engine.arena);
                auto label=replay_->publicLabel(r.cards[0]);shown.push_back(label);
                const auto type=getCardType(static_cast<CardId>(r.cards[0][0]));
                for(auto p:b.profiles())if(getCardType(static_cast<CardId>(replay_->publicCard(p)[0]))==type)eligible.push_back(p);
                b.reconcileEligible(eligible,shown);
                r.kind=Mechanism::RankedRemove;r.selector=Selector::PublicBottomRank;r.rank=0;
            } else throw Failure("unknown_retrieval_contract");
        } else throw Failure("unknown_public_event");
        emit(std::move(r));
    }
public:
    // These methods are deliberately separate from order-only semantic ACK.
    // Committing a journal is not a claim that its whole combat is searchable.
    J probeResolved(const J& expected) {
        if(!failure_.empty()||ended_||expected!=boundary_)throw std::runtime_error("stale_live_root_request");
        if(!replay_||!view_.is_null()||replay_->belief.hasActiveReferences()||replay_->belief.pendingRandomizedView())
            throw std::runtime_error("unresolved_live_order_anchor");
        combat_.observe(boundary_,[this](const J& c){return profile(c);});
        auto b=resolved(combat_.state());
        return J{{"version",liveRootVersion},{"state",combat_.state()},
            {"public",policyView(*b,combat_.relics())},{"actions",spire_search::actions(*b,true)}};
    }
    std::string probe(const std::string& expected) {return probeResolved(J::parse(expected)).dump();}
private:
    auto rootFactory() {
        std::shared_ptr<spire_belief::trace::MechanismReplay> order=replay_->compact();
        auto state=combat_.state();auto enemies=combat_.enemies();
        return [order,state,enemies](const spire_search::RootSeeds& plan,spire_belief::Budget& work) {
            work.tick(state.dump().size()+1);auto b=resolved(state);enemies.reconstruct(*b,plan);
            auto& belief=order->belief;belief.engine.budget.begin(work.deadline);
            auto expression=belief.publicRoot();std::mt19937_64 random(plan[6]);
            auto sample=belief.engine.sample(expression,random);
            if(belief.engine.budget.randomWords>work.limits.randomWords-work.randomWords)throw spire_belief::Failure("random_limit");
            work.randomWords+=belief.engine.budget.randomWords;work.tick(belief.engine.budget.work);
            std::map<Profile,std::vector<CardInstance>> inventory;
            for(const auto& c:b->cards.drawPile)inventory[spire_belief::trace::project(c)].push_back(c);
            b->cards.drawPile.clear();
            for(auto it=sample.rbegin();it!=sample.rend();++it) {
                auto& bucket=inventory[order->card(*it)];if(bucket.empty())throw spire_belief::Failure("live_sampled_inventory");
                b->cards.drawPile.push_back(bucket.back());bucket.pop_back();
            }
            for(const auto& row:inventory)if(!row.second.empty())throw spire_belief::Failure("live_sampled_inventory");
            if(snapshot(*b)!=state)throw spire_belief::Failure("live_sample_projection_mismatch");
            return spire_search::SearchRoot{std::move(b),-1};
        };
    }
public:
    // Diagnostic only: advance one independently sampled public root. Never
    // installs a branch as the real state or reads the real game's RNG.
    std::string previewResolved(const std::string& expected,const std::string& action,
                                const spire_search::RootSeeds& seeds) {
        auto current=probeResolved(J::parse(expected));auto choice=J::parse(action);
        if(std::find(current["actions"].begin(),current["actions"].end(),choice)==current["actions"].end())
            throw std::runtime_error("illegal_live_preview_action");
        spire_belief::Limits limits;limits.work=8000000;limits.randomWords=1000000;
        spire_belief::Budget work(limits);work.begin(spire_search::Clock::now()+std::chrono::seconds(2));
        auto root=rootFactory()(seeds,work);
        spire_search::advance(*root.battle,choice,root.knownTop,true);
        return J{{"state",snapshot(*root.battle)},{"public",policyView(*root.battle,combat_.relics(),root.knownTop)}}.dump();
    }
    std::string searchResolved(const std::string& expected,const std::vector<spire_search::RootSeeds>& seeds,
                               const std::string& encodedBudget,const std::string& encodedMetadata,bool trace) {
        const auto request=J::parse(expected);auto current=probeResolved(request);
        auto state=combat_.state(),relics=combat_.relics();auto build=rootFactory();
        auto order=replay_->compact();
        const auto bytes=order->belief.certificate();std::string graph(bytes.begin(),bytes.end());
        spire_search::RootEvidence evidence{graph,liveRootVersion,stateVersion,"native-public-occurrences-v1",
            {state.dump()}, {},current["public"],current["actions"],relics,"IRONCLAD"};
        spire_belief::Limits work;work.work=8000000;work.randomWords=1000000;
        spire_search::ConditionalRoots roots(std::move(evidence),build,
            [](const spire_search::SearchRoot& root){return std::vector<std::string>{snapshot(*root.battle).dump()};},
            [](spire_search::SearchRoot&,const std::string&)->std::vector<std::string>{throw spire_belief::Failure("live_paused_replay_pending");},
            work,{},[](const BattleContext& b,const J& rs,int top,const std::string&){return policyView(b,rs,top);});
        auto result=spire_search::searchConditioned(roots,seeds,J::parse(encodedBudget),J::parse(encodedMetadata),trace);
        result["live_contract"]=liveRootVersion;return result.dump();
    }
    explicit LiveHistory(const std::string& dictionary) {
        const auto entries=J::parse(dictionary);
        if(!entries.is_object()||entries.empty()||entries.size()>4096)throw std::runtime_error("invalid_public_card_dictionary");
        std::map<std::string,CardId> ids;
        for(int id=0;id<=int(CardId::ZAP);++id)ids.emplace(getCardEnumName(static_cast<CardId>(id)),static_cast<CardId>(id));
        for(auto it=entries.begin();it!=entries.end();++it) {
            auto id=ids.find(it.value().get<std::string>());if(id==ids.end())throw std::runtime_error("invalid_public_card_dictionary");
            dictionary_.emplace(it.key(),id->second);
        }
    }
    std::string commit(const std::string& encoded) {
        if(!failure_.empty())throw std::runtime_error("stale_live_history:"+failure_);
        try {
            if(encoded.size()>4*1024*1024)throw std::runtime_error("journal_byte_limit");
            auto batch=J::parse(encoded);auto session=batch.at("session").get<std::string>();auto epoch=checked<std::int64_t>(batch.at("epoch"));
            if(session.empty()||session.size()>128||epoch<=0||!batch.at("records").is_array()||batch.at("records").size()>8192)
                throw std::runtime_error("public_journal_contract");
            if(!epoch_){session_=session;epoch_=epoch;}
            if(epoch_!=epoch||session_!=session)throw std::runtime_error("unexpected_battle_epoch");
            for(const auto& r:batch.at("records")) {
                if(ended_||r.at("epoch")!=epoch_||r.at("sequence")!=next_)throw std::runtime_error("event_gap");
                apply(r);++next_;
            }
            if(next_!=batch.at("next_sequence"))throw std::runtime_error("event_gap");
            return J{{"version",liveHistoryVersion},{"session",session_},{"epoch",epoch_},
                {"next_sequence",next_},{"ended",ended_},{"checkpoint",boundary_},
                {"open_view",viewClosed_?J():view_},{"order_committed",bool(replay_)}}.dump();
        }catch(const std::exception& e){failure_=e.what();throw;}
    }
};
} // namespace spire_session
