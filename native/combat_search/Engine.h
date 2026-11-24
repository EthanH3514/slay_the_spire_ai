#pragma once
#include "PublicRules.h"
#include "PotionReplacement.h"
#include "Decisions.h"
#include "JsonOrder.h"
#include "sim/PublicCombatResampling.h"
#include "sim/SimHelpers.h"
#include "sim/PublicBombState.h"
#include <chrono>
#include <memory>

namespace spire_search {
using namespace sts;
using Clock=std::chrono::steady_clock;

inline J publicCard(const CardInstance &c,const BattleContext *b=nullptr) {
    // Direct fields avoid initializer-list JSON arrays for each key/value pair.
    // This is a fresh owned value on every call, not a cross-state card cache.
    J out=J::object();out["id"]=key(getCardEnumName(c.getId()));out["up"]=c.getUpgradeCount();
    out["cost"]=int(c.costForTurn);out["base_cost"]=int(c.cost);out["free"]=c.freeToPlayOnce;
    out["type"]=cardTypeStrings[int(c.getType())];out["damage"]=0;
    out["uid"]=int(c.uniqueId);out["special"]=int(c.specialData);
    if(c.getId()==CardId::RAMPAGE)out["damage"]=8+c.specialData;
    if(c.getId()==CardId::RITUAL_DAGGER)out["damage"]=c.specialData;
    if(b)out["playable"]=c.canUseOnAnyTarget(*b);
    return out;
}
inline std::string relicKey(RelicId id) {
    // Match identifiers.canonical_identifier(..., "relic"), not display labels
    // containing punctuation. These are name aliases, not strategy exceptions.
    auto k=key(getRelicName(id));
    static const std::map<std::string,std::string> aliases={{"boot","theboot"},{"paperfrog","paperphrog"},
        {"toxicegg2","toxicegg"},{"frozenegg2","frozenegg"},{"moltenegg2","moltenegg"},
        {"wingedgreaves","wingboots"},{"sling","slingofcourage"},{"neowsblessing","neowslament"}};
    auto it=aliases.find(k);return it==aliases.end()?k:it->second;
}
// Adapter-private ordering key: exactly the binding's combatCard JSON. The
// public dense selection index must never be the latent draw-pile index.
inline std::string selectionKey(const CardInstance &c,const BattleContext *b=nullptr,bool publicOccurrences=false) {
    J row={{"id",getCardEnumName(c.getId())},{"cost",int(c.costForTurn)},{"base_cost",int(c.cost)},
        {"cost_for_turn",int(c.costForTurn)},{"free_to_play_once",c.freeToPlayOnce},{"upgrades",c.getUpgradeCount()},
        {"type",cardTypeStrings[int(c.getType())]},{"has_target",c.requiresTarget()},{"exhausts",c.doesExhaust()},
        {"ethereal",c.isEthereal()},{"special_data",c.specialData},{"unique_id",publicOccurrences?-1:c.uniqueId}};
    if(c.getId()==CardId::RAMPAGE)row["base_damage"]=8+c.specialData;
    if(c.getId()==CardId::RITUAL_DAGGER)row["base_damage"]=c.specialData;
    if(b)row["is_playable"]=c.canUseOnAnyTarget(*b);
    return row.dump();
}
inline std::vector<std::vector<int>> selectionMap(const BattleContext &b,bool publicOccurrences=false) {
    auto task=b.cardSelectInfo.cardSelectTask;
    std::vector<std::vector<int>> groups;
    if(task==CardSelectTask::GAMBLE)return choiceGroups(b.cards.cardsInHand,b.cards.cardsInHand).get<std::vector<std::vector<int>>>();
    if(task==CardSelectTask::CODEX)return {{0},{1},{2},{3}};
    if(task==CardSelectTask::FORETHOUGHT||task==CardSelectTask::EXHAUST_MANY) {
        bool forethought=task==CardSelectTask::FORETHOUGHT,any=b.cardSelectInfo.canPickAnyNumber;
        auto choices=choiceGroups(b.cards.cardsInHand,forethought?(any?10:1):b.cardSelectInfo.pickCount,forethought,forethought?any:true);
        return choices.get<std::vector<std::vector<int>>>();
    }
    bool canonical=task==CardSelectTask::EXHUME||task==CardSelectTask::SECRET_TECHNIQUE||task==CardSelectTask::SECRET_WEAPON||task==CardSelectTask::DUAL_WIELD||task==CardSelectTask::LIQUID_MEMORIES_POTION;
    std::vector<std::pair<std::string,int>> indices;
    auto add=[&](const CardInstance &c,int idx,bool hand=false) {
        bool allowed=true;
        if(task==CardSelectTask::ARMAMENTS)allowed=c.canUpgrade();
        if(task==CardSelectTask::EXHUME)allowed=c.getId()!=CardId::EXHUME;
        if(task==CardSelectTask::SECRET_TECHNIQUE)allowed=c.getType()==CardType::SKILL;
        if(task==CardSelectTask::SECRET_WEAPON)allowed=c.getType()==CardType::ATTACK;
        if(task==CardSelectTask::DUAL_WIELD)allowed=c.getType()==CardType::ATTACK||c.getType()==CardType::POWER;
        if(allowed)indices.push_back({canonical?selectionKey(c,hand?&b:nullptr,publicOccurrences):"",idx});
    };
    if(task==CardSelectTask::DISCOVERY)for(int i=0;i<3;++i)indices.push_back({"",i});
    else if(task==CardSelectTask::HEADBUTT||task==CardSelectTask::LIQUID_MEMORIES_POTION)for(int i=0;i<int(b.cards.discardPile.size());++i)add(b.cards.discardPile[i],i);
    else if(task==CardSelectTask::EXHUME)for(int i=0;i<int(b.cards.exhaustPile.size());++i)add(b.cards.exhaustPile[i],i);
    else if(task==CardSelectTask::SECRET_TECHNIQUE||task==CardSelectTask::SECRET_WEAPON)
        for(int i=0;i<int(b.cards.drawPile.size());++i)add(b.cards.drawPile[i],i);
    else if(task==CardSelectTask::ARMAMENTS||task==CardSelectTask::DUAL_WIELD||task==CardSelectTask::WARCRY||task==CardSelectTask::EXHAUST_ONE)
        for(int i=0;i<b.cards.cardsInHand;++i)add(b.cards.hand[i],i,true);
    else throw NotPorted("selection window");
    if(canonical)std::stable_sort(indices.begin(),indices.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    if(task==CardSelectTask::LIQUID_MEMORIES_POTION) {
        for(const auto &group:retrievalGroups(int(indices.size()),b.cardSelectInfo.pickCount)) {
            std::vector<int> mapped;
            for(int i:group)mapped.push_back(indices[i].second);
            groups.push_back(mapped);
        }
        return groups;
    }
    for(const auto &x:indices)groups.push_back({x.second});
    return groups;
}
inline J project(const BattleContext &b,const J &relics,int knownTop=-1,const std::string &character="IRONCLAD",bool publicOccurrences=false,
                 bool anonymousCards=false) {
    spire_perf::Scope timing(spire_perf::Phase::Projection);
    // Session projections discard copy IDs. Do that before the one canonical
    // pile sort, instead of sorting with private IDs and sorting again without.
    auto cardView=[&](const CardInstance& c,const BattleContext* context=nullptr){
        auto row=publicCard(c,context);if(anonymousCards)row["uid"]=-1;return row;};
    J powers=J::object(),records=J::array();
    int panache=0;
    for(const auto &x:public_state::playerPowers(b.player)) {
        auto id=key(playerStatusEnumStrings[int(x.type)]);
        powers[id]=x.amount;J row=J::object();row["id"]=id;row["amount"]=x.amount;
        if(x.type==PS::THE_BOMB||x.type==PS::PANACHE)row["damage"]=x.damage;
        if(x.type==PS::PANACHE)panache=x.damage;
        if(x.type==PS::COMBUST)row["hp_loss"]=x.hpLoss;
        records.push_back(std::move(row));
    }
    J out=J::object();auto& player=out["player"];player=J::object();
    player["current_hp"]=b.player.curHp;player["max_hp"]=b.player.maxHp;player["block"]=b.player.block;
    player["energy"]=b.player.energy;player["powers"]=std::move(powers);player["panache_damage"]=panache;
    out["input"]=b.inputState==InputState::PLAYER_NORMAL?"PLAYER_NORMAL":b.inputState==InputState::CARD_SELECT?"CARD_SELECT":b.inputState==InputState::EXECUTING_ACTIONS?"EXECUTING_ACTIONS":"OTHER";
    out["relics"]=relics;out["selection"]=b.inputState==InputState::CARD_SELECT?cardSelectTaskStrings[int(b.cardSelectInfo.cardSelectTask)]:"";
    out["power_records"]=std::move(records);out["character"]=character;out["times_damaged"]=b.player.timesDamagedThisCombat;
    out["bottom_tracked"]=true;out["bomb_schema"]=public_state::bombStateSchema;
    out["powers_complete"]=public_state::powersComplete(b.player);out["resolved"]=public_state::effectsResolved(b);
    out["limbo_empty"]=true;out["selection_candidates"]=J::array();out["selection_options"]=J::array();
    out["selection_any"]=false;out["selection_max"]=0;out["selection_copies"]=0;
    out["selection_pick"]=1;out["selection_type"]="INVALID";
    out["opening_schema"]=1;out["opening_priority_ids"]=b.cards.publicOpeningPriorityIds;
    out["opening_uncertain"]=b.cards.publicOpeningOrderUncertain;
    out["uncertain_order"]=b.cards.publicTopOrderUncertain||b.cards.publicBottomOrderUncertain||b.cards.publicOpeningOrderUncertain;
    out["monsters"]=J::array();out["held"]=J::array();
    out["potions"]=J::array();for(int i=0;i<b.potionCapacity;++i)out["potions"].push_back(key(potionEnumNames[int(b.potions[i])]));
    for(int i=0;i<b.monsters.monsterCount;++i) {
        const auto &m=b.monsters.arr[i];J ps=J::object();
        for(int j=0;j<int(MonsterStatus::INVALID);++j){int n=m.getStatusInternal(MonsterStatus(j));if(n)ps[key(monsterStatusEnumStrings[j])]=n;}
        auto damage=m.getMoveBaseDamage(b);
        J row=J::object();row["current_hp"]=m.curHp;row["max_hp"]=m.maxHp;row["block"]=m.block;
        row["move_base_damage"]=damage.damage;row["move_hits"]=damage.attackCount;
        row["is_gone"]=m.isDeadOrEscaped();row["half_dead"]=m.halfDead;row["powers"]=std::move(ps);
        out["monsters"].push_back(std::move(row));
        if(m.publicGeneration<0)throw NotPorted("invalid public monster generation");
        if(m.id==MonsterId::BRONZE_ORB&&!m.isDeadOrEscaped()&&m.hasStatus<MS::STASIS>())
            out["held"].push_back(cardView(b.cards.stasisCards[std::min(1,i)]));
    }
    if(b.player.hasStatus<PS::SURROUNDED>() && (b.player.lastTargetedMonster<0||b.player.lastTargetedMonster>=b.monsters.monsterCount))
        throw NotPorted("invalid public facing slot");
    out["hand"]=J::array();for(int i=0;i<b.cards.cardsInHand;++i)out["hand"].push_back(cardView(b.cards.hand[i],&b));
    out["draw"]=J::array();for(const auto &c:b.cards.drawPile)out["draw"].push_back(cardView(c));
    // No latent ordering or BattleContext reference leaves this projection.
    sortByJsonEncoding(out["draw"]);
    out["discard"]=J::array();for(const auto &c:b.cards.discardPile)out["discard"].push_back(cardView(c));
    out["exhaust"]=J::array();for(const auto &c:b.cards.exhaustPile)out["exhaust"].push_back(cardView(c));
    J top=J::array();const auto &pile=b.cards.drawPile;
    if(!b.cards.publicKnownTopIds.empty())knownTop=b.cards.publicKnownTopIds.back();
    if(knownTop>=0&&(pile.empty()||pile.back().uniqueId!=knownTop))throw std::runtime_error("stale public top identity");
    if(!pile.empty()&&(knownTop>=0||has(relics,"frozeneye")))top.push_back(cardView(pile.back()));
    else if(!b.cards.publicOpeningPriorityIds.empty()&&!b.cards.publicOpeningOrderUncertain) {
        for(const auto &c:pile)if(std::binary_search(b.cards.publicOpeningPriorityIds.begin(),
            b.cards.publicOpeningPriorityIds.end(),int(c.uniqueId)))top.push_back(cardView(c));
    }
    else if(pile.size()>b.cards.publicKnownBottomIds.size()) {
        for(std::size_t i=b.cards.publicKnownBottomIds.size();i<pile.size();++i)top.push_back(cardView(pile[i]));
    } else if(!pile.empty())top.push_back(cardView(pile.back()));
    else for(const auto &c:b.cards.discardPile)top.push_back(cardView(c));
    sortByJsonEncoding(top);out["top_candidates"]=std::move(top);
    if(b.inputState==InputState::CARD_SELECT) {
        auto task=b.cardSelectInfo.cardSelectTask;auto groups=selectionMap(b,publicOccurrences);
        if(task==CardSelectTask::FORETHOUGHT) {out["selection_any"]=b.cardSelectInfo.canPickAnyNumber;out["selection_options"]=groups;}
        if(task==CardSelectTask::EXHAUST_MANY) {out["selection_max"]=b.cardSelectInfo.pickCount;out["selection_options"]=groups;}
        if(task==CardSelectTask::GAMBLE)out["selection_options"]=groups;
        if(task==CardSelectTask::DUAL_WIELD)out["selection_copies"]=b.cardSelectInfo.data0;
        if(task==CardSelectTask::DISCOVERY)out["selection_type"]=cardTypeStrings[int(b.cardSelectInfo.discoveryType)];
        if(task==CardSelectTask::LIQUID_MEMORIES_POTION) {
            auto sorted=b.cards.discardPile;
            std::stable_sort(sorted.begin(),sorted.end(),[publicOccurrences](const auto&a,const auto&b){return selectionKey(a,nullptr,publicOccurrences)<selectionKey(b,nullptr,publicOccurrences);});
            for(const auto &c:sorted)out["selection_candidates"].push_back(cardView(c));
            out["selection_pick"]=b.cardSelectInfo.pickCount;
            out["selection_options"]=retrievalGroups(int(sorted.size()),b.cardSelectInfo.pickCount);
        }
        if(task==CardSelectTask::DISCOVERY||task==CardSelectTask::CODEX) {
            out["selection_copies"]=b.cardSelectInfo.data0;
            for(auto id:b.cardSelectInfo.cards) {
                CardInstance preview(id);if(id==CardId::BLOOD_FOR_BLOOD)preview.updateCost(-b.player.timesDamagedThisCombat);
                out["selection_candidates"].push_back(cardView(preview));
            }
        }
        for(const auto &g:groups) {
            int i=g.empty()?-1:g[0];
            if(task==CardSelectTask::DUAL_WIELD||task==CardSelectTask::ARMAMENTS)
                out["selection_candidates"].push_back(cardView(b.cards.hand[i],&b));
            if(task==CardSelectTask::EXHUME)out["selection_candidates"].push_back(cardView(b.cards.exhaustPile[i]));
            if(task==CardSelectTask::SECRET_TECHNIQUE||task==CardSelectTask::SECRET_WEAPON)out["selection_candidates"].push_back(cardView(b.cards.drawPile[i]));
        }
    }
    return out;
}
inline int replacementSource(const BattleContext& b,int discard) {
    return spire_potion_replacement::source(b.potions,b.potionCapacity,discard,
        Potion::EMPTY_POTION_SLOT,Potion::ENTROPIC_BREW,b.player.hasRelic<RelicId::SOZU>(),
        [&](int i){return b.canUsePotion(i);});
}
inline J actions(const BattleContext &b,bool publicOccurrences=false,bool policy=true) {
    spire_perf::Scope timing(spire_perf::Phase::LegalActions);
    if(b.inputState==InputState::CARD_SELECT) {
        auto choices=selectionMap(b,publicOccurrences);J out=J::array();
        for(int i=0;i<int(choices.size());++i)out.push_back({{"kind","choose"},{"choice_index",i}});return out;
    }
    if(b.inputState!=InputState::PLAYER_NORMAL)throw NotPorted("selection window");
    J out=J::array();
    const bool constrained=policy&&spire_potion_replacement::enabled;
    if(constrained)++spire_potion_replacement::menus;
    for(int i=0;i<b.potionCapacity;++i) {
        const auto potion=b.potions[i];
        if(potion==Potion::EMPTY_POTION_SLOT||potion==Potion::INVALID)continue;
        if(b.canUsePotion(i)) {
            if(potionRequiresTarget(potion)) {
                for(int j=0;j<b.monsters.monsterCount;++j)if(b.monsters.arr[j].isTargetable())
                    out.push_back({{"kind","use_potion"},{"potion_index",i},{"target_index",j}});
            } else out.push_back({{"kind","use_potion"},{"potion_index",i}});
        }
        if(!constrained||replacementSource(b,i)>=0)out.push_back({{"kind","discard_potion"},{"potion_index",i}});
        else ++spire_potion_replacement::excluded;
    }
    if(b.isCardPlayAllowed())for(int i=0;i<b.cards.cardsInHand;++i) {
        const auto &c=b.cards.hand[i];if(!c.canUseOnAnyTarget(b))continue;
        if(c.requiresTarget()) {for(int j=0;j<b.monsters.monsterCount;++j)if(b.monsters.arr[j].isTargetable())
            out.push_back({{"kind","play_card"},{"card_index",i},{"target_index",j}});}
        else out.push_back({{"kind","play_card"},{"card_index",i}});
    }
    out.push_back({{"kind","end_turn"}});return out;
}
inline void advance(BattleContext &b,const J &a,int &knownTop,bool publicOccurrences=false) {
    spire_perf::Scope timing(spire_perf::Phase::Simulation);
    const auto legal=actions(b,publicOccurrences,false);
    if(std::find(legal.begin(),legal.end(),a)==legal.end())throw std::runtime_error("native rollout returned an illegal action");
    b.publicMovedDrawTopUniqueId=-1;
    if(a.at("kind")=="choose") {
        auto group=selectionMap(b,publicOccurrences).at(a.at("choice_index").get<int>());int idx=group.empty()?-1:group[0];
        fixed_list<int,10> indices;for(int i:group)indices.push_back(i);
        switch(b.cardSelectInfo.cardSelectTask) {
            case CardSelectTask::EXHAUST_ONE:b.chooseExhaustOneCard(idx);break;
            case CardSelectTask::ARMAMENTS:b.chooseArmamentsCard(idx);break;
            case CardSelectTask::HEADBUTT:knownTop=b.cards.discardPile[idx].uniqueId;b.chooseHeadbuttCard(idx);break;
            case CardSelectTask::WARCRY:knownTop=b.cards.hand[idx].uniqueId;b.chooseWarcryCard(idx);break;
            case CardSelectTask::FORETHOUGHT:b.chooseForethoughtCards(indices);break;
            case CardSelectTask::EXHAUST_MANY:b.chooseExhaustCards(indices);break;
            case CardSelectTask::EXHUME:b.chooseExhumeCard(idx);break;
            case CardSelectTask::SECRET_TECHNIQUE:case CardSelectTask::SECRET_WEAPON:b.chooseDrawToHandCards(&idx,1);break;
            case CardSelectTask::DUAL_WIELD:b.chooseDualWieldCard(idx);break;
            case CardSelectTask::DISCOVERY:b.chooseDiscoveryCard(b.cardSelectInfo.cards[idx]);break;
            case CardSelectTask::GAMBLE:b.chooseGambleCards(indices);break;
            case CardSelectTask::CODEX:if(idx!=3)b.chooseCodexCard(b.cardSelectInfo.cards[idx]);break;
            case CardSelectTask::LIQUID_MEMORIES_POTION:b.chooseDiscardToHandCards(group,true);break;
            default:throw NotPorted("selection advance");
        }
    } else if(a.at("kind")=="use_potion") {b.drinkPotion(number(a,"potion_index"),number(a,"target_index",0));knownTop=-1;}
    else if(a.at("kind")=="discard_potion")b.discardPotion(number(a,"potion_index"));
    else if(a.at("kind")=="end_turn") {if(b.player.hasStatus<PS::MAYHEM>())knownTop=-1;b.endTurn();}
    else {
        const auto &card=b.cards.hand[a.at("card_index").get<int>()];
        if(card.getId()==CardId::HAVOC||(card.getId()==CardId::DEEP_BREATH&&!b.cards.discardPile.empty()))knownTop=-1;
        if(card.getId()==CardId::HEADBUTT&&b.cards.discardPile.size()==1)knownTop=b.cards.discardPile[0].uniqueId;
        b.addToBotCard(CardQueueItem(card,a.value("target_index",0),b.player.energy));
    }
    b.inputState=InputState::EXECUTING_ACTIONS;b.executeActions();
    if(!b.cards.publicKnownTopIds.empty())knownTop=-1;
    else if(b.publicMovedDrawTopUniqueId>=0)knownTop=b.publicMovedDrawTopUniqueId;
    if(knownTop>=0&&(b.cards.drawPile.empty()||b.cards.drawPile.back().uniqueId!=knownTop))knownTop=-1;
}
inline int outcome(const BattleContext &b) {
    if(b.outcome==Outcome::PLAYER_VICTORY)return 1;
    if(b.outcome==Outcome::PLAYER_LOSS)return 2;
    if(b.outcome==Outcome::PLAYER_ESCAPED)return 3;
    return 0;
}

struct SearchRoot {
    std::unique_ptr<BattleContext> battle;
    int knownTop = -1;
    // Optional adapter-owned replay observer. Its lifetime follows this world,
    // never the live session; legacy callers leave it empty.
    std::shared_ptr<void> replayContext;
};

// The flat evaluator is shared by the legacy sampler and versioned conditional
// roots. A provider prepares ONE latent plan per round, then constructs an
// independently owned continuation for every candidate. In particular, it must
// not clone opaque paused queues whose closures can capture another context.
template<class RootProvider>
inline J flatSearch(RootProvider &roots,const J &original,const J &relics,const std::string &character,
    const std::vector<std::array<std::uint64_t,7>> &seeds,const J &candidates,
    const J &budget,const J &meta,bool trace=false,Clock::time_point started=Clock::now(),bool publicOccurrences=false) {
    double seconds=budget.at("max_seconds").is_null()?-1:budget.at("max_seconds").get<double>();
    auto expired=[&](){return seconds>=0 && std::chrono::duration<double>(Clock::now()-started).count()>=seconds;};
    int samples=0,steps=0,values=0,policies=0,discarded=0,peak=0;
    J rounds=J::array(),horizons=J::array(),traces=J::array();
    auto finish=[&](const std::string &stop,const std::string &unavailable="") {
        if(roots.atomic()&&(stop=="deadline"||stop=="unsupported"||stop=="native_unavailable")) {
            discarded=samples;rounds=J::array();horizons=J::array();traces=J::array();
        }
        return J{{"stop",stop},{"unavailable",unavailable},{"samples",samples},{"steps",steps},
            {"values",values},{"policies",policies},{"discarded",discarded},{"peak",peak},
            {"rounds",rounds},{"horizons",horizons},{"traces",traces}};
    };
    try {
        roots.validate(original,candidates,meta);
        const int count=int(candidates.size());
        if(count<2)throw std::invalid_argument("native search requires at least two candidates");
        int seedCount=std::min({number(budget,"samples_per_action"),number(budget,"max_samples")/count,
            number(budget,"max_value_evaluations")/count,number(budget,"max_branch_steps")/count});
        if(seeds.size()!=std::size_t(seedCount))throw std::invalid_argument("wrong independent seed plan size");
        for(int round=0;round<number(budget,"samples_per_action");++round) {
            if(expired())return finish("deadline");
            if(number(budget,"max_samples")-samples<count)return finish("sample_budget");
            if(number(budget,"max_value_evaluations")-values<count)return finish("value_budget");
            int horizon=std::min(number(budget,"rollout_horizon"),(number(budget,"max_branch_steps")-steps)/count);
            if(horizon<1)return finish("step_budget");
            const auto &seed=seeds.at(round); // A budget stop precedes exhaustion of the seed plan.
            J rows=J::array();
            try {
                auto until=seconds<0?Clock::time_point::max():started+
                    std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
                roots.prepare(round,seed,until);
            } catch(const std::runtime_error &e) {
                if(roots.atomic()) {discarded=samples;rounds=J::array();horizons=J::array();traces=J::array();}
                auto rejected=finish("unsupported");
                rejected["refusal"]=std::string("native combat sampling rejected root: ")+e.what();
                return rejected;
            }
            for(const auto &candidate:candidates) {
                if(expired()){
                    discarded+=int(rows.size());
                    if(roots.atomic()) {discarded=samples;rounds=J::array();horizons=J::array();traces=J::array();}
                    return finish("deadline");
                }
                ++samples;
                SearchRoot root;
                try {
                    root=roots.branch(round);
                    if(!root.battle)throw std::runtime_error("empty sampled root");
                }catch(const std::runtime_error &e) {
                    discarded+=int(rows.size());
                    if(roots.atomic()) {discarded=samples;rounds=J::array();horizons=J::array();traces=J::array();}
                    auto rejected=finish("unsupported");
                    rejected["refusal"]=std::string("native combat sampling rejected root: ")+e.what();
                    return rejected;
                }
                auto branch=std::move(root.battle);
                peak=1;
                int branchTop=root.knownTop;
                J view=roots.projectView(*branch,relics,branchTop,character);
                if(view!=original || !roots.sameActions(actions(*branch,publicOccurrences),candidates))throw std::runtime_error("native resampling changed public information");
                J action=candidate,traceSteps=J::array();
                for(int depth=0;depth<horizon;++depth) {
                    if(expired()){discarded+=int(rows.size())+1;return finish("deadline");}
                    if(trace)traceSteps.push_back({{"public",view},{"action",action}});
                    ++steps;advance(*branch,action,branchTop,publicOccurrences);view=roots.projectView(*branch,relics,branchTop,character);
                    if(outcome(*branch))break;
                    try {
                        guard(view,meta); // Validate public rules, not a new resampling root.
                    } catch(const NotPorted &e) {
                        discarded+=int(rows.size())+1;
                        const bool scopeGap=dynamic_cast<const ScopeGap *>(&e)!=nullptr;
                        if(scopeGap) {discarded=samples;rounds=J::array();horizons=J::array();}
                        auto rejected=finish("native_unavailable",e.what());
                        rejected["scope_gap"]=scopeGap;
                        // Only the owned PUBLIC projection, never the sampled RNG,
                        // queue internals, latent pile order or native branch object.
                        rejected["public_failure"]={{"phase","rollout_guard"},{"round",round},
                            {"candidate_index",int(rows.size())},{"depth",depth+1},{"last_action",action},{"public",view}};
                        return rejected;
                    }
                    if(depth+1==horizon)break;
                    ++policies;action=combatChoice(view,actions(*branch,publicOccurrences),meta);
                }
                if(expired()){discarded+=int(rows.size())+1;return finish("deadline");}
                ++values;int result=outcome(*branch);double value=combatValue(view,result);
                if(!std::isfinite(value))throw std::runtime_error("native non-finite combat value");
                if(expired()){discarded+=int(rows.size())+1;return finish("deadline");}
                rows.push_back(spire_objective::record(view,original,result,value,spire_objective::name(budget)));
                if(trace)traces.push_back({{"steps",traceSteps},{"final",view},{"outcome",result}});
            }
            rounds.push_back(rows);horizons.push_back(horizon);
        }
        return finish("requested_samples");
    }catch(const ScopeGap &e){auto rejected=finish("native_unavailable",e.what());rejected["scope_gap"]=true;return rejected;}
    catch(const NotPorted &e){return finish("native_unavailable",e.what());}
}

inline J search(const GameContext &game,const BattleContext &source,const public_sampling::PublicEnemyHistory &history,
    const std::vector<std::array<std::uint64_t,7>> &seeds,int knownTop,const J &expected,const J &candidates,
    const J &budget,const J &meta,bool trace=false) {
    auto started=Clock::now();
    struct LegacyRoots {
        const GameContext &game; const BattleContext &source;
        const public_sampling::PublicEnemyHistory &history;
        int knownTop;
        std::array<std::uint64_t,7> seed{};
        void validate(const J &original,const J &candidates,const J &meta) const {
            guardResamplingRoot(original,meta);
            if(actions(source)!=candidates)throw std::runtime_error("native legal ordering differs from policy input");
        }
        void prepare(int,const std::array<std::uint64_t,7> &value,Clock::time_point) {seed=value;}
        SearchRoot branch(int) const {
            auto copy=std::make_unique<BattleContext>(source);
            public_sampling::resampleCombatContinuation(game,*copy,seed,knownTop,&history);
            return {std::move(copy),knownTop};
        }
        bool atomic() const {return false;}
        J projectView(const BattleContext& b,const J& relics,int top,const std::string& character) const {
            return project(b,relics,top,character);
        }
        bool sameActions(const J& actual,const J& expected) const {return actual==expected;}
    } roots{game,source,history,knownTop};
    J relics=J::array();for(const auto &r:game.relics.relics)relics.push_back(relicKey(r.id));
    std::sort(relics.begin(),relics.end());
    const auto character=characterClassEnumNames[int(game.cc)];
    // Keep legacy validation/error semantics and the public-only rule path.
    try {
        auto original=project(source,relics,knownTop,character);
        if(original!=expected)throw std::runtime_error("native public projection differs from fresh policy input");
        return flatSearch(roots,original,relics,character,seeds,candidates,budget,meta,trace,started);
    } catch(const NotPorted &e) {
        return J{{"stop","native_unavailable"},{"unavailable",e.what()},{"samples",0},{"steps",0},
            {"values",0},{"policies",0},{"discarded",0},{"peak",0},{"rounds",J::array()},
            {"horizons",J::array()},{"traces",J::array()}};
    }
}
} // namespace spire_search
