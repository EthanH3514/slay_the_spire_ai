#pragma once
// All public combat decisions, including the small fast-mode potion controller.
// No BattleContext, RNG, hidden world or Python policy callback is available here.
#include "PublicRules.h"
#include "CombatObjective.h"
#include <array>
#include <limits>

namespace spire_search {
inline J nonPotionActions(const J &actions) {
    J cards=J::array();
    for(const auto &a:actions)if(a.at("kind")!="use_potion"&&a.at("kind")!="discard_potion")cards.push_back(a);
    return cards;
}

inline J ruleDecision(const J &view,const J &actions,const J &metadata,std::vector<double>* captured=nullptr) {
    auto cards=nonPotionActions(actions);
    if(cards.empty())throw std::invalid_argument("no non-potion legal actions");
    J gaps;
    guard(view,metadata,&gaps);
    if(view.at("input")=="CARD_SELECT") {
        auto task=view.value("selection","");
        const char *field=(task=="FORETHOUGHT"||task=="EXHAUST_MANY"||task=="GAMBLE"||task=="LIQUID_MEMORIES_POTION")?"selection_options":
            (task=="HEADBUTT"?"discard":((task=="EXHAUST_ONE"||task=="WARCRY")?"hand":"selection_candidates"));
        for(const auto &a:cards)if(a.at("kind")=="choose") {
            int i=number(a,"choice_index",-1);
            if(task=="CODEX"&&i==3)continue;
            if(i<0||i>=int(view.at(field).size()))throw std::invalid_argument("selection references invalid public candidate index");
        }
    }
    return J{{"action",Rules(view,metadata).choose(cards,captured)},{"scope_gaps",gaps}};
}
inline J ruleChoice(const J &view,const J &actions,const J &metadata) {
    return ruleDecision(view,actions,metadata).at("action");
}

// Existing JSON request API remains available at the binding boundary. Internal
// rollout calls borrow immutable metadata for this call only, never cache it.
inline J potionChoice(const J &view,const J &actions,const J &request,const J* sharedMetadata=nullptr,
                      std::vector<double>* captured=nullptr) {
    const auto &base=request.at("base");
    if(std::find(actions.begin(),actions.end(),base)==actions.end())
        throw std::invalid_argument("base choice is not a legal action");
    if(captured)captured->assign(actions.size(),std::numeric_limits<double>::quiet_NaN());
    if(view.at("input")!="PLAYER_NORMAL")return base;
    const auto &metadata=sharedMetadata?*sharedMetadata:request.at("metadata");
    Rules rules(view,metadata);
    double best=0.;J chosen=base;
    for(std::size_t i=0;i<actions.size();++i){const auto& a=actions[i];
      if(captured&&a.at("kind")=="discard_potion")(*captured)[i]=rules.potionScore(a);
      if(a.at("kind")=="use_potion") {
        const int slot=number(a,"potion_index",-1);
        if(slot<0||slot>=int(view.at("potions").size()))throw std::invalid_argument("invalid potion slot");
        // Fast baseline reports unknown identities as gaps and leaves them alone.
        // Search admission separately refuses them; never guess an effect.
        if(!metadata.at("$potions").contains(view.at("potions")[slot].get<std::string>()))continue;
        const double score=rules.potionScore(a);
        if(captured)(*captured)[i]=score;
        if(score>best){best=score;chosen=a;}
    }}
    return chosen;
}
inline J combatChoice(const J &view,const J &actions,const J &metadata) {
    spire_perf::Scope timing(spire_perf::Phase::Rule);
    const auto base=ruleChoice(view,actions,metadata);
    return potionChoice(view,actions,J{{"base",base}},&metadata);
}

struct ScoredCombatChoice {J prior;std::vector<double> scores;unsigned extraDiscardScores=0;};
inline ScoredCombatChoice scoredCombatChoice(const J& view,const J& actions,const J& metadata){
    spire_perf::Scope timing(spire_perf::Phase::Rule);
    std::vector<double> cards;const auto base=ruleDecision(view,actions,metadata,&cards).at("action");
    ScoredCombatChoice result;
    result.prior=potionChoice(view,actions,J{{"base",base}},&metadata,&result.scores);
    std::size_t card=0;
    for(std::size_t i=0;i<actions.size();++i){const auto& kind=actions[i].at("kind");
        if(kind!="use_potion"&&kind!="discard_potion")result.scores[i]=cards.at(card++);
        result.extraDiscardScores+=kind=="discard_potion";
        // Existing Rules uses -infinity for a certainly fatal on-play trigger.
        // Keep it as a last-ranked legal action, not a coverage refusal.
        if(std::isnan(result.scores[i])||result.scores[i]==std::numeric_limits<double>::infinity())
            throw std::invalid_argument("score_order_invalid_or_missing_score");
    }
    if(card!=cards.size())throw std::logic_error("score_order_card_alignment");
    return result;
}

inline J rankedChoice(const J &view,const J &metadata,const J &estimates) {
    if(estimates.empty())throw std::invalid_argument("no complete candidate estimates");
    auto risk=[](const J &e) {return spire_objective::fromEstimate(e);};
    auto best=risk(estimates[0]);J tied=J::array();
    for(const auto &e:estimates) {
        auto value=risk(e);
        for(double x:value)if(!std::isfinite(x))throw std::invalid_argument("non-finite risk estimate");
        if(value>best){best=value;tied=J::array();}
        if(value==best)tied.push_back(e.at("action"));
    }
    J first=tied[0],chosen=first;
    if(tied.size()>1) {
        std::stable_sort(tied.begin(),tied.end(),[](const J&a,const J&b){return a.dump()<b.dump();});
        auto cards=nonPotionActions(tied);
        chosen=cards.empty()?tied[0]:ruleChoice(view,cards,metadata);
    }
    return J{{"action",chosen},{"top_tie_count",tied.size()},
             {"tie_breaker",tied.size()>1?J("public_rule_canonical_actions"):J(nullptr)},
             {"tie_changed_action",chosen!=first}};
}
} // namespace spire_search
