#pragma once
#include "draw_belief/SimulatorMechanisms.h"

namespace spire_draw_costs {
using namespace spire_belief;
using Profile=trace::Card;
inline constexpr auto version="public-observed-draw-cost-event-v1";
struct Evidence { Profile before,after; bool confused=false,corruption=false; };

// A prefix with a unique pre-draw public profile has no ambiguous copy choice.
// Condition only the random cost channel; keep the existing weighted order law
// and compare the complete original callback history after replay.
inline std::vector<int> rolls(const Evidence& e) {
    std::vector<int> result;
    const int count=e.confused&&e.before[2]>=0?4:1;
    for(int value=0;value<count;++value)
        if(trace::drawProfile(e.before,value,e.confused,e.corruption)==e.after)result.push_back(value);
    if(result.empty())throw Failure("impossible_observed_draw_cost");
    return result;
}
struct Program {
    std::vector<Evidence> events;
    std::size_t cursor=0;
    Budget* budget=nullptr;
    bool active() const {return cursor<events.size();}
    int roll(const sts::CardInstance& card,int sampled) {
        if(!active())return sampled;
        const auto& e=events[cursor];
        if(!e.confused||e.before[2]<0||trace::project(card)!=e.before)
            throw Failure("observed_draw_cost_preimage_mismatch");
        const auto possible=rolls(e);
        if(budget)budget->tick(4);
        // For this channel each changed cost identifies a unique roll; in the
        // unconstrained case all four remain possible. Never bias a partial set.
        if(possible.size()==1)return possible.front();
        if(possible.size()!=4)throw Failure("unreviewed_observed_draw_cost_preimages");
        return sampled;
    }
    void drawn(const sts::CardInstance& card,bool confused,bool corruption) {
        if(!active())return;
        const auto& e=events[cursor];
        if(trace::project(card)!=e.after||confused!=e.confused||corruption!=e.corruption)
            throw Failure("observed_draw_cost_postimage_mismatch");
        ++cursor;
    }
};
inline thread_local Program* active=nullptr;
inline thread_local const sts::CardManager* target=nullptr;
struct Scope {
    Program* old; const sts::CardManager* oldTarget;
    Scope(Program* program,const sts::CardManager& cards):old(active),oldTarget(target) {
        if(program&&old)throw Failure("nested_observed_draw_cost_scope");
        active=program;target=program?&cards:nullptr;
    }
    ~Scope(){active=old;target=oldTarget;}
};
inline int confusionRoll(sts::BattleContext& battle,const sts::CardInstance& card) {
    // Preserve native stream advancement; this sampled stream is NOT the real
    // game's seed. Under the event law future events are independent of past
    // public outcomes; no native-seed-posterior equivalence is asserted.
    const int sampled=battle.cardRandomRng.random(3);
    return active&&target==&battle.cards?active->roll(card,sampled):sampled;
}
inline void drawn(const sts::CardManager& cards,const sts::CardInstance& card,bool confused,bool corruption) {
    if(active&&target==&cards)active->drawn(card,confused,corruption);
}
} // namespace spire_draw_costs
