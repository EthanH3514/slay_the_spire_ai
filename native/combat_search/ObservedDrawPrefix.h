#pragma once
#include "Engine.h"
#include "../draw_belief/ObservedDrawCosts.h"
#include "../draw_belief/Continuation.h"

namespace spire_observed_prefix {
using namespace spire_belief;
using J = nlohmann::json;
using Profile = trace::Card;
inline constexpr auto version = "public-weighted-draw-prefix-v1";
struct Seen { Profile after; bool confused, corruption; };

// Condition the existing order DAG, not a fresh uniform bag. Each branch has
// prior top probability * observed cost likelihood * remaining suffix likelihood.
// Restore ORIGINAL profiles in the conditioned DAG so anchor construction and
// all pending simulator actions still execute normally. Memoization shares equal
// (remaining-order node, observation index) subproblems; the existing arena/node/
// work limits bound growth, and observed-prefix recursion is limited to64.
class Law {
    trace::MechanismReplay& order_;
    Engine& engine_;
    const std::vector<Seen>& seen_;
    std::vector<Profile> remaining_;
    std::pmr::map<std::pair<Id,std::size_t>,Result> memo_;
public:
    Law(trace::MechanismReplay& order, const std::vector<Seen>& seen,
        std::vector<Profile> remaining)
        :order_(order),engine_(order.belief.engine),seen_(seen),
         remaining_(std::move(remaining)),memo_(&engine_.arena) {
        if(seen.empty()||seen.size()>64)throw Failure("observed_prefix_length_limit");
        std::sort(remaining_.begin(),remaining_.end());
    }
    Result condition(Id root,std::size_t depth=0) {
        auto& work=engine_.budget;work.tick();
        const auto key=std::make_pair(root,depth);
        if(auto it=memo_.find(key);it!=memo_.end())return it->second;
        if(memo_.size()>=work.limits.nodes)throw Failure("observed_prefix_state_limit");
        Result result{Rational(),root};
        if(depth==seen_.size()) {
            std::vector<Profile> actual;
            for(auto label:engine_.node(root).cards){work.tick();actual.push_back(order_.card(label));}
            std::sort(actual.begin(),actual.end());
            result.likelihood=Rational(actual==remaining_?1:0);
        } else {
            Cards options(engine_.node(root).cards,&engine_.arena);
            options.erase(std::unique(options.begin(),options.end()),options.end());
            Vec<Term> terms(&engine_.arena);
            const auto& observed=seen_[depth];
            for(auto label:options) {
                const auto before=order_.card(label);
                const int count=observed.confused&&before[2]>=0?4:1;
                int matches=0;
                for(int roll=0;roll<count;++roll){work.tick();
                    matches+=trace::drawProfile(before,roll,observed.confused,observed.corruption)==observed.after;}
                if(!matches)continue;
                const auto next=engine_.pop(root,label);if(!next.likelihood)continue;
                const auto suffix=condition(next.root,depth+1);if(!suffix.likelihood)continue;
                auto mass=Rational::multiply(next.likelihood,Rational::ratio(matches,count,work),work);
                mass=Rational::multiply(mass,suffix.likelihood,work);
                terms.push_back({engine_.concat(engine_.word(Cards({label},&engine_.arena)),suffix.root),mass});
            }
            if(!terms.empty())result=engine_.mixture(terms);
        }
        if(memo_.size()>=work.limits.nodes)throw Failure("observed_prefix_state_limit");
        memo_.emplace(key,result);return result;
    }
    std::size_t states()const{return memo_.size();}
};

struct Plan {
    bool eligible=false,prepared=false;
    std::size_t setupWork=0,states=0;
    std::vector<Seen> seen;
    std::vector<Profile> remaining;
    std::unique_ptr<trace::MechanismReplay> order;
    std::optional<Id> root;
    Plan(const std::vector<PublicStep>& tail,bool enabled) {
        if(!enabled||tail.empty()||J::parse(tail[0].action).value("kind","")=="opening_prelude")return;
        const auto& observations=tail[0].observations;
        bool mapped=false;
        for(std::size_t i=0;i<observations.size();++i) {
            setupWork+=observations[i].size()+1;const auto e=J::parse(observations[i]);
            if(e.value("kind",-1)!=int(Mechanism::Draw)) {
                // Only uninterrupted draws followed by their complete public
                // snapshot qualify. Import/maps/insertions retain their own law.
                if(i+1==observations.size()&&e.contains("resolved")&&e.contains("draw")&&!seen.empty()&&mapped) {
                    remaining=e.at("draw").get<std::vector<Profile>>();eligible=true;
                }
                return;
            }
            if(seen.size()==64)return;
            if(e.at("cards").size()!=1)throw Failure("public_draw_prefix_contract");
            seen.push_back({e.at("cards").at(0).get<Profile>(),e.at("confused"),e.at("corruption")});
            mapped|=seen.back().confused||seen.back().corruption;
        }
    }
    void prepare(trace::MechanismReplay& prior,Budget& work) {
        if(!eligible||prepared)return;
        work.tick(setupWork);prior.belief.engine.budget.begin(work.deadline);
        try{order=prior.compact();}
        catch(...){work.tick(prior.belief.engine.budget.work);throw;}
        work.tick(prior.belief.engine.budget.work);
        auto& b=order->belief;auto& local=b.engine.budget;
        local.begin(work.deadline);local.limits.work=std::min(local.limits.work,work.limits.work-work.work);
        try {
            Law law(*order,seen,remaining);const auto result=law.condition(b.publicRoot());
            if(!result.likelihood)throw Failure("impossible_observed_draw_prefix");
            root=result.root;states=law.states();prepared=true;
        } catch(...){work.tick(local.work);throw;}
        work.tick(local.work);
    }
    spire_draw_costs::Program costs(const sts::CardManager& cards,Budget& work)const {
        spire_draw_costs::Program p;p.budget=&work;
        if(!prepared||cards.drawPile.size()<seen.size())throw Failure("observed_prefix_not_prepared");
        for(std::size_t i=0;i<seen.size();++i){work.tick(5);
            spire_draw_costs::Evidence e{trace::project(cards.drawPile[cards.drawPile.size()-1-i]),
                seen[i].after,seen[i].confused,seen[i].corruption};
            spire_draw_costs::rolls(e);p.events.push_back(e);
        }
        return p;
    }
};
}
