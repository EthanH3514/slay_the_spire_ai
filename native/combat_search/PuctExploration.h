#pragma once
#include "HistoryTree.h"
#include "BoundaryValue.h"
#include <cmath>
#include <stdexcept>

// Experiment-only allocation control. No value, prior, FPU or backup changes.
namespace spire_puct_exploration {
using J=nlohmann::json;
inline constexpr auto version="public-puct-exploration-v1";
inline thread_local double coefficient=1.;
inline thread_local bool active=false;
struct Scope {
    explicit Scope(const J& budget) {
        if(active)throw std::invalid_argument("puct_exploration_nested_scope");
        double next=1.;
        if(budget.contains("puct_exploration")) {
            const auto& c=budget.at("puct_exploration");
            if(!c.is_object()||c.size()!=2||!c.contains("version")||!c.contains("coefficient")
                ||c.at("version")!=version||!c.at("coefficient").is_number()
                ||budget.value("value_leaf","")!="learned"||budget.value("neural_prior","")!="learned"
                ||!budget.contains("max_seconds")||(!budget.at("max_seconds").is_null()&&
                   !spire_boundary_value::timedPuctAllowed(budget)))
                throw std::invalid_argument("puct_exploration_contract");
            next=c.at("coefficient").get<double>();
            if(!std::isfinite(next)||(next!=.5&&next!=1.&&next!=2.))
                throw std::invalid_argument("puct_exploration_coefficient");
        }
        coefficient=next;active=true;
    }
    ~Scope(){coefficient=1.;active=false;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
};
}
