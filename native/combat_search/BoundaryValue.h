#pragma once
// Experiment-only scheduling. No game state, RNG or model parameters.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include "../draw_belief/Exact.h"
namespace spire_boundary_value {
using J=nlohmann::json;
inline constexpr auto version="public-boundary-value-v4";
struct Config { bool enabled=false, direct=false; unsigned limit=1024; };
inline thread_local const Config* active=nullptr;
inline unsigned limit(){return active?active->limit:1024;}
inline unsigned atoms(){return 63*limit();}
inline Config parse(const J& budget){
    Config c;
    if(!budget.contains("boundary_value"))return c;
    const auto& x=budget.at("boundary_value");
    if(!x.is_object()||x.size()!=3||!x.contains("version")||!x.contains("mode")||!x.contains("limit")||
       x.at("version")!=version||!x.at("limit").is_number_integer())
        throw std::invalid_argument("boundary_value_config");
    const auto n=x.at("limit").get<long long>();
    if(n<1||n>4096||(x.at("mode")!="control"&&x.at("mode")!="direct")||
       budget.value("value_leaf","")!="learned"||budget.value("leaf_actor","")!="trained"||
       budget.value("neural_prior","")!="learned"||budget.value("rollout_horizon",0)!=16||
       budget.at("max_samples")!=n||budget.at("max_value_evaluations")!=n||
       budget.at("tree").at("training_rollouts")!=n)
        throw std::invalid_argument("boundary_value_contract");
    for(const char* name:{"frontier_collection","paired_value_audit","objective_audit","frozen_audit"})
        if(budget.contains(name))throw std::invalid_argument("boundary_value_auxiliary_capture");
    c.enabled=true;c.direct=x.at("mode")=="direct";c.limit=unsigned(n);return c;
}
struct Scope {
    Config config;
    explicit Scope(const J& budget):config(parse(budget)){
        if(active)throw std::logic_error("nested_boundary_value");
        if(config.enabled)active=&config;
    }
    ~Scope(){if(config.enabled)active=nullptr;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
};
// Only a validated, active boundary experiment may relax the old untimed PUCT
// contract. A key alone must not enable the exception, nor leak it to later calls.
inline bool timedPuctAllowed(const J& budget){
    if(!active||!active->enabled||!budget.contains("boundary_value")||
       !budget.contains("max_seconds")||!budget.at("max_seconds").is_number())return false;
    const auto c=parse(budget);
    const double seconds=budget.at("max_seconds").get<double>();
    return c.enabled&&c.direct==active->direct&&c.limit==active->limit&&std::isfinite(seconds)&&seconds>=0;
}
inline double remainingSeconds(double seconds,double elapsed){
    if(!std::isfinite(seconds)||seconds<0||!std::isfinite(elapsed)||elapsed<0)
        throw std::invalid_argument("boundary_value_deadline");
    return std::max(0.,seconds-elapsed);
}
// The shared published belief must never be invalidated by a cancelled search.
// compact() is a strong-exception-safe copy of the complete exact distribution.
// Copy once per timed request (not per rollout), charge the same work/deadline,
// and restore the source's bookkeeping even when copying itself is interrupted.
template<class Order> struct OrderSnapshot {
    std::unique_ptr<Order> value;
    Order& get(Order& prior,spire_belief::Budget& work){
        if(value)return *value;
        auto& source=prior.belief.engine.budget;
        struct Restore { spire_belief::Budget& source;spire_belief::Budget saved;
            ~Restore(){source=saved;} } restore{source,source};
        source.begin(work.deadline);
        auto copy=[&](){try{return prior.compact();}catch(...){
            const auto failure=std::current_exception();
            try{work.tick(source.work);}catch(...){std::rethrow_exception(failure);}
            std::rethrow_exception(failure);
        }}();
        work.tick(source.work);value=std::move(copy);return *value;
    }
};
inline bool stop(bool direct,unsigned depth,bool hasNode){return direct&&depth>0&&!hasNode;}
struct Counts {
    unsigned cuts=0;std::array<unsigned,17> depths{};
    bool cut(bool direct,unsigned depth,bool hasNode){
        if(!stop(direct,depth,hasNode))return false;
        if(depth>=depths.size())throw std::logic_error("boundary_value_depth");
        ++cuts;++depths[depth];return true;
    }
    J json(const Config& c,const std::string& rootFailure="")const{return {{"version",version},{"mode",c.direct?"direct":"control"},
        {"limit",c.limit},{"boundary_cutoffs",cuts},{"cutoff_depths",depths},
        {"root_failure_code",rootFailure},
        {"new_node_cap_unchanged",3},{"value_target","frozen-R2-greedy-not-current-search"}};}
};
}
