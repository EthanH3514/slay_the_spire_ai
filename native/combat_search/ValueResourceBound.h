#pragma once
#include "TerminalProof.h"
#include <optional>

// Optional projection onto a proven public support bound. Outcome logits and
// observed terminals are untouched. Unknown capabilities leave inference exact.
namespace spire_value_resource_bound {
using J=nlohmann::json;
inline constexpr auto version="public-value-resource-bound-v1";
class Scope;
inline thread_local Scope* current=nullptr;

class Scope {
    bool enabled_=false,closed_=false;
    std::string gap_;
    double rootHp_=0,maximumExcess_=0;
    unsigned calls_=0,changedCalls_=0,changedQuantiles_=0;
public:
    Scope(const J& budget,const J& root,const J& metadata) {
        if(current)throw std::invalid_argument("value_resource_bound_nested_scope");
        if(budget.contains("value_resource_bound")) {
            if(budget.at("value_resource_bound")!=version||
                    budget.value("terminal_proof","")!=spire_terminal_proof::version||
                    budget.value("value_leaf","")!="learned")
                throw std::invalid_argument("value_resource_bound_contract");
            enabled_=true;gap_=spire_terminal_proof::boundGap(root,metadata);closed_=gap_.empty();
            if(closed_) {
                rootHp_=root.at("player").at("current_hp").get<double>();
                if(!std::isfinite(rootHp_)||rootHp_<=0)throw std::invalid_argument("value_resource_bound_root_hp");
            }
        }
        current=this;
    }
    ~Scope(){current=nullptr;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
    bool enabled()const{return enabled_;}
    std::optional<double> apply(J& raw,const J& view) {
        if(!enabled_)return {};
        ++calls_;if(!closed_)return {};
        const double hp=view.at("player").at("current_hp").get<double>();
        if(!std::isfinite(hp)||hp<=0||hp>rootHp_)
            throw std::runtime_error("value_resource_bound_capability_violated");
        auto& q=raw.at("resource_quantiles");
        if(!q.is_array()||q.size()!=3)throw std::invalid_argument("value_resource_bound_shape");
        const double cap=hp/100.;bool changed=false;
        for(auto& branch:q) {
            if(!branch.is_array()||branch.size()!=21)throw std::invalid_argument("value_resource_bound_shape");
            double previous=0;
            for(auto& x:branch) {
                const double v=x.get<double>();
                if(!std::isfinite(v)||v<previous)throw std::invalid_argument("value_resource_bound_quantile_order");
                previous=v;
                if(v>cap){maximumExcess_=std::max(maximumExcess_,100.*v-hp);x=cap;++changedQuantiles_;changed=true;}
            }
        }
        changedCalls_+=changed;return hp;
    }
    J diagnostic()const {
        return {{"version",version},{"closed",closed_},{"bound_gap",gap_},
                {"root_hp",closed_?J(rootHp_):J()},{"prediction_calls",calls_},
                {"changed_predictions",changedCalls_},{"changed_quantiles",changedQuantiles_},
                {"maximum_removed_excess_hp",maximumExcess_},{"outcome_logits_unchanged",true},
                {"terminal_observations_unchanged",true},{"extra_simulator_steps",0}};
    }
};
inline std::optional<double> apply(J& raw,const J& view) {
    return current?current->apply(raw,view):std::optional<double>{};
}
}
