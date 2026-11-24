#pragma once
#include "ValueDistribution.h"
#include "NeuralPrior.h"
#include "FrontierCollection.h"
#include "BoundaryValue.h"
#include "SearchValueContract.h"
#include "ValueResourceBound.h"
#include "ValueMemo.h"
namespace spire_value_leaf {
using J=nlohmann::json;
inline constexpr auto version="public-value-leaf-v2-escape";
inline thread_local std::unique_ptr<spire_rich_model::Model> model;
inline thread_local std::string identity,continuation,priorIdentity;
inline void bind(const J& envelope,const J& schema,const std::string& digest,const std::string& id) {
    const auto c=spire_search_value::binding(envelope);
    if(        c.at("weights_sha256")!=spire_cold::identity||spire_cold::identity.size()!=64||
        spire_neural_prior::continuationIdentity!=spire_cold::identity||!spire_neural_prior::model||id.size()!=64)
        throw std::invalid_argument("value_continuation_binding");
    auto next=std::make_unique<spire_rich_model::Model>(envelope.at("model"),schema,digest);
    model=std::move(next);identity=id;continuation=spire_cold::identity;priorIdentity=spire_neural_prior::identity;
}
class Evaluator {
    spire_frontier::Collector collection_;
    spire_value_memo::Cache memo_;
    std::string mode_;unsigned calls_=0;double seconds_=0;
    J infer(const J& view,const J& legal,const J& rich) {
        if(++calls_>spire_boundary_value::limit())throw std::runtime_error("value_call_limit");
        auto at=std::chrono::steady_clock::now();auto out=memo_.get(view,legal,rich,[&]{return model->inspect(view,legal,rich,false,false,true);});
        seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();return out;
    }
public:
    explicit Evaluator(const J& budget):collection_(budget),mode_(budget.value("value_leaf",std::string())) {
        if(mode_.empty())return;
        if((mode_!="terminal"&&mode_!="learned"&&mode_!="probe")||!model||!spire_neural_prior::active||
            continuation!=spire_cold::identity||priorIdentity!=spire_neural_prior::identity||
            budget.value("neural_prior","")!="learned"||budget.value("leaf_actor","")!="trained"||
            budget.value("combat_objective","")!=spire_value_distribution::version||
            budget.value("rollout_horizon",0)!=(mode_=="learned"?16:192)||budget.value("max_value_evaluations",0)>int(spire_boundary_value::limit()))
            throw std::invalid_argument("value_leaf_request");
        for(const auto* k:{"objective_audit","frozen_audit","exploration_profile","branching_profile"})
            if(budget.contains(k))throw std::invalid_argument("value_leaf_no_auxiliary_profiles");
    }
    spire_frontier::Collector& collection(){return collection_;}
    bool enabled()const{return !mode_.empty();}
    void probe(unsigned depth,const J& view,const J& legal,const J& rich) {
        collection_.observe(depth,view,legal,rich);
        if(mode_=="probe"&&depth==16)(void)infer(view,legal,rich);
    }
    spire_value_distribution::Distribution evaluate(const J& view,const J& root,int terminal,const J& legal,const J& rich) {
        const double reserve=spire_search::inventoryReserve(root),maxhp=std::max(1,spire_search::number(view.at("player"),"max_hp",80));
        if(collection_.enabled()){
            if(!terminal)throw std::runtime_error("frontier_nonterminal_label");
            collection_.finish({{"version","public-battle-result-v1"},{"outcome",terminal},
                {"hp",spire_search::number(view.at("player"),"current_hp",0)},
                {"potion_reserve",spire_search::inventoryReserve(view)}});
        }
        if(terminal)return spire_value_distribution::terminal(
            spire_objective::record(view,root,terminal,0.,spire_objective::terminalVersion),reserve,maxhp);
        if(mode_!="learned")throw std::runtime_error("value_terminal_control_censored");
        auto raw=infer(view,legal,rich);const auto cap=spire_value_resource_bound::apply(raw,view);
        auto result=spire_value_distribution::prediction(raw,reserve,maxhp);
        if(cap)result.evidence["resource_bound_hp"]=*cap;
        return result;
    }
    J diagnostic()const{return {{"version",version},{"mode",mode_},{"weights",identity},{"continuation_weights",continuation},
        {"prior_weights",priorIdentity},{"network_calls",calls_},{"inference_seconds",seconds_},{"call_limit",spire_boundary_value::limit()},
        {"value_memo",memo_.diagnostic()},{"prediction_is_terminal_observation",false},{"one_visit_per_simulation",true},{"full_belief",false}};}
};
}
