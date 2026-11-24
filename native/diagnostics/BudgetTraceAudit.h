#pragma once
#include "../combat_search/FrozenTreeAudit.h"

namespace spire_search {
inline constexpr auto budgetTraceVersion="public-budget-continuation-trace-v1";

inline J publicNodeStats(const spire_tree::Node* n) {
    if(!n)return nullptr;
    J edges=J::array();
    for(auto* e=n->edges;e;e=e->next)edges.push_back({{"action",J::parse(e->action)},
        {"visits",e->visits},{"mean",e->visits?J(e->sum/e->visits):J(nullptr)}});
    return {{"id",n->id},{"depth",n->depth},{"visits",n->visits},{"actions",edges}};
}

// Offline-only dispatch installed in an isolated header overlay. Normal audit
// contracts and the production implementation are untouched.
inline J auditBudgetTrace(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                         const J& budget,const J& metadata,bool detail=false) {
    const auto& spec=budget.at("frozen_audit");
    if(spec.value("version","")!=budgetTraceVersion)
        return auditStandaloneTree(roots,seeds,budget,metadata,detail);
    if(spec.size()!=6||!budget.at("max_seconds").is_null()||
       budget.value("tree_algorithm","")!=spire_tree::standaloneVersion||detail)
        throw std::invalid_argument("budget_trace_contract");
    const int traceDepth=spec.at("trace_depth").get<int>();
    const bool record=spec.at("record_paths").get<bool>();
    const int longH=spec.at("long_horizon").get<int>();
    auto candidates=spec.at("actions");
    const auto eval=spec.at("seeds").get<std::vector<RootSeeds>>();
    if(traceDepth<1||traceDepth>16||longH!=192||eval.empty()||eval.size()>64||
       !candidates.is_array()||candidates.empty()||candidates.size()>4)
        throw std::invalid_argument("budget_trace_limits");
    std::sort(candidates.begin(),candidates.end());
    if(std::adjacent_find(candidates.begin(),candidates.end())!=candidates.end())
        throw std::invalid_argument("budget_trace_duplicate");
    for(const auto& a:candidates)if(std::find(roots.evidence().candidates.begin(),roots.evidence().candidates.end(),a)==roots.evidence().candidates.end())
        throw std::invalid_argument("budget_trace_action");
    for(std::size_t i=0;i<eval.size();++i)if(std::find(seeds.begin(),seeds.end(),eval[i])!=seeds.end()||
        std::find(eval.begin(),eval.begin()+i,eval[i])!=eval.begin()+i)
        throw std::invalid_argument("budget_trace_seed_overlap");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    std::unique_ptr<spire_tree::FrozenTree> frozen;
    auto observer=[&](const spire_tree::StandaloneTree& t,int,bool finish){
        if(finish)frozen=std::make_unique<spire_tree::FrozenTree>(limits,t.root());};
    auto result=searchStandaloneTree(roots,seeds,budget,metadata,false,observer);
    if(!frozen||!result.at("refusal").is_null())throw std::runtime_error("budget_trace_construction");
    const auto before=frozen->diagnostic();
    J audit={{"version",budgetTraceVersion},{"rows",J::array()},{"paths",J::array()},
        {"batches",J::array()},{"final_tree",before},{"status","incomplete"},
        {"trace_depth",traceDepth},{"steps",0},{"samples",0},{"values",0},{"policies",0},
        {"step_cap",eval.size()*candidates.size()*longH},{"trace_bytes",0}};
    std::size_t traceBytes=0;constexpr std::size_t byteCap=128*1024*1024;
    const auto& input=roots.evidence();
    for(std::size_t sample=0;sample<eval.size();++sample){
        auto batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
        auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("budget_trace_deadline");};
        batch->prepare(0,eval[sample],until);
        for(const auto& action:candidates){
            auto world=batch->branch(0);auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
            auto view=project();if(view!=input.policy)throw std::runtime_error("budget_trace_projection");
            const auto* node=frozen->root();std::string reason="root";J path=J::array();
            J row={{"sample",sample},{"action",action},{"arm","final"},{"steps",0},
                {"tree_steps",0},{"leaf_steps",0},{"changed_from_rule",0},
                {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
            for(int depth=0;depth<longH;++depth){
                checkTime();const auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;
                J chosen,prior=nullptr;std::string mode="forced_root";
                if(depth==0){chosen=action;edge=frozen->edge(node,action);}
                else {
                    prior=combatChoice(view,legal,metadata);edge=frozen->choose(node,legal);
                    chosen=edge?J::parse(edge->action):prior;
                    mode=edge?"visited_tree":node?"unvisited_frontier":"rule_after_exit";
                    if(edge){row["tree_steps"]=row["tree_steps"].get<int>()+1;
                        row["changed_from_rule"]=row["changed_from_rule"].get<int>()+(chosen!=prior);}
                    else {row["leaf_steps"]=row["leaf_steps"].get<int>()+1;
                        if(node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;}
                    audit["policies"]=audit["policies"].get<int>()+1;
                }
                if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("budget_trace_illegal");
                J recordRow;
                if(record&&depth<traceDepth)recordRow={{"depth",depth},{"public",view},{"legal",legal},
                    {"action",chosen},{"rule",prior},{"mode",mode},{"entry",reason},{"node",publicNodeStats(node)}};
                const auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                const auto key=spire_tree::observationKey(observed,limits.observationBytes);
                const int terminal=outcome(*world.battle);row["steps"]=depth+1;
                audit["steps"]=audit["steps"].get<int>()+1;
                if(terminal||depth+1==48||depth+1==longH){
                    auto v=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal));
                    audit["values"]=audit["values"].get<int>()+1;
                    if(terminal||depth+1==longH)row["long"]=v;
                    if(row["short"].is_null())row["short"]=v;
                }
                if(!terminal){
                    guard(view,metadata);
                    if(depth+1<longH&&edge){node=frozen->child(edge,key,actions(*world.battle,true));
                        reason=node?"known_child":"unknown_history";
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    }else {if(node)reason="unvisited_frontier";node=nullptr;}
                }
                if(record&&depth<traceDepth){recordRow["observations"]=observed;recordRow["terminal"]=terminal;
                    recordRow["successor"]=terminal?"terminal":reason;
                    traceBytes+=recordRow.dump().size();if(traceBytes>byteCap)throw std::runtime_error("budget_trace_bytes");
                    path.push_back(std::move(recordRow));}
                if(terminal)break;
            }
            audit["rows"].push_back(row);audit["samples"]=audit["samples"].get<int>()+1;
            if(record)audit["paths"].push_back({{"sample",sample},{"action",action},{"path",path}});
        }
        audit["batches"].push_back(batch->diagnostics());
    }
    if(before!=frozen->diagnostic())throw std::runtime_error("budget_trace_snapshot_mutated");
    audit["trace_bytes"]=traceBytes;audit["snapshots_unchanged"]=true;audit["status"]="complete";
    result["budget_trace"]=std::move(audit);return result;
}
} // namespace spire_search
