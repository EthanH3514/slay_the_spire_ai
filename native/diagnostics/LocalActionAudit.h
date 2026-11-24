#pragma once
#include "BudgetTraceAudit.h"

namespace spire_search {
inline constexpr auto localActionVersion="public-local-action-audit-v1";

// Explicit offline intervention. Prefixes in this protocol must have identical
// public outcomes in every world; mismatch is an error, NOT a rejected sample.
inline J auditLocalAction(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                          const J& budget,const J& metadata,bool detail=false) {
    const auto& spec=budget.at("frozen_audit");
    if(spec.value("version","")!=localActionVersion)
        return auditBudgetTrace(roots,seeds,budget,metadata,detail);
    if(spec.size()!=6||!budget.at("max_seconds").is_null()||
       budget.value("tree_algorithm","")!=spire_tree::standaloneVersion||detail)
        throw std::invalid_argument("local_action_contract");
    const auto& prefix=spec.at("prefix");const auto& decision=spec.at("decision");
    auto candidates=spec.at("actions");const int longH=spec.at("long_horizon");
    const auto eval=spec.at("seeds").get<std::vector<RootSeeds>>();
    if(!prefix.is_array()||prefix.empty()||prefix.size()>4||longH!=192||
       eval.empty()||eval.size()>64||!candidates.is_array()||candidates.size()!=2||
       decision.size()!=2||!decision.contains("public")||!decision.contains("legal")||
       spec.dump().size()>8*1024*1024)
        throw std::invalid_argument("local_action_limits");
    for(const auto& p:prefix)if(p.size()!=4||!p.contains("action")||!p.contains("observations")||
       !p.contains("public")||!p.contains("legal"))throw std::invalid_argument("local_action_prefix_contract");
    std::sort(candidates.begin(),candidates.end());
    if(candidates[0]==candidates[1])throw std::invalid_argument("local_action_duplicate");
    for(const auto& a:candidates)if(std::find(decision.at("legal").begin(),decision.at("legal").end(),a)==decision.at("legal").end())
        throw std::invalid_argument("local_action_candidate");
    for(std::size_t i=0;i<eval.size();++i)if(std::find(seeds.begin(),seeds.end(),eval[i])!=seeds.end()||
        std::find(eval.begin(),eval.begin()+i,eval[i])!=eval.begin()+i)
        throw std::invalid_argument("local_action_seed_overlap");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    std::unique_ptr<spire_tree::FrozenTree> frozen;
    auto observer=[&](const spire_tree::StandaloneTree& t,int,bool finish){
        if(finish)frozen=std::make_unique<spire_tree::FrozenTree>(limits,t.root());};
    auto result=searchStandaloneTree(roots,seeds,budget,metadata,false,observer);
    if(!frozen||!result.at("refusal").is_null())throw std::runtime_error("local_action_construction");
    const auto before=frozen->diagnostic();const auto& input=roots.evidence();
    J audit={{"version",localActionVersion},{"status","incomplete"},{"rows",J::array()},
        {"batches",J::array()},{"final_tree",before},{"decision_node",nullptr},{"frozen_choice",nullptr},
        {"steps",0},{"prefix_steps",0},{"samples",0},{"values",0},{"policies",0},
        {"prefix_length",prefix.size()},{"worlds",eval.size()},{"short_horizon",48},{"long_horizon",longH},
        {"step_cap",eval.size()*candidates.size()*longH}};
    const auto started=Clock::now();
    for(std::size_t sample=0;sample<eval.size();++sample){
        auto batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
        auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("local_action_deadline");};
        batch->prepare(0,eval[sample],until);
        for(const auto& action:candidates){
            checkTime();auto world=batch->branch(0);
            auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
            auto view=project();if(view!=input.policy)throw std::runtime_error("local_action_root_projection");
            const auto* node=frozen->root();
            for(const auto& p:prefix){
                checkTime();auto legal=actions(*world.battle,true);
                if(view!=p.at("public")||legal!=p.at("legal")||
                   std::find(legal.begin(),legal.end(),p.at("action"))==legal.end())
                    throw std::runtime_error("local_action_prefix_state");
                const auto* edge=frozen->edge(node,p.at("action"));
                const auto observed=batch->advancePublic(world,p.at("action"));view=project();
                audit["steps"]=audit["steps"].get<int>()+1;
                audit["prefix_steps"]=audit["prefix_steps"].get<int>()+1;
                if(observed!=p.at("observations").get<std::vector<std::string>>()||outcome(*world.battle))
                    throw std::runtime_error("local_action_prefix_observations");
                guard(view,metadata);node=frozen->child(edge,spire_tree::observationKey(observed,limits.observationBytes),actions(*world.battle,true));
            }
            auto legal=actions(*world.battle,true);
            if(view!=decision.at("public")||legal!=decision.at("legal")||!node)
                throw std::runtime_error("local_action_decision_state");
            const auto* ordinary=frozen->choose(node,legal);
            if(!ordinary)throw std::runtime_error("local_action_unvisited_decision");
            const auto stats=publicNodeStats(node);const auto choice=J::parse(ordinary->action);
            if(audit["decision_node"].is_null()){audit["decision_node"]=stats;audit["frozen_choice"]=choice;}
            else if(audit["decision_node"]!=stats||audit["frozen_choice"]!=choice)
                throw std::runtime_error("local_action_policy_changed");
            J row={{"sample",sample},{"action",action},{"steps",prefix.size()},
                {"tree_steps",0},{"leaf_steps",0},{"unknown_history",0},{"unvisited_frontier",0},
                {"short",nullptr},{"long",nullptr}};
            for(int depth=int(prefix.size());depth<longH;++depth){
                checkTime();legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;J chosen;
                if(depth==int(prefix.size())){chosen=action;edge=frozen->edge(node,chosen);}
                else {
                    edge=frozen->choose(node,legal);chosen=edge?J::parse(edge->action):combatChoice(view,legal,metadata);
                    const char* field=edge?"tree_steps":"leaf_steps";row[field]=row[field].get<int>()+1;
                    if(!edge&&node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;
                    audit["policies"]=audit["policies"].get<int>()+1;
                }
                if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("local_action_illegal");
                const auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                const int terminal=outcome(*world.battle);row["steps"]=depth+1;
                audit["steps"]=audit["steps"].get<int>()+1;
                if(terminal||depth+1==48||depth+1==longH){
                    auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal));
                    audit["values"]=audit["values"].get<int>()+1;
                    if(row["short"].is_null())row["short"]=value;
                    if(terminal||depth+1==longH)row["long"]=value;
                }
                if(terminal)break;
                guard(view,metadata);
                if(depth+1<longH&&edge){node=frozen->child(edge,spire_tree::observationKey(observed,limits.observationBytes),actions(*world.battle,true));
                    if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                }else node=nullptr; // Never rejoin after an unknown-history/rule exit.
            }
            audit["rows"].push_back(row);audit["samples"]=audit["samples"].get<int>()+1;
        }
        audit["batches"].push_back(batch->diagnostics());
    }
    if(before!=frozen->diagnostic())throw std::runtime_error("local_action_snapshot_mutated");
    audit["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    audit["snapshots_unchanged"]=true;audit["prefix_exact"]=true;audit["status"]="complete";
    result["local_action"]=std::move(audit);return result;
}
} // namespace spire_search
