#pragma once
#include "StandaloneTree.h"

namespace spire_tree {
inline constexpr auto roundVersion="public-history-tree-v11-root-rounds";
}

namespace spire_search {
// Same adaptive public-history continuation as V9. Only root allocation and
// root-world coupling change. A partial round never enters the root estimator.
inline J searchRootRounds(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                          const J& budget,const J& metadata,bool detail=false) {
    const auto started=Clock::now();
    const auto coupling=budget.value("root_coupling","");
    if(budget.value("tree_algorithm","")!=spire_tree::roundVersion||!spire_objective::enabled(budget)||
       (coupling!="common"&&coupling!="independent")||budget.contains("frozen_audit"))
        throw std::invalid_argument("root_round_version_contract");
    auto limits=spire_tree::Limits::parse(budget.at("tree"));
    for(const auto* k:{"samples_per_action","rollout_horizon","max_branch_steps","max_samples","max_value_evaluations"})
        if(!budget.at(k).is_number_integer()||budget.at(k)<1||budget.at(k)>2147483647)
            throw std::invalid_argument("root_round_budget");
    const auto& input=roots.evidence();const auto& legalRoot=input.candidates;
    if(legalRoot.size()<2||legalRoot.size()>2147483647u)throw std::invalid_argument("root_round_requires_choices");
    const int count=int(legalRoot.size()),stepCap=number(budget,"max_branch_steps");
    const int simulationCap=std::min({limits.trainingRollouts,number(budget,"max_samples"),
        number(budget,"max_value_evaluations"),stepCap});
    const int horizon=std::min(number(budget,"rollout_horizon"),stepCap/count);
    if(seeds.size()!=std::size_t(simulationCap))throw std::invalid_argument("root_round_seed_plan_size");
    const double seconds=budget.at("max_seconds").is_null()?-1:budget.at("max_seconds").get<double>();
    if(!budget.at("max_seconds").is_null()&&(!std::isfinite(seconds)||seconds<0))
        throw std::invalid_argument("root_round_deadline_contract");
    const auto until=seconds<0?Clock::time_point::max():started+
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto checkTime=[&](){if(Clock::now()>=until)throw std::runtime_error("root_round_deadline");};

    // Bounded scratch holds PUBLIC tree paths and outcome scalars, not worlds.
    // Charge its actual reserved capacity against the original tree allowance.
    struct Pending {spire_tree::StandaloneTree::Path path;spire_objective::Outcome outcome;};
    const auto originalBytes=limits.bytes;
    const auto minPending=std::size_t(count)*(sizeof(Pending)+limits.depth*sizeof(spire_tree::StandaloneTree::Path::value_type));
    if(minPending>=originalBytes)throw std::invalid_argument("root_round_pending_capacity");
    std::vector<Pending> pending(count);
    std::size_t pendingBytes=pending.capacity()*sizeof(Pending);
    for(auto& p:pending){p.path.reserve(limits.depth);pendingBytes+=p.path.capacity()*sizeof(p.path[0]);}
    if(pendingBytes>=originalBytes)throw std::invalid_argument("root_round_pending_capacity");
    limits.bytes-=pendingBytes;
    spire_tree::StandaloneTree tree(limits,legalRoot,simulationCap);
    int samples=0,steps=0,values=0,policies=0,prepared=0,rounds=0,treeSteps=0,leafSteps=0;
    J traces=J::array();
    auto finish=[&](const std::string& stop,const std::string& error=""){
        tree.freeze();auto* best=error.empty()?tree.recommend():nullptr;
        std::string refusal=error;if(!best&&refusal.empty())refusal="root_round_incomplete_coverage";
        auto stats=tree.diagnostic(detail);
        stats["retained_payload_bytes"]=stats.at("retained_payload_bytes").get<std::size_t>()+pendingBytes;
        stats.update({{"version",spire_tree::roundVersion},{"root_coupling",coupling},
            {"root_exploration","complete-canonical-round-before-world"},
            {"recommendation","complete-construction-round-distributional-objective"},
            {"construction_rounds",rounds},{"simulation_horizon",horizon},{"simulation_cap",simulationCap},
            {"tree_policy_steps",treeSteps},{"rule_leaf_steps",leafSteps},{"rule_calls",policies},
            {"interrupted_simulations",samples-int(tree.completed())},
            {"uncommitted_completed_outcomes",values-int(tree.completed())},
            {"interrupted_trajectories",samples-values},{"unused_steps",stepCap-steps},
            {"pending_slots",count},{"pending_capacity_bytes",pendingBytes},
            {"tree_arena_bytes",limits.bytes},{"arena_bytes",originalBytes},
            {"ignored_legacy_allocation_fields",{"samples_per_action","tree.training_horizon"}}});
        return J{{"contract",spire_tree::roundVersion},{"action",best?J::parse(best->action):J()},
            {"stop",stop},{"refusal",refusal.empty()?J():J(refusal)},{"unavailable",""},
            {"samples",samples},{"steps",steps},{"values",values},{"policies",policies},
            {"discarded",best?samples-int(tree.completed()):samples},{"peak",samples?1:0},
            {"root_evidence",tree.evidence()},{"tree",stats},{"traces",traces},
            {"root_sampling",roots.diagnostics()}};
    };
    try {
        checkTime();roots.validate(input.policy,legalRoot,metadata);
        if(roots.certificate(budget,metadata,seeds).size()>16*1024*1024)
            throw std::runtime_error("root_round_certificate_limit");
        if(!horizon||simulationCap<count)return finish("coverage_budget");
        while(true){
            checkTime();
            if(simulationCap-samples<count)return finish("simulation_cap");
            // Both coupling arms use the same conservative reservation. Do not
            // restart the step clock, conditioning budget or deadline here.
            if(stepCap-steps<count*horizon)return finish("step_budget");
            int slot=0;
            // Canonical root schedule exists before constructing any latent world.
            for(auto* first=tree.root()->edges;first;first=first->next,++slot){
                checkTime();auto& p=pending[slot];p.path.clear();
                const int index=samples++;
                if(coupling=="independent"||slot==0){
                    roots.prepare(prepared,seeds.at(coupling=="common"?rounds:index),until);++prepared;
                }
                auto world=roots.branch(prepared-1);
                auto project=[&]{return roots.projectView(*world.battle,input.relics,world.knownTop,input.character);};
                auto view=project();checkTime();
                if(view!=input.policy)throw std::runtime_error("root_round_projection");
                auto legal=actions(*world.battle,true);tree.checkRoot(legal);
                auto* node=tree.root();unsigned newNodes=0;
                J traceSteps=J::array();const bool capture=detail&&traces.size()<8;
                for(int depth=0;depth<horizon;++depth){
                    checkTime();spire_tree::Edge* edge=nullptr;J action;
                    if(depth==0){edge=first;action=J::parse(first->action);}
                    else {++policies;auto prior=combatChoice(view,legal,metadata);checkTime();
                        edge=node?tree.select(node,legal,prior):nullptr;action=edge?J::parse(edge->action):prior;}
                    if(std::find(legal.begin(),legal.end(),action)==legal.end())
                        throw std::runtime_error("root_round_illegal_action");
                    if(edge){++treeSteps;p.path.emplace_back(node,edge);}else ++leafSteps;
                    if(capture)traceSteps.push_back({{"depth",depth},{"node",node?J(node->id):J()},
                        {"action",action},{"tree_policy",bool(edge)},{"public",view}});
                    ++steps;auto observed=roots.advancePublic(world,action);view=project();checkTime();
                    auto key=spire_tree::observationKey(observed,limits.observationBytes);
                    if(outcome(*world.battle))break;
                    guard(view,metadata);if(depth+1==horizon)break;
                    legal=actions(*world.battle,true);node=tree.descend(node,edge,key,legal,newNodes);
                }
                ++values;const int terminal=outcome(*world.battle);const double value=combatValue(view,terminal);
                if(!std::isfinite(value))throw std::runtime_error("root_round_nonfinite_value");
                checkTime();auto row=spire_objective::record(view,input.policy,terminal,value,spire_objective::name(budget));
                p.outcome={row.at("value"),row.at("hp"),row.at("resource_hp"),row.at("outcome")};
                if(capture)traces.push_back({{"simulation",index},{"construction_round",rounds},
                    {"steps",traceSteps},{"outcome",row},{"path_length",p.path.size()},{"new_nodes",newNodes}});
            }
            checkTime();
            // Transactional statistical commit: no root gains a partial-round
            // advantage. Distinct roots never share descendants within a round.
            for(const auto& p:pending)tree.backup(p.path,p.outcome);
            ++rounds;
        }
    }catch(const std::exception& e){
        const std::string error=e.what();
        if(error=="root_round_deadline"||roots.failureCode()=="deadline")return finish("deadline");
        return finish("tree_refused",error);
    }
}
} // namespace spire_search
