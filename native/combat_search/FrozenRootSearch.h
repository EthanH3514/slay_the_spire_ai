#pragma once
#include "StandaloneTree.h"

namespace spire_tree {
inline constexpr auto frozenRootVersion="public-history-tree-v10-frozen-root";
// A const view while the construction tree is alive/frozen; no second arena.
inline const Edge* frozenChoice(const Node* node,const J& legal) {
    if(!node)return nullptr;
    std::vector<std::string> names;for(const auto& a:legal)names.push_back(a.dump());
    std::sort(names.begin(),names.end());auto* edge=node->edges;
    for(const auto& name:names){if(!edge||edge->action!=name)throw std::runtime_error("frozen_public_legal_set_mismatch");edge=edge->next;}
    if(edge)throw std::runtime_error("frozen_public_legal_set_mismatch");
    const Edge* best=nullptr;double score=-1e100;
    for(auto* e=node->edges;e;e=e->next)if(e->visits){const double q=e->sum/e->visits;
        if(!best||q>score){best=e;score=q;}}
    return best;
}
inline const Node* frozenChild(const Edge* edge,std::string_view observed) {
    if(edge)for(auto* n=edge->children;n;n=n->sibling)if(n->observation==observed)return n;
    return nullptr;
}
} // namespace spire_tree

namespace spire_search {
inline J searchFrozenRoot(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                          const J& budget,const J& metadata,bool detail=false) {
    const auto started=Clock::now();
    if(budget.value("tree_algorithm","")!=spire_tree::frozenRootVersion||budget.contains("frozen_audit")||!spire_objective::enabled(budget))
        throw std::invalid_argument("frozen_root_version_contract");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    const auto evalSeeds=budget.at("frozen_evaluation_seeds").get<std::vector<RootSeeds>>();
    const auto& input=roots.evidence();const int n=int(input.candidates.size());
    const int cap=std::min({limits.trainingRollouts,number(budget,"max_samples"),number(budget,"max_value_evaluations"),number(budget,"max_branch_steps")});
    const int stepCap=number(budget,"max_branch_steps"),horizon=n?std::min(number(budget,"rollout_horizon"),stepCap/n):0;
    if(n<2||seeds.size()!=std::size_t(cap)||evalSeeds.size()!=std::size_t(cap))throw std::invalid_argument("frozen_root_seed_plan");
    for(std::size_t i=0;i<evalSeeds.size();++i)if(std::find(seeds.begin(),seeds.end(),evalSeeds[i])!=seeds.end()||
        std::find(evalSeeds.begin(),evalSeeds.begin()+i,evalSeeds[i])!=evalSeeds.begin()+i)throw std::invalid_argument("frozen_root_seed_overlap");
    const double seconds=budget.at("max_seconds").is_null()?-1:budget.at("max_seconds").get<double>();
    if(!budget.at("max_seconds").is_null()&&(!std::isfinite(seconds)||seconds<0))throw std::invalid_argument("frozen_root_deadline_contract");
    const auto until=seconds<0?Clock::time_point::max():started+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("frozen_root_deadline");};
    if(!horizon||cap<2*n||stepCap/2<n*horizon)
        return {{"contract",spire_tree::frozenRootVersion},{"action",nullptr},{"refusal","frozen_root_coverage_budget"},
            {"stop","coverage_budget"},{"unavailable",""},{"samples",0},{"steps",0},{"values",0},{"policies",0},
            {"discarded",0},{"peak",0},{"root_evidence",J::array()},{"tree",nullptr},{"root_sampling",roots.diagnostics()}};
    J training=budget;training.erase("frozen_evaluation_seeds");training["tree_algorithm"]=spire_tree::standaloneVersion;
    const int trainCap=std::min(cap-n,stepCap/2);
    training["max_branch_steps"]=stepCap/2;training["tree"]["training_rollouts"]=trainCap;
    if(seconds>=0)training["max_seconds"]=std::max(0.,std::chrono::duration<double>(until-Clock::now()).count());
    std::vector<RootSeeds> trainingSeeds(seeds.begin(),seeds.begin()+trainCap);
    int samples=0,steps=0,values=0,policies=0,rounds=0,treeSteps=0,leafSteps=0,trainSamples=0;
    J traces=J::array();std::string stop="not_started",error;
    J rootActions=J::array(),planningRootSampling;
    std::vector<std::vector<spire_objective::Outcome>> rows;
    auto observer=[&](const spire_tree::StandaloneTree& tree,int trainingSteps,bool finish){
        if(!finish)return;
        trainSamples=tree.completed();std::vector<const spire_tree::Edge*> edges;
        for(auto* e=tree.root()->edges;e;e=e->next){edges.push_back(e);rootActions.push_back(J::parse(e->action));}
        planningRootSampling=roots.diagnostics();
        rows.resize(edges.size());int prepared=roots.diagnostics().at("prepared_rounds");
        try {
            while(true){
                checkTime();
                if(stepCap-trainingSteps-steps<n*horizon){stop="step_budget";break;}
                if(cap-trainSamples-samples<n){stop="simulation_cap";break;}
                std::vector<spire_objective::Outcome> pending;
                // All forced roots/canonical order chosen before this world.
                ++samples;roots.prepare(prepared,evalSeeds.at(rounds),until);++prepared;
                for(int i=0;i<n;++i){
                    if(i)++samples;checkTime();auto world=roots.branch(prepared-1);
                    auto project=[&]{return roots.projectView(*world.battle,input.relics,world.knownTop,input.character);};
                    auto view=project();if(view!=input.policy)throw std::runtime_error("frozen_root_projection");
                    const spire_tree::Node* node=tree.root();J path=J::array();
                    for(int depth=0;depth<horizon;++depth){
                        checkTime();auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;J action;
                        if(!depth){edge=edges[i];action=J::parse(edge->action);}
                        else {++policies;auto prior=combatChoice(view,legal,metadata);
                            edge=spire_tree::frozenChoice(node,legal);action=edge?J::parse(edge->action):prior;}
                        if(std::find(legal.begin(),legal.end(),action)==legal.end())throw std::runtime_error("frozen_root_illegal");
                        if(edge)++treeSteps;else ++leafSteps;
                        if(detail&&rounds<1)path.push_back({{"action",action},{"tree_policy",bool(edge)},{"public",view}});
                        ++steps;auto observed=roots.advancePublic(world,action);view=project();checkTime();
                        auto key=spire_tree::observationKey(observed,limits.observationBytes);
                        if(outcome(*world.battle))break;
                        guard(view,metadata);node=spire_tree::frozenChild(edge,key);
                    }
                    ++values;const auto row=spire_objective::record(view,input.policy,outcome(*world.battle),combatValue(view,outcome(*world.battle)),spire_objective::name(budget));
                    checkTime();pending.push_back(spire_objective::Outcome::read(row));
                    if(detail&&rounds<1)traces.push_back({{"round",rounds},{"action",J::parse(edges[i]->action)},{"steps",path},{"outcome",row}});
                }
                for(int i=0;i<n;++i)rows[i].push_back(pending[i]);++rounds;
            }
        }catch(const std::exception& e){const std::string why=e.what();
            if(why=="frozen_root_deadline"||roots.failureCode()=="deadline")stop="deadline";
            else {stop="tree_refused";error=why;}}
    };
    auto learning=searchStandaloneTree(roots,trainingSeeds,training,metadata,detail,observer);
    if(!planningRootSampling.is_null())learning["root_sampling"]=planningRootSampling;
    if(!learning.at("refusal").is_null()){error=learning.at("refusal").get<std::string>();stop=learning.at("stop");}
    J evidence=J::array();int best=-1;spire_objective::Key bestKey{};
    for(int i=0;i<int(rows.size());++i){J outcomes=J::array();for(const auto& o:rows[i])outcomes.push_back(
        {{"value",o.value},{"hp",o.hp},{"resource_hp",o.resource},{"outcome",o.terminal}});
        auto key=rows[i].empty()?spire_objective::Key{}:spire_objective::summarize(rows[i]);
        evidence.push_back({{"action",rootActions[i]},{"visits",rows[i].size()},
            {"outcomes",outcomes},{"objective_key",rows[i].empty()?J():J(key)}});
        if(!rows[i].empty()&&(best<0||key>bestKey)){best=i;bestKey=key;}
    }
    if(!rounds&&error.empty())error="frozen_root_incomplete_evaluation";
    // The original tree is gone now. Root action JSON is owned by evidence;
    // never dereference captured edge pointers outside the observer lifetime.
    auto stats=learning.at("tree");stats.update({{"version",spire_tree::frozenRootVersion},
        {"recommendation","independent-frozen-continuation-distributional-objective"},{"simulation_cap",cap},
        {"construction_simulations",learning["tree"]["completed_simulations"]},{"evaluation_simulations",rounds*n},
        {"completed_simulations",learning["tree"]["completed_simulations"].get<int>()+rounds*n},
        {"interrupted_simulations",samples-rounds*n+learning["discarded"].get<int>()},
        {"unused_steps",stepCap-steps-learning["steps"].get<int>()},
        {"completed_evaluation_rounds",rounds},{"construction_step_cap",stepCap/2}});
    auto eval=J{{"samples",samples},{"steps",steps},{"values",values},{"policies",policies},{"rounds",rounds},
        {"tree_policy_steps",treeSteps},{"rule_leaf_steps",leafSteps},{"committed_simulations",rounds*n}};
    return {{"contract",spire_tree::frozenRootVersion},{"action",error.empty()&&best>=0?evidence[best]["action"]:J()},
        {"stop",stop},{"refusal",error.empty()?J():J(error)},{"unavailable",""},
        {"steps",steps+learning["steps"].get<int>()},{"samples",samples+learning["samples"].get<int>()},
        {"values",values+learning["values"].get<int>()},{"policies",policies+learning["policies"].get<int>()},
        {"discarded",samples-rounds*n+learning["discarded"].get<int>()},{"peak",1},
        {"root_evidence",evidence},{"tree",stats},{"learning",learning},{"evaluation",eval},
        {"traces",traces},{"root_sampling",roots.diagnostics()}};
}
} // namespace spire_search
