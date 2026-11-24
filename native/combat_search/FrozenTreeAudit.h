#pragma once
#include "StandaloneTree.h"
#include "FrozenTree.h"

namespace spire_search {
inline constexpr auto frozenAuditVersion="public-frozen-tree-audit-v1";
inline constexpr auto frozenFinalAuditVersion="public-frozen-final-audit-v1";

inline J auditStandaloneTree(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                            const J& budget,const J& metadata,bool detail=false) {
    const auto& spec=budget.at("frozen_audit");
    const bool finalOnly=spec.at("version")==frozenFinalAuditVersion;
    if(spec.size()!=5||(!finalOnly&&spec.at("version")!=frozenAuditVersion)||!budget.at("max_seconds").is_null())
        throw std::invalid_argument("frozen_audit_contract");
    const int checkpoint=spec.at("checkpoint_steps"),shortH=spec.at("short_horizon"),longH=spec.at("long_horizon");
    const auto evalSeeds=spec.at("seeds").get<std::vector<RootSeeds>>();
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    if(checkpoint<1||checkpoint>=budget.at("max_branch_steps").get<int>()||shortH<1||longH<shortH||longH>512||
       evalSeeds.empty()||evalSeeds.size()>64||roots.evidence().candidates.size()>64)
        throw std::invalid_argument("frozen_audit_limits");
    for(std::size_t i=0;i<evalSeeds.size();++i){
        if(std::find(seeds.begin(),seeds.end(),evalSeeds[i])!=seeds.end()||
           std::find(evalSeeds.begin(),evalSeeds.begin()+i,evalSeeds[i])!=evalSeeds.begin()+i)
            throw std::invalid_argument("frozen_audit_seed_overlap");
    }
    std::unique_ptr<spire_tree::FrozenTree> early,final;
    int earlySteps=0,finalSteps=0;J earlyEvidence;
    auto observer=[&](const spire_tree::StandaloneTree& tree,int steps,bool finish){
        if(!finalOnly&&!early&&steps>=checkpoint){early=std::make_unique<spire_tree::FrozenTree>(limits,tree.root());
            earlySteps=steps;earlyEvidence=tree.evidence();}
        if(finish){final=std::make_unique<spire_tree::FrozenTree>(limits,tree.root());finalSteps=steps;}
    };
    auto result=searchStandaloneTree(roots,seeds,budget,metadata,detail,observer);
    J audit={{"version",finalOnly?frozenFinalAuditVersion:frozenAuditVersion},{"status","incomplete"},{"error",nullptr},
        {"snapshot_arena_cap_bytes",(finalOnly?1:2)*limits.bytes},{"checkpoint_steps",earlySteps},{"final_steps",finalSteps},
        {"short_horizon",shortH},{"long_horizon",longH},{"worlds",evalSeeds.size()},
        {"rows",J::array()},{"batches",J::array()},{"steps",0},{"values",0},{"samples",0},{"policies",0},
        {"step_cap",evalSeeds.size()*roots.evidence().candidates.size()*(finalOnly?1:3)*longH},
        {"production_work_included",false},{"snapshot_missing_at_checkpoint",!early}};
    if(!final){audit["error"]="construction_incomplete";result["frozen_audit"]=audit;return result;}
    // If construction terminates before the declared checkpoint, final is also
    // the early policy. This is explicitly reported, never a second learned arm.
    const auto* earlyPolicy=early?early.get():final.get();
    audit["early_tree"]=earlyPolicy->diagnostic();audit["final_tree"]=final->diagnostic();
    audit["early_evidence"]=early?earlyEvidence:result["root_evidence"];
    if(finalOnly){audit.erase("early_tree");audit.erase("early_evidence");
        audit.erase("checkpoint_steps");audit.erase("snapshot_missing_at_checkpoint");audit["only_final"]=true;}
    const auto earlyBefore=earlyPolicy->diagnostic(),finalBefore=final->diagnostic();
    const auto started=Clock::now();std::unique_ptr<ConditionalRoots> batch;
    try {
        const auto& input=roots.evidence();J candidates=input.candidates;std::sort(candidates.begin(),candidates.end());
        for(std::size_t sample=0;sample<evalSeeds.size();++sample){
            batch=roots.diagnosticBatch();
            // A separate predeclared cap per diagnostic world, shared across ALL
            // actions/arms/rebuilds. Batches are counted, never hidden restarts.
            const auto until=Clock::now()+std::chrono::seconds(30);
            auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("frozen_audit_deadline");};
            batch->prepare(0,evalSeeds[sample],until);
            for(const auto& action:candidates)for(int armIndex=0;armIndex<(finalOnly?1:3);++armIndex){
                const int arm=finalOnly?1:armIndex;
                checkTime();audit["samples"]=audit["samples"].get<int>()+1;
                auto world=batch->branch(0);auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
                auto view=project();if(view!=input.policy)throw std::runtime_error("frozen_audit_root_projection");
                const auto* policy=arm==0?earlyPolicy:arm==1?final.get():nullptr;
                const spire_tree::Node* node=policy?policy->root():nullptr;
                J row={{"sample",sample},{"action",action},{"arm",arm==0?"early":arm==1?"final":"rule"},
                    {"steps",0},{"tree_steps",0},{"leaf_steps",0},{"changed_from_rule",0},
                    {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
                for(int depth=0;depth<longH;++depth){
                    checkTime();auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;J chosen;
                    if(!depth){chosen=action;if(policy)edge=policy->edge(node,action);}
                    else {
                        audit["policies"]=audit["policies"].get<int>()+1;auto prior=combatChoice(view,legal,metadata);
                        edge=policy?policy->choose(node,legal):nullptr;chosen=edge?J::parse(edge->action):prior;
                        if(edge){row["tree_steps"]=row["tree_steps"].get<int>()+1;
                            row["changed_from_rule"]=row["changed_from_rule"].get<int>()+(chosen!=prior);}
                        else {row["leaf_steps"]=row["leaf_steps"].get<int>()+1;
                            if(node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;}
                    }
                    if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("frozen_audit_illegal");
                    audit["steps"]=audit["steps"].get<int>()+1;row["steps"]=depth+1;
                    auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                    auto key=spire_tree::observationKey(observed,limits.observationBytes);
                    const int terminal=outcome(*world.battle);
                    if(terminal||depth+1==shortH||depth+1==longH){
                        audit["values"]=audit["values"].get<int>()+1;
                        auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal),spire_objective::name(budget));
                        if(terminal||depth+1==longH)row["long"]=value;
                        if(row["short"].is_null())row["short"]=value;
                    }
                    if(terminal)break;
                    guard(view,metadata);
                    if(depth+1<longH&&policy&&edge){
                        node=policy->child(edge,key,actions(*world.battle,true));
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    } else node=nullptr;
                }
                audit["rows"].push_back(std::move(row));
            }
            audit["batches"].push_back(batch->diagnostics());batch.reset();
        }
        if(earlyBefore!=earlyPolicy->diagnostic()||finalBefore!=final->diagnostic())
            throw std::runtime_error("frozen_audit_mutated_snapshot");
        audit["snapshots_unchanged"]=true;audit["status"]="complete";
    }catch(const std::exception& e){audit["error"]=e.what();if(batch)audit["batches"].push_back(batch->diagnostics());}
    audit["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    result["frozen_audit"]=std::move(audit);return result;
}
} // namespace spire_search
