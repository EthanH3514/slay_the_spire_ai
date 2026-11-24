#pragma once
#include "StandaloneTree.h"
#include "FrozenTree.h"
#include "TeacherCollector.h"

namespace spire_search {
inline constexpr auto continuationAuditVersion="public-continuation-audit-v1";
inline constexpr auto continuationAuditAllVersion="public-continuation-audit-v2";

// Separate offline diagnostic. The immutable snapshot has only public histories
// and backed-up statistics; evaluation never changes construction/recommendation.
inline J auditContinuations(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                           const J& budget,const J& metadata,const J& spec,bool detail=false) {
    if(spec.value("version",std::string())==spire_teacher::version)
        return spire_teacher::collect(roots,budget,metadata,spec);
    const bool prior=budget.value("tree_algorithm","")==spire_tree::deepPriorVersion;
    const bool all=spec.is_object()&&spec.value("version","")==continuationAuditAllVersion;
    if((!prior&&budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)||
       !budget.at("max_seconds").is_null()||budget.contains("frozen_audit")||
       !spec.is_object()||spec.size()!=(all?6:5)||(!all&&spec.at("version")!=continuationAuditVersion))
        throw std::invalid_argument("continuation_audit_contract");
    const auto order=all?spec.at("evaluation_order").get<std::string>():"canonical";
    if(order!="canonical"&&order!="reverse")throw std::invalid_argument("continuation_audit_order");
    for(const auto* key:{"short_horizon","long_horizon"})
        if(!spec.at(key).is_number_integer())throw std::invalid_argument("continuation_audit_horizon");
    const int shortH=spec.at("short_horizon"),longH=spec.at("long_horizon");
    const auto eval=spec.at("seeds").get<std::vector<RootSeeds>>();
    auto candidates=spec.at("actions");
    if(shortH<1||longH<shortH||longH>192||eval.empty()||eval.size()>64||
       !candidates.is_array()||candidates.size()<2||candidates.size()>(all?8:2))
        throw std::invalid_argument("continuation_audit_limits");
    std::sort(candidates.begin(),candidates.end());
    if(std::adjacent_find(candidates.begin(),candidates.end())!=candidates.end())throw std::invalid_argument("continuation_audit_duplicate_action");
    for(const auto& action:candidates)
        if(std::find(roots.evidence().candidates.begin(),roots.evidence().candidates.end(),action)==roots.evidence().candidates.end())
            throw std::invalid_argument("continuation_audit_illegal_action");
    for(std::size_t i=0;i<eval.size();++i)
        if(std::find(seeds.begin(),seeds.end(),eval[i])!=seeds.end()||
           std::find(eval.begin(),eval.begin()+i,eval[i])!=eval.begin()+i)
            throw std::invalid_argument("continuation_audit_seed_overlap");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    std::unique_ptr<spire_tree::FrozenTree> frozen;
    auto observer=[&](const spire_tree::StandaloneTree& tree,int,bool finish){
        if(finish)frozen=std::make_unique<spire_tree::FrozenTree>(limits,tree.root());
    };
    auto result=prior?searchStandaloneTree<false,false,false,true>(roots,seeds,budget,metadata,detail,observer):
        searchStandaloneTree(roots,seeds,budget,metadata,detail,observer);
    J audit={{"version",all?continuationAuditAllVersion:continuationAuditVersion},{"status","incomplete"},{"error",nullptr},
        {"snapshot_arena_cap_bytes",limits.bytes},{"short_horizon",shortH},{"long_horizon",longH},
        {"actions",candidates},{"worlds",eval.size()},{"rows",J::array()},{"batches",J::array()},
        {"steps",0},{"samples",0},{"values",0},{"policies",0},{"pending",nullptr},
        {"step_cap",eval.size()*candidates.size()*2*longH},{"production_work_included",false},{"traces",J::array()}};
    if(all)audit["evaluation_order"]=order;
    if(!frozen||!result.at("refusal").is_null()){
        audit["error"]="construction_incomplete";result["continuation_audit"]=audit;return result;
    }
    audit["frozen_tree"]=frozen->diagnostic();const auto before=frozen->diagnostic();
    const auto started=Clock::now();std::unique_ptr<ConditionalRoots> batch;
    try {
        const auto& input=roots.evidence();
        for(std::size_t sample=0;sample<eval.size();++sample){
            batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
            auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("continuation_audit_deadline");};
            batch->prepare(0,eval[sample],until);
            // Identity/selection never depend on a sampled hidden world.
            auto traversal=candidates;if(order=="reverse")std::reverse(traversal.begin(),traversal.end());
            for(const auto& action:traversal)for(int armIndex=0;armIndex<2;++armIndex){
                const int arm=order=="reverse"?1-armIndex:armIndex;
                checkTime();audit["samples"]=audit["samples"].get<int>()+1;
                audit["pending"]={{"sample",sample},{"action",action},{"arm",arm?"rule":"frozen"},
                    {"steps",0},{"tree_steps",0},{"leaf_steps",0},{"changed_from_rule",0},
                    {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
                auto& row=audit["pending"];auto world=batch->branch(0);
                auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
                auto view=project();if(view!=input.policy)throw std::runtime_error("continuation_audit_projection");
                const spire_tree::Node* node=arm?nullptr:frozen->root();J path=J::array();
                for(int depth=0;depth<longH;++depth){
                    checkTime();auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;J chosen;
                    if(!depth){chosen=action;if(!arm)edge=frozen->edge(node,action);}
                    else {
                        audit["policies"]=audit["policies"].get<int>()+1;
                        auto rule=combatChoice(view,legal,metadata);edge=arm?nullptr:frozen->choose(node,legal);
                        chosen=edge?J::parse(edge->action):rule;
                        if(edge){row["tree_steps"]=row["tree_steps"].get<int>()+1;
                            row["changed_from_rule"]=row["changed_from_rule"].get<int>()+(chosen!=rule);}
                        else {row["leaf_steps"]=row["leaf_steps"].get<int>()+1;
                            if(node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;}
                    }
                    if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("continuation_audit_illegal");
                    if(detail&&sample<2)path.push_back({{"depth",depth},{"action",chosen},{"tree_policy",bool(edge)},{"public",view}});
                    row["steps"]=depth+1;
                    audit["steps"]=audit["steps"].get<int>()+1;
                    auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                    auto key=spire_tree::observationKey(observed,limits.observationBytes);const int terminal=outcome(*world.battle);
                    if(terminal||depth+1==shortH||depth+1==longH){
                        audit["values"]=audit["values"].get<int>()+1;
                        auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal),spire_objective::name(budget));
                        if(terminal||depth+1==longH)row["long"]=value;
                        if(row["short"].is_null())row["short"]=value;
                    }
                    if(terminal)break;guard(view,metadata);
                    if(depth+1<longH&&!arm&&edge){node=frozen->child(edge,key,actions(*world.battle,true));
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    }else node=nullptr;
                }
                if(detail&&sample<2)audit["traces"].push_back({{"sample",sample},{"action",action},{"arm",row["arm"]},{"path",path}});
                audit["rows"].push_back(row);audit["pending"]=nullptr;
            }
            audit["batches"].push_back(batch->diagnostics());batch.reset();
        }
        if(before!=frozen->diagnostic())throw std::runtime_error("continuation_audit_frozen_mutation");
        audit["snapshots_unchanged"]=true;audit["status"]="complete";
    }catch(const std::exception& e){audit["error"]=e.what();if(batch)audit["batches"].push_back(batch->diagnostics());}
    audit["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    result["continuation_audit"]=std::move(audit);return result;
}
} // namespace spire_search
