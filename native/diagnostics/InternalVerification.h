#pragma once
#include "PrefixAudit.h"

namespace spire_search {
inline constexpr auto internalVerificationVersion="bounded-internal-verification-v1";
inline J canonicalCandidates(J legal){
    if(!legal.is_array()||legal.empty())throw std::invalid_argument("verification_legal");
    std::sort(legal.begin(),legal.end(),[](const J& a,const J& b){return a.dump()<b.dump();});
    for(std::size_t i=1;i<legal.size();++i)if(legal[i]==legal[i-1])throw std::invalid_argument("verification_duplicate_action");
    return legal;
}

// Pure selector. Held-out assessment rows have no parameter or path into it.
inline J internalRecommendation(const J& candidates,const J& original,const J& rows,int allowance){
    if(std::find(candidates.begin(),candidates.end(),original)==candidates.end())throw std::invalid_argument("verification_original_missing");
    J out={{"action",original},{"reason","verified"},{"scores",J::array()},
        {"required_steps",candidates.size()*16*192},{"selection_worlds",16}};
    if(candidates.size()<2||candidates.size()>8){out["reason"]="ineligible_branching";return out;}
    if(allowance<int(candidates.size()*16*192)){out["reason"]="insufficient_budget";return out;}
    if(rows.size()!=candidates.size()*16)throw std::runtime_error("verification_incomplete_selection");
    std::vector<spire_objective::Key> keys;bool censored=false;
    for(const auto& a:candidates){std::vector<spire_objective::Outcome> values;std::array<bool,16> seen{};
        for(const auto& row:rows)if(row.at("action")==a){const int i=row.at("sample");
            if(i<0||i>=16||seen[i])throw std::runtime_error("verification_sample_inventory");seen[i]=true;
            auto value=spire_objective::Outcome::read(row.at("long"));censored|=value.terminal==0;values.push_back(value);}
        if(values.size()!=16)throw std::runtime_error("verification_candidate_inventory");
        const auto key=spire_objective::summarize(values);keys.push_back(key);
        out["scores"].push_back({{"action",a},{"key",key},{"samples",values.size()}});
    }
    if(censored){out["reason"]="censored_selection";return out;}
    auto best=std::max_element(keys.begin(),keys.end());const auto value=*best;
    out["action"]=candidates[std::size_t(best-keys.begin())];
    for(std::size_t i=0;i<candidates.size();++i)if(candidates[i]==original&&keys[i]==value)out["action"]=original;
    return out;
}

inline J verifyInternalNode(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                           const J& budget,const J& metadata,bool detail=false){
    const auto& spec=budget.at("frozen_audit");
    if(spec.value("version","")!=internalVerificationVersion)return auditPrefix(roots,seeds,budget,metadata,detail);
    const auto decisionStarted=Clock::now();
    if(spec.size()!=7||detail||!budget.at("max_seconds").is_null()||budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)
        throw std::invalid_argument("verification_contract");
    const auto& prefix=spec.at("prefix");const auto& decision=spec.at("decision");
    if(!spec.at("verify_steps").is_number_integer()||!spec.at("long_horizon").is_number_integer())throw std::invalid_argument("verification_integer_limits");
    const int allowance=spec.at("verify_steps"),horizon=spec.at("long_horizon");
    auto selection=spec.at("selection_seeds").get<std::vector<RootSeeds>>();
    auto assessment=spec.at("assessment_seeds").get<std::vector<RootSeeds>>();
    if(!prefix.is_array()||prefix.empty()||prefix.size()>4||decision.size()!=2||!decision.contains("public")||
       !decision.contains("legal")||allowance<0||allowance>24576||horizon!=192||selection.size()!=16||
       assessment.empty()||assessment.size()>64||spec.dump().size()>8*1024*1024)
        throw std::invalid_argument("verification_limits");
    for(const auto& p:prefix)if(p.size()!=4||!p.contains("public")||!p.contains("legal")||!p.contains("action")||!p.contains("observations"))
        throw std::invalid_argument("verification_prefix_contract");
    auto all=selection;all.insert(all.end(),assessment.begin(),assessment.end());
    for(std::size_t i=0;i<all.size();++i)if(std::find(seeds.begin(),seeds.end(),all[i])!=seeds.end()||
        std::find(all.begin(),all.begin()+i,all[i])!=all.begin()+i)throw std::invalid_argument("verification_seed_overlap");
    const auto candidates=canonicalCandidates(decision.at("legal"));
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));std::unique_ptr<spire_tree::FrozenTree> frozen;
    auto observer=[&](const spire_tree::StandaloneTree& tree,int,bool finish){
        if(finish)frozen=std::make_unique<spire_tree::FrozenTree>(limits,tree.root());};
    auto result=searchStandaloneTree(roots,seeds,budget,metadata,false,observer);
    if(!frozen||!result.at("refusal").is_null())throw std::runtime_error("verification_construction");
    const auto before=frozen->diagnostic();const auto& input=roots.evidence();
    const auto* target=frozen->root();
    for(std::size_t i=0;i<prefix.size();++i){auto* e=frozen->edge(target,prefix[i].at("action"));
        const auto& next=i+1<prefix.size()?prefix[i+1].at("legal"):decision.at("legal");
        target=frozen->child(e,spire_tree::observationKey(prefix[i].at("observations").get<std::vector<std::string>>(),limits.observationBytes),next);}
    if(!target)throw std::invalid_argument("verification_target_missing");
    const auto* ordinary=frozen->choose(target,decision.at("legal"));
    if(!ordinary)throw std::invalid_argument("verification_target_unvisited");
    const J original=J::parse(ordinary->action);const auto nodeStats=publicNodeStats(target);
    auto run=[&](const std::vector<RootSeeds>& worlds,const J& forced){
        J stage={{"rows",J::array()},{"batches",J::array()},{"steps",0},{"prefix_steps",0},
            {"samples",0},{"values",0},{"policies",0},{"step_cap",worlds.size()*forced.size()*192}};
        const auto started=Clock::now();
        for(std::size_t sample=0;sample<worlds.size()&&!forced.empty();++sample){
            auto batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
            auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("verification_deadline");};
            batch->prepare(0,worlds[sample],until);
            for(const auto& action:forced){
                checkTime();auto world=batch->branch(0);auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
                auto view=project();if(view!=input.policy)throw std::runtime_error("verification_root_projection");
                const auto* node=frozen->root();
                J row={{"sample",sample},{"action",action},{"steps",0},{"tree_steps",0},{"leaf_steps",0},
                    {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
                for(int depth=0;depth<192;++depth){
                    checkTime();const auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;J chosen;
                    if(depth<int(prefix.size())){const auto& p=prefix[depth];
                        if(view!=p.at("public")||legal!=p.at("legal"))throw std::runtime_error("verification_prefix_state");
                        chosen=p.at("action");edge=frozen->edge(node,chosen);stage["prefix_steps"]=stage["prefix_steps"].get<int>()+1;
                    }else if(depth==int(prefix.size())){
                        if(view!=decision.at("public")||legal!=decision.at("legal")||node!=target)throw std::runtime_error("verification_target_state");
                        chosen=action;edge=frozen->edge(node,chosen);
                    }else{
                        edge=frozen->choose(node,legal);chosen=edge?J::parse(edge->action):combatChoice(view,legal,metadata);
                        const char* field=edge?"tree_steps":"leaf_steps";row[field]=row[field].get<int>()+1;
                        if(!edge&&node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;
                        stage["policies"]=stage["policies"].get<int>()+1;
                    }
                    if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("verification_illegal");
                    const auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                    const int terminal=outcome(*world.battle);row["steps"]=depth+1;stage["steps"]=stage["steps"].get<int>()+1;
                    if(depth<int(prefix.size())&&(observed!=prefix[depth].at("observations").get<std::vector<std::string>>()||terminal))
                        throw std::runtime_error("verification_prefix_observations");
                    if(terminal||depth+1==48||depth+1==192){const auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal));
                        stage["values"]=stage["values"].get<int>()+1;if(row["short"].is_null())row["short"]=value;
                        if(terminal||depth+1==192)row["long"]=value;}
                    if(terminal)break;
                    guard(view,metadata);
                    if(depth+1<192&&edge){node=frozen->child(edge,spire_tree::observationKey(observed,limits.observationBytes),actions(*world.battle,true));
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    }else node=nullptr;
                }
                stage["rows"].push_back(row);stage["samples"]=stage["samples"].get<int>()+1;
            }stage["batches"].push_back(batch->diagnostics());
        }
        stage["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();return stage;
    };
    const bool eligible=candidates.size()>=2&&candidates.size()<=8&&allowance>=int(candidates.size()*16*192);
    const auto selectionResult=run(selection,eligible?candidates:J::array());
    const auto choice=internalRecommendation(candidates,original,selectionResult.at("rows"),allowance);
    const J selected=choice.at("action");J compared=J::array({original});if(selected!=original)compared.push_back(selected);
    compared=canonicalCandidates(compared);
    const double decisionSeconds=std::chrono::duration<double>(Clock::now()-decisionStarted).count();
    const auto assessmentResult=run(assessment,compared);
    if(before!=frozen->diagnostic())throw std::runtime_error("verification_snapshot_mutated");
    J work;for(auto key:{"steps","samples","values","policies"})work[key]=result.at(key).get<int>()+selectionResult.at(key).get<int>();
    result["internal_verification"]={{"version",internalVerificationVersion},{"status","complete"},{"candidates",candidates},
        {"original_action",original},{"recommendation",choice},{"selection",selectionResult},{"assessment",assessmentResult},
        {"decision_node",nodeStats},{"final_tree",before},{"snapshots_unchanged",true},{"prefix_exact",true},
        {"online_work",work},{"decision_seconds",decisionSeconds},{"verification_allowance",allowance},
        {"online_step_cap",budget.at("max_branch_steps").get<int>()+allowance}};
    return result;
}
} // namespace spire_search
