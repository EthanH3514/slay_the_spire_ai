#pragma once
// Opt-in offline teacher only. No tree construction, recommendation or fallback.
#include "ConditionalRoots.h"
#include "CombatObjective.h"

namespace spire_teacher {
using namespace spire_search;
inline constexpr auto version="public-paired-rule-teacher-v1";

inline J collect(ConditionalRoots& roots,const J& budget,const J& metadata,const J& spec) {
    const auto mode=spec.value("mode",std::string());
    const int count=mode=="pilot"?32:mode=="engineering"?2:0;
    const auto order=spec.value("order",std::string());
    if(spec.size()!=7||spec.at("version")!=version||!count||
       (order!="canonical"&&order!="reverse")||spec.at("checkpoint")!=192||spec.at("horizon")!=768||
       spire_objective::name(budget)!=spire_objective::terminalVersion)
        throw std::invalid_argument("teacher_contract");
    const double seconds=spec.at("max_seconds");
    if(!std::isfinite(seconds)||seconds<=0||seconds>295)throw std::invalid_argument("teacher_deadline");
    const auto seeds=spec.at("seeds").get<std::vector<std::vector<RootSeeds>>>();
    if(seeds.size()!=2||seeds[0].size()!=std::size_t(count)||seeds[1].size()!=std::size_t(count))
        throw std::invalid_argument("teacher_world_count");
    std::vector<RootSeeds> seen;
    for(const auto& bank:seeds)for(const auto& seed:bank){
        if(std::find(seen.begin(),seen.end(),seed)!=seen.end())throw std::invalid_argument("teacher_duplicate_world");
        seen.push_back(seed);
    }
    const auto started=Clock::now();
    const auto until=started+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("teacher_deadline");};
    const auto& input=roots.evidence();const auto& legal=input.candidates;
    J out={{"version",version},{"mode",mode},{"status","incomplete"},{"error",nullptr},
        {"objective",spire_objective::terminalVersion},{"origin","public_posterior_fixed_rule"},
        {"banks",J::array()},{"sampling",J::array()},{"pending",nullptr},{"rule_action",nullptr},
        {"attempted_advances",0},{"completed_advances",0},{"policies",0},{"completed_branches",0},
        {"step_cap",2*count*legal.size()*768},{"checkpoint",192},{"horizon",768},
        {"construction_budget_scope","per_world_including_all_candidate_rebuilds"},
        {"order",order},{"training_label_eligible",mode=="pilot"}};
    if(legal.empty()||!roots.sameActions(legal,legal))throw std::invalid_argument("teacher_actions");
    std::unique_ptr<ConditionalRoots> batch;
    int bankIndex=0,worldIndex=0;
    try {
        roots.validate(input.policy,legal,metadata);
        if(legal.size()>32)throw std::runtime_error("teacher_action_budget_gap");
        out["rule_action"]=combatChoice(input.policy,legal,metadata);
        if(legal.size()==1){out["status"]="forced";out["training_label_eligible"]=false;}
        else {
            for(int bank=0;bank<2;++bank){J rows=J::array();
                for(const auto& action:legal)rows.push_back({{"action",action},{"samples",J::array()}});
                out["banks"].push_back(rows);
            }
            std::vector<int> traversal;for(std::size_t i=0;i<legal.size();++i)traversal.push_back(int(i));
            std::sort(traversal.begin(),traversal.end(),[&](int a,int b){return legal[a]<legal[b];});
            if(order=="reverse")std::reverse(traversal.begin(),traversal.end());
            for(bankIndex=0;bankIndex<2;++bankIndex)for(worldIndex=0;worldIndex<count;++worldIndex){
                checkTime();batch=roots.diagnosticBatch();batch->prepare(0,seeds[bankIndex][worldIndex],until);
                for(int actionIndex:traversal){
                    checkTime();out["pending"]={{"bank",bankIndex},{"world",worldIndex},{"action",legal[actionIndex]},
                        {"steps",0},{"checkpoint",nullptr},{"result",nullptr}};
                    auto& row=out["pending"];auto world=batch->branch(0);
                    auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
                    auto view=project();
                    if(view!=input.policy||!batch->sameActions(actions(*world.battle,true),legal))
                        throw std::runtime_error("teacher_root_projection");
                    for(int depth=0;depth<768;++depth){
                        checkTime();auto candidates=actions(*world.battle,true);
                        J chosen=legal[actionIndex];
                        if(depth){out["policies"]=out["policies"].get<int>()+1;chosen=combatChoice(view,candidates,metadata);}
                        if(std::find(candidates.begin(),candidates.end(),chosen)==candidates.end())
                            throw std::runtime_error("teacher_illegal_continuation");
                        out["attempted_advances"]=out["attempted_advances"].get<int>()+1;
                        batch->advancePublic(world,chosen);
                        out["completed_advances"]=out["completed_advances"].get<int>()+1;row["steps"]=depth+1;
                        view=project();const int terminal=outcome(*world.battle);
                        if(terminal||depth+1==192||depth+1==768){
                            auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal),
                                spire_objective::terminalVersion);
                            if(row["checkpoint"].is_null())row["checkpoint"]=value;
                            if(terminal||depth+1==768)row["result"]=value;
                        }
                        if(terminal)break;
                        guard(view,metadata);
                    }
                    out["banks"][bankIndex][actionIndex]["samples"].push_back(row);
                    out["completed_branches"]=out["completed_branches"].get<int>()+1;out["pending"]=nullptr;
                }
                out["sampling"].push_back({{"bank",bankIndex},{"world",worldIndex},{"diagnostics",batch->diagnostics()}});
                batch.reset();
            }
            out["status"]="complete";
        }
    }catch(const std::exception& e){out["error"]=e.what();out["training_label_eligible"]=false;
        if(batch)out["sampling"].push_back({{"bank",bankIndex},{"world",worldIndex},{"diagnostics",batch->diagnostics()}});
    }
    out["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    // The enclosing Session adapter appends its aggregate offer/prefix counters.
    return {{"teacher",out},{"root_sampling",roots.diagnostics()}};
}
} // namespace spire_teacher
