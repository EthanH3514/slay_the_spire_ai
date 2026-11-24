#pragma once
#include "InternalVerification.h"

namespace spire_search {
inline constexpr auto internalConfirmationVersion="bounded-internal-confirmation-v1";
inline constexpr auto continuationInterventionVersion="frozen-versus-rule-continuation-v1";
inline J screenRecommendation(const J& candidates,const J& original,const J& rows,int allowance){
    if(std::find(candidates.begin(),candidates.end(),original)==candidates.end())throw std::invalid_argument("verification_original_missing");
    J out={{"action",original},{"reason","verified"},{"scores",J::array()},
        {"required_steps",candidates.size()*16*192},{"selection_worlds",8}};
    if(candidates.size()<2||candidates.size()>8){out["reason"]="ineligible_branching";return out;}
    if(allowance<int(candidates.size()*16*192)){out["reason"]="insufficient_budget";return out;}
    if(rows.size()!=candidates.size()*8)throw std::runtime_error("verification_incomplete_selection");
    std::vector<spire_objective::Key> keys;bool censored=false;
    for(const auto& a:candidates){std::vector<spire_objective::Outcome> values;std::array<bool,8> seen{};
        for(const auto& row:rows)if(row.at("action")==a){const int i=row.at("sample");
            if(i<0||i>=8||seen[i])throw std::runtime_error("verification_sample_inventory");seen[i]=true;
            auto value=spire_objective::Outcome::read(row.at("long"));censored|=value.terminal==0;values.push_back(value);}
        if(values.size()!=8)throw std::runtime_error("verification_candidate_inventory");
        const auto key=spire_objective::summarize(values);keys.push_back(key);
        out["scores"].push_back({{"action",a},{"key",key},{"samples",values.size()}});
    }
    if(censored){out["reason"]="censored_selection";return out;}
    auto best=std::max_element(keys.begin(),keys.end());const auto value=*best;
    out["action"]=candidates[std::size_t(best-keys.begin())];
    for(std::size_t i=0;i<candidates.size();++i)if(candidates[i]==original&&keys[i]==value)out["action"]=original;
    return out;
}


// Fixed-sample, one-sided exact discordant-pair test. No held-out assessment input.
inline J confirmRecommendation(const J& original,const J& candidate,const J& rows,int pairs){
    if(pairs<8||pairs>32||rows.size()!=std::size_t(2*pairs)||candidate==original)
        throw std::invalid_argument("confirmation_inventory");
    int gains=0,losses=0;bool censored=false;
    for(int i=0;i<pairs;++i){
        const J* ref=nullptr;const J* alt=nullptr;
        for(const auto& r:rows)if(r.at("sample")==i){
            if(r.at("action")==original){if(ref)throw std::invalid_argument("confirmation_duplicate");ref=&r;}
            else if(r.at("action")==candidate){if(alt)throw std::invalid_argument("confirmation_duplicate");alt=&r;}
        }
        if(!ref||!alt)throw std::invalid_argument("confirmation_pair_missing");
        const auto r=spire_objective::Outcome::read(ref->at("long")),a=spire_objective::Outcome::read(alt->at("long"));
        censored|=r.terminal==0||a.terminal==0;
        gains+=r.terminal==2&&a.terminal!=2;losses+=r.terminal!=2&&a.terminal==2;
    }
    const int n=gains+losses;std::uint64_t choose=1,numerator=0,denominator=std::uint64_t(1)<<n;
    for(int j=0;j<=n;++j){if(j>=gains)numerator+=choose;if(j<n)choose=choose*(n-j)/(j+1);}
    const bool approve=!censored&&gains>losses&&20*numerator<=denominator;
    return {{"action",approve?candidate:original},{"candidate",candidate},{"reason",censored?"censored_confirmation":approve?"confirmed_survival":"unconfirmed_survival"},
        {"pairs",pairs},{"avoided_deaths",gains},{"introduced_deaths",losses},{"tail_numerator",numerator},
        {"tail_denominator",denominator},{"p_one_sided",double(numerator)/double(denominator)}};
}

inline J confirmInternalNode(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                             const J& budget,const J& metadata,bool detail=false){
    const auto& spec=budget.at("frozen_audit");const auto version=spec.value("version","");
    const bool intervention=version==continuationInterventionVersion;
    if(version!=internalConfirmationVersion&&!intervention)return verifyInternalNode(roots,seeds,budget,metadata,detail);
    const auto decisionStarted=Clock::now();
    if(spec.size()!=(intervention?7:8)||detail||!budget.at("max_seconds").is_null()||budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)
        throw std::invalid_argument("confirmation_contract");
    const auto& prefix=spec.at("prefix");const auto& decision=spec.at("decision");
    if(!spec.at("long_horizon").is_number_integer()||spec.at("long_horizon")!=192||
       !prefix.is_array()||prefix.empty()||prefix.size()>4||decision.size()!=2||!decision.contains("public")||
       !decision.contains("legal")||spec.dump().size()>8*1024*1024)throw std::invalid_argument("confirmation_limits");
    for(const auto& p:prefix)if(p.size()!=4||!p.contains("public")||!p.contains("legal")||!p.contains("action")||!p.contains("observations"))
        throw std::invalid_argument("confirmation_prefix_contract");
    const auto candidates=canonicalCandidates(decision.at("legal"));
    std::vector<RootSeeds> selection,confirmation,assessment;int allowance=0;J forced=J::array();bool rule=false;
    if(intervention){
        selection=spec.at("seeds").get<std::vector<RootSeeds>>();forced=canonicalCandidates(spec.at("actions"));
        const auto mode=spec.at("continuation").get<std::string>();
        if((mode!="frozen"&&mode!="rule")||selection.empty()||selection.size()>64||forced.size()>8)
            throw std::invalid_argument("intervention_limits");
        rule=mode=="rule";for(const auto& a:forced)if(std::find(candidates.begin(),candidates.end(),a)==candidates.end())
            throw std::invalid_argument("intervention_illegal");
    }else{
        if(!spec.at("verify_steps").is_number_integer())throw std::invalid_argument("confirmation_integer");
        allowance=spec.at("verify_steps");selection=spec.at("selection_seeds").get<std::vector<RootSeeds>>();
        confirmation=spec.at("confirmation_seeds").get<std::vector<RootSeeds>>();assessment=spec.at("assessment_seeds").get<std::vector<RootSeeds>>();
        if(allowance<0||allowance>24576||selection.size()!=8||confirmation.size()!=4*std::min(std::size_t(8),candidates.size())||
           assessment.empty()||assessment.size()>64)throw std::invalid_argument("confirmation_limits");
    }
    auto all=selection;all.insert(all.end(),confirmation.begin(),confirmation.end());all.insert(all.end(),assessment.begin(),assessment.end());
    for(std::size_t i=0;i<all.size();++i)if(std::find(seeds.begin(),seeds.end(),all[i])!=seeds.end()||
        std::find(all.begin(),all.begin()+i,all[i])!=all.begin()+i)throw std::invalid_argument("confirmation_seed_overlap");
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
    auto run=[&](const std::vector<RootSeeds>& worlds,const J& forced,bool rule=false,int traceDepth=0){
        J stage={{"rows",J::array()},{"batches",J::array()},{"steps",0},{"prefix_steps",0},
            {"samples",0},{"values",0},{"policies",0},{"step_cap",worlds.size()*forced.size()*192}};
        if(traceDepth)stage["paths"]=J::array();
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
                J path=J::array();
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
                    J trace;if(depth<traceDepth)trace={{"depth",depth},{"public",view},{"legal",legal},{"action",chosen},
                        {"mode",depth<int(prefix.size())?"forced_prefix":depth==int(prefix.size())?"forced_action":edge?"visited_tree":node?"unvisited_frontier":"rule_leaf"}};
                    const auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                    const int terminal=outcome(*world.battle);row["steps"]=depth+1;stage["steps"]=stage["steps"].get<int>()+1;
                    if(depth<int(prefix.size())&&(observed!=prefix[depth].at("observations").get<std::vector<std::string>>()||terminal))
                        throw std::runtime_error("verification_prefix_observations");
                    if(terminal||depth+1==48||depth+1==192){const auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal));
                        stage["values"]=stage["values"].get<int>()+1;if(row["short"].is_null())row["short"]=value;
                        if(terminal||depth+1==192)row["long"]=value;}
                    if(depth<traceDepth){trace["observations"]=observed;trace["terminal"]=terminal;path.push_back(trace);}
                    if(terminal)break;
                    guard(view,metadata);
                    if(rule&&depth>=int(prefix.size()))node=nullptr;
                    else if(depth+1<192&&edge){node=frozen->child(edge,spire_tree::observationKey(observed,limits.observationBytes),actions(*world.battle,true));
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    }else node=nullptr;
                }
                if(traceDepth)stage["paths"].push_back({{"sample",sample},{"action",action},{"path",path}});
                stage["rows"].push_back(row);stage["samples"]=stage["samples"].get<int>()+1;
            }stage["batches"].push_back(batch->diagnostics());
        }
        stage["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();return stage;
    };

    if(intervention){
        const auto stage=run(selection,forced,rule,8);
        if(before!=frozen->diagnostic())throw std::runtime_error("confirmation_snapshot_mutated");
        result["continuation_intervention"]={{"version",version},{"status","complete"},{"continuation",rule?"rule":"frozen"},
            {"stage",stage},{"original_action",original},{"decision_node",nodeStats},{"final_tree",before},
            {"snapshots_unchanged",true},{"prefix_exact",true}};
        if(result.dump().size()>64*1024*1024)throw std::runtime_error("intervention_output_limit");
        return result;
    }
    const bool eligible=candidates.size()>=2&&candidates.size()<=8&&allowance>=int(candidates.size()*16*192);
    const auto screen=run(selection,eligible?candidates:J::array());
    const auto screened=screenRecommendation(candidates,original,screen.at("rows"),allowance);
    const J candidate=screened.at("action");const bool test=screened.at("reason")=="verified"&&candidate!=original;
    const auto confirm=run(confirmation,test?canonicalCandidates(J::array({original,candidate})):J::array());
    const J recommendation=test?confirmRecommendation(original,candidate,confirm.at("rows"),int(confirmation.size())):
        J{{"action",original},{"candidate",candidate},{"reason",screened.at("reason")=="verified"?J("screen_retained_original"):screened.at("reason")}};
    const double decisionSeconds=std::chrono::duration<double>(Clock::now()-decisionStarted).count();
    // Diagnostic assessment only AFTER the final native recommendation; never an input to selection.
    const auto assessed=run(assessment,candidates.size()<=8?candidates:J::array({original}));
    if(before!=frozen->diagnostic())throw std::runtime_error("confirmation_snapshot_mutated");
    J work;for(auto key:{"steps","samples","values","policies"})work[key]=result.at(key).get<int>()+screen.at(key).get<int>()+confirm.at(key).get<int>();
    result["internal_confirmation"]={{"version",version},{"status","complete"},{"candidates",candidates},{"original_action",original},
        {"screening",screened},{"recommendation",recommendation},{"selection",screen},{"confirmation",confirm},{"assessment",assessed},
        {"decision_node",nodeStats},{"final_tree",before},{"snapshots_unchanged",true},{"prefix_exact",true},
        {"online_work",work},{"decision_seconds",decisionSeconds},{"verification_allowance",allowance},
        {"online_step_cap",budget.at("max_branch_steps").get<int>()+allowance}};
    return result;
}
} // namespace spire_search
