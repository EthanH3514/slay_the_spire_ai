#pragma once
#include "InternalConfirmation.h"
#include <set>

namespace spire_search {
inline constexpr auto allocationVersion="bounded-confirmation-allocation-v1";

// Fixed 192-bit unsigned arithmetic; max intermediate is20*2^128.
struct TailInteger {
    std::array<std::uint64_t,3> words{};
    static TailInteger one(){TailInteger x;x.words[0]=1;return x;}
    void add(const TailInteger& b){
        std::uint64_t carry=0;
        for(int i=0;i<3;++i){auto first=words[i]+b.words[i];const bool c1=first<words[i];
            auto second=first+carry;const bool c2=second<first;words[i]=second;carry=c1||c2;}
        if(carry)throw std::overflow_error("tail_integer_overflow");
    }
    bool le(const TailInteger& b)const{for(int i=2;i>=0;--i)if(words[i]!=b.words[i])return words[i]<b.words[i];return true;}
    std::string hex()const{
        std::string s;bool started=false;for(int i=2;i>=0;--i)for(int n=15;n>=0;--n){
            const int digit=int((words[i]>>(4*n))&15);if(digit||started){s.push_back("0123456789abcdef"[digit]);started=true;}}
        return s.empty()?"0":s;
    }
    double value()const{double x=0;for(int i=2;i>=0;--i)x=std::ldexp(x,64)+double(words[i]);return x;}
};
inline J exactPairedTail(int gains,int losses){
    const int n=gains+losses;if(gains<0||losses<0||n>128)throw std::invalid_argument("tail_bounds");
    std::array<TailInteger,129> binomial{};binomial[0]=TailInteger::one();
    for(int i=1;i<=n;++i)for(int j=i;j>0;--j)binomial[j].add(binomial[j-1]);
    TailInteger numerator,denominator,scaled;denominator.words[n/64]=std::uint64_t(1)<<(n%64);
    for(int j=gains;j<=n;++j)numerator.add(binomial[j]);
    for(int i=0;i<20;++i)scaled.add(numerator);
    return {{"numerator_hex",numerator.hex()},{"denominator_hex",denominator.hex()},
        {"p",numerator.value()/denominator.value()},{"passes",scaled.le(denominator)}};
}
inline J allocatedConfirmation(const J& original,const J& candidate,const J& rows,int pairs){
    if(pairs<8||pairs>128||rows.size()!=std::size_t(2*pairs)||candidate==original)
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

    const auto tail=exactPairedTail(gains,losses);const bool approve=!censored&&gains>losses&&tail.at("passes").get<bool>();
    return {{"action",approve?candidate:original},{"candidate",candidate},{"reason",censored?"censored_confirmation":approve?"confirmed_survival":"unconfirmed_survival"},
        {"pairs",pairs},{"avoided_deaths",gains},{"introduced_deaths",losses},{"tail_numerator_hex",tail.at("numerator_hex")},
        {"tail_denominator_hex",tail.at("denominator_hex")},{"p_one_sided",tail.at("p")}};
}

inline J allocatedRank(const J& candidates,const J& original,const J& stage){
    const int n=stage.at("complete_worlds");const auto& rows=stage.at("rows");
    J out={{"action",original},{"reason","no_complete_round"},{"scores",J::array()},{"complete_worlds",n},
        {"ignored_paths",int(rows.size())-n*int(candidates.size())}};
    if(n==0)return out;
    std::vector<spire_objective::Key> keys;bool ongoing=false;
    for(std::size_t a=0;a<candidates.size();++a){
        std::vector<spire_objective::Outcome> values;
        for(int i=0;i<n;++i){const auto& row=rows.at(i*candidates.size()+a);
            if(row.at("sample")!=i||row.at("action")!=candidates[a])throw std::invalid_argument("allocated_rank_inventory");
            auto value=spire_objective::Outcome::read(row.at("long"));ongoing|=value.terminal==0;values.push_back(value);}
        auto key=spire_objective::summarize(values);keys.push_back(key);
        out["scores"].push_back({{"action",candidates[a]},{"key",key},{"samples",n}});
    }
    if(ongoing){out["reason"]="censored_selection";return out;}
    const auto best=*std::max_element(keys.begin(),keys.end());out["reason"]="ranked";
    for(std::size_t i=0;i<candidates.size();++i)if(keys[i]==best){out["action"]=candidates[i];break;}
    for(std::size_t i=0;i<candidates.size();++i)if(candidates[i]==original&&keys[i]==best)out["action"]=original;
    return out;
}
inline J allocateInternalNode(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                             const J& budget,const J& metadata,bool detail=false){
    const auto& spec=budget.at("frozen_audit");
    if(spec.value("version","")!=allocationVersion)return confirmInternalNode(roots,seeds,budget,metadata,detail);
    const auto decisionStarted=Clock::now();
    if(spec.size()!=10||detail||!budget.at("max_seconds").is_null()||budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)
        throw std::invalid_argument("allocation_contract");
    const auto& prefix=spec.at("prefix");const auto& decision=spec.at("decision");
    if(!spec.at("long_horizon").is_number_integer()||spec.at("long_horizon")!=192||
       !prefix.is_array()||prefix.empty()||prefix.size()>4||decision.size()!=2||!decision.contains("public")||
       !decision.contains("legal")||spec.dump().size()>8*1024*1024)throw std::invalid_argument("allocation_limits");
    for(const auto& p:prefix)if(p.size()!=4||!p.contains("public")||!p.contains("legal")||!p.contains("action")||!p.contains("observations"))
        throw std::invalid_argument("allocation_prefix");
    if(!spec.at("verify_steps").is_number_integer()||!spec.at("confirmation_pairs").is_number_integer())throw std::invalid_argument("allocation_integer");
    const int allowance=spec.at("verify_steps"),pairs=spec.at("confirmation_pairs");const auto mode=spec.at("method").get<std::string>();
    const bool empirical=mode=="empirical";const auto candidates=canonicalCandidates(decision.at("legal"));
    auto selection=spec.at("selection_seeds").get<std::vector<RootSeeds>>();
    auto confirmation=spec.at("confirmation_seeds").get<std::vector<RootSeeds>>();
    auto assessment=spec.at("assessment_seeds").get<std::vector<RootSeeds>>();
    if((mode!="confirmation"&&!empirical)||allowance<0||allowance>24576||candidates.size()<2||candidates.size()>8||
       assessment.empty()||assessment.size()>64||selection.empty()||selection.size()>6145||
       (!empirical&&(selection.size()!=8||(pairs!=64&&pairs!=128)||confirmation.size()!=std::size_t(pairs)))||
       (empirical&&(pairs!=0||!confirmation.empty())))throw std::invalid_argument("allocation_limits");
    std::set<RootSeeds> unique(seeds.begin(),seeds.end());
    for(const auto* group:{&selection,&confirmation,&assessment})for(const auto& seed:*group)
        if(!unique.insert(seed).second)throw std::invalid_argument("allocation_seed_overlap");
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
    auto run=[&](const std::vector<RootSeeds>& worlds,const J& forced,int& remaining){
        J stage={{"rows",J::array()},{"batches",J::array()},{"steps",0},{"prefix_steps",0},
            {"samples",0},{"values",0},{"policies",0},{"step_cap",remaining},{"requested_worlds",worlds.size()},{"actions",forced}};
        const auto started=Clock::now();
        for(std::size_t sample=0;sample<worlds.size()&&!forced.empty()&&remaining>0;++sample){
            auto batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
            auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("verification_deadline");};
            batch->prepare(0,worlds[sample],until);
            for(const auto& action:forced){
                if(remaining<=0)break;
                checkTime();auto world=batch->branch(0);auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
                auto view=project();if(view!=input.policy)throw std::runtime_error("verification_root_projection");
                const auto* node=frozen->root();
                J row={{"sample",sample},{"action",action},{"steps",0},{"tree_steps",0},{"leaf_steps",0},
                    {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
                for(int depth=0;depth<192&&remaining>0;++depth){
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
                    const auto observed=batch->advancePublic(world,chosen);view=project();--remaining;checkTime();
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
        int full=0;if(!forced.empty())for(std::size_t start=0;start+forced.size()<=stage["rows"].size();start+=forced.size()){
            bool complete=true;for(std::size_t j=0;j<forced.size();++j)complete&=!stage["rows"][start+j]["long"].is_null();
            if(!complete)break;++full;
        }
        stage["complete_worlds"]=full;stage["complete"]=forced.empty()||full==int(worlds.size());
        stage["budget_exhausted"]=remaining==0&&!stage["complete"].get<bool>();
        stage["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();return stage;
    };

    int remaining=allowance;const auto screen=run(selection,candidates,remaining);
    const auto screened=allocatedRank(candidates,original,screen);
    J confirm=run(confirmation,J::array(),remaining),recommendation;
    if(empirical)recommendation=screened;
    else if(!screen.at("complete").get<bool>())recommendation={{"action",original},{"candidate",original},{"reason","screen_budget_exhausted"}};
    else if(screened.at("reason")!="ranked"||screened.at("action")==original)
        recommendation={{"action",original},{"candidate",screened.at("action")},{"reason",screened.at("reason")=="ranked"?J("screen_retained_original"):screened.at("reason")}};
    else{
        const J candidate=screened.at("action");
        confirm=run(confirmation,canonicalCandidates(J::array({original,candidate})),remaining);
        recommendation=confirm.at("complete").get<bool>()?allocatedConfirmation(original,candidate,confirm.at("rows"),pairs):
            J{{"action",original},{"candidate",candidate},{"reason","confirmation_budget_exhausted"}};
    }
    const double decisionSeconds=std::chrono::duration<double>(Clock::now()-decisionStarted).count();
    int offline=int(assessment.size()*candidates.size()*192);const auto assessed=run(assessment,candidates,offline);
    if(before!=frozen->diagnostic())throw std::runtime_error("allocation_snapshot_mutated");
    J work;for(auto key:{"steps","samples","values","policies"})work[key]=result.at(key).get<int>()+screen.at(key).get<int>()+confirm.at(key).get<int>();
    result["allocation"]={{"version",allocationVersion},{"method",mode},{"status","complete"},{"candidates",candidates},{"original_action",original},
        {"screening",screened},{"recommendation",recommendation},{"selection",screen},{"confirmation",confirm},{"assessment",assessed},
        {"decision_node",nodeStats},{"final_tree",before},{"snapshots_unchanged",true},{"prefix_exact",true},{"confirmation_pairs",pairs},
        {"online_work",work},{"decision_seconds",decisionSeconds},{"verification_allowance",allowance},{"remaining_steps",remaining},
        {"online_step_cap",budget.at("max_branch_steps").get<int>()+allowance}};
    if(result.dump().size()>64*1024*1024)throw std::runtime_error("allocation_output_limit");
    return result;
}
} // namespace spire_search
