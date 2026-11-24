#pragma once
#include "LocalActionAudit.h"

namespace spire_search {
inline constexpr auto prefixAuditVersion="public-prefix-ledger-audit-v1";

// Read-only observer of existing completed simulations. Only public outcomes
// from the just-completed root visit are associated with a target-node visit.
class PrefixLedger {
    J prefix_,rows_=J::array();spire_tree::Limits limits_;
    std::map<std::string,unsigned> rootVisits_,visits_;
    unsigned lastCompleted_=0;
    const spire_tree::Node* target(const spire_tree::Node* node)const{
        for(const auto& p:prefix_){
            if(!node)return nullptr;
            const spire_tree::Edge* edge=nullptr;const auto action=p.at("action").dump();
            for(auto* e=node->edges;e;e=e->next)if(e->action==action){edge=e;break;}
            if(!edge)return nullptr;
            const auto key=spire_tree::observationKey(p.at("observations").get<std::vector<std::string>>(),limits_.observationBytes);
            node=nullptr;for(auto* c=edge->children;c;c=c->sibling)if(c->observation==key){node=c;break;}
        }return node;
    }
public:
    PrefixLedger(J prefix,spire_tree::Limits limits):prefix_(std::move(prefix)),limits_(limits){}
    void observe(const spire_tree::StandaloneTree& tree,int steps,bool finish){
        if(finish){if(lastCompleted_!=tree.completed())throw std::runtime_error("prefix_ledger_finish");return;}
        if(tree.completed()!=lastCompleted_+1)throw std::runtime_error("prefix_ledger_schedule");
        lastCompleted_=tree.completed();std::string rootAction;unsigned changedRoots=0;
        for(auto* e=tree.root()->edges;e;e=e->next){auto key=std::string(e->action);auto& old=rootVisits_[key];
            if(e->visits!=old){if(e->visits!=old+1)throw std::runtime_error("prefix_ledger_root_jump");
                ++changedRoots;rootAction=key;old=e->visits;}}
        if(changedRoots!=1)throw std::runtime_error("prefix_ledger_root_count");
        const auto* node=target(tree.root());if(!node)return;
        unsigned changes=0;
        for(auto* e=node->edges;e;e=e->next){auto action=std::string(e->action);auto& old=visits_[action];
            if(e->visits==old)continue;
            if(e->visits!=old+1||++changes>1)throw std::runtime_error("prefix_ledger_edge_jump");
            old=e->visits;J outcome;
            for(const auto& r:tree.evidence())if(r.at("action").dump()==rootAction)outcome=r.at("outcomes").back();
            if(outcome.is_null())throw std::runtime_error("prefix_ledger_missing_outcome");
            rows_.push_back({{"simulation",tree.completed()-1},{"construction_steps",steps},
                {"action",J::parse(action)},{"outcome",outcome},
                {"utility",std::clamp((outcome.at("value").get<double>()+3.)/6.,0.,1.)}});
        }
        if(rows_.size()>4096)throw std::runtime_error("prefix_ledger_limit");
    }
    J result(const spire_tree::StandaloneTree& tree)const{
        const auto* node=target(tree.root());
        if(node)for(auto* e=node->edges;e;e=e->next){double sum=0;unsigned count=0;
            for(const auto& r:rows_)if(r.at("action").dump()==e->action){sum+=r.at("utility").get<double>();++count;}
            if(count!=e->visits||sum!=e->sum)throw std::runtime_error("prefix_ledger_reconciliation");}
        return {{"node",publicNodeStats(node)},{"rows",rows_},{"completed_simulations",lastCompleted_}};
    }
};

inline J auditPrefix(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                     const J& budget,const J& metadata,bool detail=false){
    const auto& spec=budget.at("frozen_audit");
    if(spec.value("version","")!=prefixAuditVersion)return auditLocalAction(roots,seeds,budget,metadata,detail);
    if(spec.size()!=8||detail||!budget.at("max_seconds").is_null()||budget.value("tree_algorithm","")!=spire_tree::standaloneVersion)
        throw std::invalid_argument("prefix_audit_contract");
    const auto& prefix=spec.at("prefix");const auto& decision=spec.at("decision");
    auto candidates=spec.at("actions");const bool record=spec.at("record_paths").get<bool>();
    const int traceDepth=spec.at("trace_depth"),longH=spec.at("long_horizon");
    const auto eval=spec.at("seeds").get<std::vector<RootSeeds>>();
    if(!prefix.is_array()||prefix.empty()||prefix.size()>4||!candidates.is_array()||
       (!candidates.empty()&&candidates.size()!=2)||traceDepth!=16||longH!=192||
       eval.empty()||eval.size()>64||decision.size()!=2||!decision.contains("public")||!decision.contains("legal")||spec.dump().size()>8*1024*1024)
        throw std::invalid_argument("prefix_audit_limits");
    for(const auto& p:prefix)if(p.size()!=4||!p.contains("public")||!p.contains("legal")||!p.contains("action")||!p.contains("observations"))
        throw std::invalid_argument("prefix_audit_step");
    std::sort(candidates.begin(),candidates.end());
    if(candidates.size()==2&&candidates[0]==candidates[1])throw std::invalid_argument("prefix_audit_duplicate");
    for(const auto& a:candidates)if(std::find(decision.at("legal").begin(),decision.at("legal").end(),a)==decision.at("legal").end())
        throw std::invalid_argument("prefix_audit_candidate");
    for(std::size_t i=0;i<eval.size();++i)if(std::find(seeds.begin(),seeds.end(),eval[i])!=seeds.end()||
        std::find(eval.begin(),eval.begin()+i,eval[i])!=eval.begin()+i)throw std::invalid_argument("prefix_audit_seed_overlap");
    const bool forced=!candidates.empty();if(!forced)candidates.push_back(nullptr);
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));PrefixLedger ledger(prefix,limits);
    std::unique_ptr<spire_tree::FrozenTree> frozen;J history;
    auto observer=[&](const spire_tree::StandaloneTree& t,int steps,bool finish){ledger.observe(t,steps,finish);
        if(finish){history=ledger.result(t);frozen=std::make_unique<spire_tree::FrozenTree>(limits,t.root());}};
    auto result=searchStandaloneTree(roots,seeds,budget,metadata,false,observer);
    if(!frozen||!result.at("refusal").is_null())throw std::runtime_error("prefix_audit_construction");
    const auto before=frozen->diagnostic();const auto& input=roots.evidence();
    J audit={{"version",prefixAuditVersion},{"status","incomplete"},{"rows",J::array()},{"paths",J::array()},
        {"batches",J::array()},{"history",history},{"final_tree",before},{"forced",forced},{"prefix_length",prefix.size()},
        {"trace_depth",16},{"steps",0},{"prefix_steps",0},{"policies",0},{"values",0},{"samples",0},
        {"step_cap",eval.size()*candidates.size()*longH},{"trace_bytes",0}};
    std::size_t bytes=0;const auto started=Clock::now();
    for(std::size_t sample=0;sample<eval.size();++sample){
        auto batch=roots.diagnosticBatch();const auto until=Clock::now()+std::chrono::seconds(30);
        auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("prefix_audit_deadline");};
        batch->prepare(0,eval[sample],until);
        for(const auto& candidate:candidates){
            checkTime();auto world=batch->branch(0);
            auto project=[&]{return batch->projectView(*world.battle,input.relics,world.knownTop,input.character);};
            auto view=project();if(view!=input.policy)throw std::runtime_error("prefix_audit_root_projection");
            const auto* node=frozen->root();std::string reason="root";J path=J::array();
            J row={{"sample",sample},{"action",candidate},{"steps",0},{"tree_steps",0},{"leaf_steps",0},
                {"unknown_history",0},{"unvisited_frontier",0},{"short",nullptr},{"long",nullptr}};
            for(int depth=0;depth<longH;++depth){
                checkTime();const auto legal=actions(*world.battle,true);const spire_tree::Edge* edge=nullptr;
                J chosen,prior=nullptr;std::string mode;
                if(depth<int(prefix.size())){
                    const auto& p=prefix[depth];if(view!=p.at("public")||legal!=p.at("legal"))throw std::runtime_error("prefix_audit_prefix_state");
                    chosen=p.at("action");edge=frozen->edge(node,chosen);mode="forced_prefix";
                    audit["prefix_steps"]=audit["prefix_steps"].get<int>()+1;
                }else{
                    if(depth==int(prefix.size())&&(view!=decision.at("public")||legal!=decision.at("legal")||publicNodeStats(node)!=history.at("node")))
                        throw std::runtime_error("prefix_audit_decision");
                    if(forced&&depth==int(prefix.size())){chosen=candidate;edge=frozen->edge(node,chosen);mode="forced_action";}
                    else {prior=combatChoice(view,legal,metadata);edge=frozen->choose(node,legal);chosen=edge?J::parse(edge->action):prior;
                        mode=edge?"visited_tree":node?"unvisited_frontier":"rule_after_exit";
                        const char* field=edge?"tree_steps":"leaf_steps";row[field]=row[field].get<int>()+1;
                        if(!edge&&node)row["unvisited_frontier"]=row["unvisited_frontier"].get<int>()+1;
                        audit["policies"]=audit["policies"].get<int>()+1;}
                }
                if(std::find(legal.begin(),legal.end(),chosen)==legal.end())throw std::runtime_error("prefix_audit_illegal");
                J recordRow;if(record&&depth<traceDepth)recordRow={{"depth",depth},{"public",view},{"legal",legal},{"action",chosen},
                    {"rule",prior},{"mode",mode},{"entry",reason},{"node",publicNodeStats(node)}};
                const auto observed=batch->advancePublic(world,chosen);view=project();checkTime();
                const int terminal=outcome(*world.battle);row["steps"]=depth+1;audit["steps"]=audit["steps"].get<int>()+1;
                if(depth<int(prefix.size())&&(observed!=prefix[depth].at("observations").get<std::vector<std::string>>()||terminal))
                    throw std::runtime_error("prefix_audit_observations");
                if(terminal||depth+1==48||depth+1==longH){auto value=spire_objective::record(view,input.policy,terminal,combatValue(view,terminal));
                    audit["values"]=audit["values"].get<int>()+1;if(row["short"].is_null())row["short"]=value;
                    if(terminal||depth+1==longH)row["long"]=value;}
                if(!terminal){guard(view,metadata);
                    if(depth+1<longH&&edge){node=frozen->child(edge,spire_tree::observationKey(observed,limits.observationBytes),actions(*world.battle,true));
                        reason=node?"known_child":"unknown_history";
                        if(!node)row["unknown_history"]=row["unknown_history"].get<int>()+1;
                    }else {if(node)reason="unvisited_frontier";node=nullptr;}}
                if(record&&depth<traceDepth){recordRow["observations"]=observed;recordRow["terminal"]=terminal;recordRow["successor"]=terminal?"terminal":reason;
                    bytes+=recordRow.dump().size();if(bytes>64*1024*1024)throw std::runtime_error("prefix_audit_trace_limit");path.push_back(std::move(recordRow));}
                if(terminal)break;
            }
            audit["rows"].push_back(row);audit["samples"]=audit["samples"].get<int>()+1;
            if(record)audit["paths"].push_back({{"sample",sample},{"action",candidate},{"path",path}});
        }audit["batches"].push_back(batch->diagnostics());
    }
    if(before!=frozen->diagnostic())throw std::runtime_error("prefix_audit_snapshot_mutated");
    audit["status"]="complete";audit["snapshots_unchanged"]=true;audit["prefix_exact"]=true;
    audit["trace_bytes"]=bytes;audit["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    result["prefix_audit"]=std::move(audit);return result;
}
} // namespace spire_search
