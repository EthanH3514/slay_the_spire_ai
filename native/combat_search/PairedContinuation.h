#pragma once
#include "ConditionalRoots.h"
#include "HistoryTree.h"
#include "ObjectiveTree.h"
#include <array>

namespace spire_tree {
inline constexpr auto pairedVersion="public-history-tree-v3";
inline constexpr int verificationPairs=4, verificationProposals=64;

// A fixed-batch empirical screen, deliberately NOT a calibrated confidence bound.
struct PairScreen {
    unsigned count=0;double mean=0,m2=0;
    void add(double delta) {
        if(!std::isfinite(delta)||delta < -1.||delta > 1.)throw std::runtime_error("invalid_paired_value");
        ++count;double d=delta-mean;mean+=d/count;m2+=d*(delta-mean);
    }
    double se() const {return count<2?0.:std::sqrt(std::max(0.,m2)/(count-1)/count);}
    bool approve(unsigned required) const {return count==required&&mean-2*se()>1e-12;}
    J diagnostic(unsigned required) const {
        return {{"complete_pairs",count},{"required_pairs",required},{"mean_difference",count?J(mean):J()},
            {"standard_error",count>1?J(se()):J()},{"screen_margin",count>1?J(mean-2*se()):J()},
            {"approved",approve(required)},{"calibrated_confidence_bound",false}};
    }
};

struct PairedProposal {
    struct Step {J action;std::string key;};
    Node* node=nullptr;Edge* challenger=nullptr;Edge* reference=nullptr;
    std::vector<Step> prefix;
    J action,rule,support;
    J diagnostic(bool detail) const {
        if(action.is_null())return nullptr;
        J out={{"node",node?J(node->id):support.at("id")},{"depth",prefix.size()},
            {"action",action},{"rule",rule},{"support",support}};
        if(detail){out["prefix"]=J::array();for(const auto& step:prefix)
            out["prefix"].push_back({{"action",step.action},{"observation_key",step.key}});}
        return out;
    }
};

class PairedCandidates {
    struct Link {Node* node=nullptr;Node* parent=nullptr;Edge* via=nullptr;Edge* prior=nullptr;};
    std::vector<Link> links_;
public:
    explicit PairedCandidates(int cap):links_(cap) {}
    void link(Node* node,Node* parent,Edge* via) {
        if(!node)return;auto& item=links_.at(node->id);
        if(item.node&&(item.parent!=parent||item.via!=via))throw std::runtime_error("paired_history_link_changed");
        item.node=node;item.parent=parent;item.via=via;
    }
    void prior(Tree& tree,Node* node,const J& action) {
        if(!node)return;auto& item=links_.at(node->id);auto* edge=tree.edge(node,action);
        if(item.prior&&item.prior!=edge)throw std::runtime_error("paired_public_rule_changed");
        item.prior=edge;
    }
    std::size_t bytes() const {return links_.capacity()*sizeof(Link);}
    PairedProposal propose(Tree& tree,int horizon,unsigned& eligible,const ObjectiveStats* objective=nullptr,
                           unsigned minimum=minimumContinuationVisits) const {
        PairedProposal out;unsigned bestVisits=0;double bestAdvantage=0;eligible=0;
        for(const auto& link:links_) {
            auto* n=link.node;auto* ref=link.prior;
            if(!n||!n->depth||n->depth>=unsigned(horizon)||!ref)continue;
            auto* alt=objective?objective->select(tree,n,J::parse(ref->action),minimum):
                tree.selectGuarded(n,false,J::parse(ref->action));
            if(!alt||alt==ref)continue;
            ++eligible;unsigned visits=std::min(alt->visits,ref->visits);
            double advantage=alt->sum/alt->visits-ref->sum/ref->visits;
            if(objective){auto a=objective->key(alt),b=objective->key(ref);
                advantage=a[spire_objective::firstDifference(a,b)]-b[spire_objective::firstDifference(a,b)];}
            if(out.node&&(n->depth>out.node->depth||(n->depth==out.node->depth&&
                (visits<bestVisits||(visits==bestVisits&&(advantage<bestAdvantage||
                    (advantage==bestAdvantage&&n->id>out.node->id)))))))continue;
            out.node=n;out.challenger=alt;out.reference=ref;bestVisits=visits;bestAdvantage=advantage;
        }
        if(!out.node)return out;
        out.action=J::parse(out.challenger->action);out.rule=J::parse(out.reference->action);
        out.support=tree.decisionDiagnostic(out.node);
        for(auto* n=out.node;n->depth;) {
            const auto& link=links_.at(n->id);
            if(!link.parent||!link.via||out.prefix.size()>=64)throw std::runtime_error("paired_prefix_contract");
            out.prefix.push_back({J::parse(link.via->action),std::string(n->observation)});n=link.parent;
        }
        std::reverse(out.prefix.begin(),out.prefix.end());return out;
    }
};
} // namespace spire_tree

namespace spire_search {
struct PairedWork {
    int steps=0,samples=0,values=0,policies=0,prefixSteps=0,rejected=0,attempts=0;
    spire_tree::PairScreen screen;
    bool objective=false;
    spire_objective::PairScreen objectiveScreen;
    bool approved(unsigned required=spire_tree::verificationPairs) const {
        return objective?objectiveScreen.approve(required):screen.approve(required);
    }
    J pairs=J::array();std::string stop="no_proposal";
    J diagnostic(bool detail,int required=spire_tree::verificationPairs) const {
        auto out=objective?objectiveScreen.diagnostic(required):screen.diagnostic(required);
        out.update({{"stop",stop},{"steps",steps},{"samples",samples},{"values",values},{"policies",policies},
            {"prefix_steps",prefixSteps},{"rejected_prefixes",rejected},{"proposals",attempts}});
        if(detail)out["pairs"]=pairs;return out;
    }
};

// All worlds are rebuilt through the SAME owned public-history root factory.
// Rejection checks the complete public event prefix, never the future value.
inline void verifyContinuation(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,int seedOffset,
        int& prepared,const spire_tree::PairedProposal& proposal,int horizon,int stepCap,int sampleCap,int valueCap,
        const J& metadata,Clock::time_point until,PairedWork& work,int required=spire_tree::verificationPairs,
        std::size_t observationBytes=1024*1024) {
    if(proposal.action.is_null())return;
    const auto& input=roots.evidence();work.stop="proposal_limit";
    auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("tree_deadline");};
    auto advance=[&](SearchRoot& world,const J& a,bool prefix) {
        checkTime();++work.steps;work.prefixSteps+=prefix;
        auto observed=roots.advancePublic(world,a);checkTime();return observed;
    };
    auto replay=[&](SearchRoot& world) {
        for(const auto& step:proposal.prefix) {
            auto observed=advance(world,step.action,true);
            if(spire_tree::observationKey(observed,observationBytes)!=step.key)return false;
        }
        auto legal=actions(*world.battle,true);
        J expected=J::array();for(const auto& edge:proposal.support.at("actions"))expected.push_back(edge.at("action"));
        if(!roots.sameActions(legal,expected))throw std::runtime_error("paired_node_legal_set_changed");
        if(std::find(legal.begin(),legal.end(),proposal.action)==legal.end()||
           std::find(legal.begin(),legal.end(),proposal.rule)==legal.end())
            throw std::runtime_error("paired_node_legal_set_changed");
        return true;
    };
    auto suffix=[&](SearchRoot& world,const J& first) {
        J view=roots.projectView(*world.battle,input.relics,world.knownTop,input.character);
        for(int depth=int(proposal.prefix.size());depth<horizon;++depth) {
            checkTime();guard(view,metadata);
            auto legal=actions(*world.battle,true);J action=first;
            if(depth!=int(proposal.prefix.size())){++work.policies;action=combatChoice(view,legal,metadata);}
            const auto observed=advance(world,action,false);
            (void)spire_tree::observationKey(observed,observationBytes);
            view=roots.projectView(*world.battle,input.relics,world.knownTop,input.character);
            if(outcome(*world.battle))break;
        }
        ++work.values;double v=combatValue(view,outcome(*world.battle));
        if(!std::isfinite(v))throw std::runtime_error("paired_nonfinite_value");checkTime();
        if(work.objective){auto row=spire_objective::record(view,input.policy,outcome(*world.battle),v,work.objectiveScreen.objectiveVersion);
            row["utility"]=std::clamp((v+3.)/6.,0.,1.);return row;}
        return J{{"utility",std::clamp((v+3.)/6.,0.,1.)},{"value",v},{"hp",world.battle->player.curHp},
            {"outcome",outcome(*world.battle)},
            {"resource_hp",world.battle->player.curHp+inventoryReserve(view)-inventoryReserve(input.policy)}};
    };
    for(int attempt=0;attempt<spire_tree::verificationProposals;++attempt) {
        if(work.screen.count==unsigned(required)){work.stop="fixed_batch_complete";break;}
        // Reserve the worst-case COMPLETE pair, including BOTH prefix replays.
        // Rejections consume their actual prefix work. No partial-pair approval.
        if(stepCap-work.steps<2*horizon||sampleCap-work.samples<2||valueCap-work.values<2){work.stop="verification_budget";break;}
        checkTime();++work.attempts;
        roots.prepare(prepared,seeds.at(seedOffset+attempt),until);++prepared;
        ++work.samples;auto first=roots.branch(prepared-1);
        if(!replay(first)){++work.rejected;continue;}
        const bool challengerFirst=work.screen.count%2==0;
        J one=suffix(first,challengerFirst?proposal.action:proposal.rule);
        // Release the first queue before rebuilding; no paused-world copying.
        first=SearchRoot{};
        ++work.samples;auto second=roots.branch(prepared-1);
        if(!replay(second))throw std::runtime_error("paired_prefix_rebuild_changed");
        J two=suffix(second,challengerFirst?proposal.rule:proposal.action);
        const auto& alt=challengerFirst?one:two;const auto& ref=challengerFirst?two:one;
        double delta=alt.at("utility").get<double>()-ref.at("utility").get<double>();
        work.screen.add(delta);work.pairs.push_back({{"challenger",alt},{"rule",ref},{"difference",delta}});
        if(work.objective)work.objectiveScreen.add(alt,ref);
    }
    if(work.screen.count==unsigned(required))work.stop="fixed_batch_complete";
}

inline J auditPairedContinuation(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                                const J& budget,const J& metadata,const J& request) {
    if(!request.is_object()||request.size()!=2||request.at("version")!="paired-continuation-audit-v1"||
       seeds.size()!=spire_tree::verificationProposals||!budget.at("max_seconds").is_null())
        throw std::invalid_argument("paired_audit_contract");
    const auto& p=request.at("proposal");spire_tree::PairedProposal proposal;
    if(!p.is_object()||p.size()!=6||!p.at("prefix").is_array()||p.at("prefix").empty()||p.at("prefix").size()>64||
       p.at("depth")!=p.at("prefix").size())throw std::invalid_argument("paired_audit_proposal");
    proposal.action=p.at("action");proposal.rule=p.at("rule");proposal.support=p.at("support");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    for(const auto& step:p.at("prefix")) {
        const auto key=step.at("observation_key").get<std::string>();
        if(step.size()!=2||key.size()>limits.observationBytes)throw std::invalid_argument("paired_audit_prefix");
        proposal.prefix.push_back({step.at("action"),key});
    }
    const int horizon=std::min(limits.trainingHorizon,number(budget,"rollout_horizon"));
    if(int(proposal.prefix.size())>=horizon||proposal.action==proposal.rule||
       number(budget,"max_branch_steps")!=4096||number(budget,"max_samples")!=256||
       number(budget,"max_value_evaluations")!=256)throw std::invalid_argument("paired_audit_budget");
    roots.validate(roots.evidence().policy,roots.evidence().candidates,metadata);
    PairedWork work;int prepared=0;
    try {
        verifyContinuation(roots,seeds,0,prepared,proposal,horizon,4096,256,256,metadata,
            Clock::time_point::max(),work,16,limits.observationBytes);
        return {{"version","paired-continuation-audit-v1"},{"verification",work.diagnostic(true,16)},
            {"root_sampling",roots.diagnostics()},{"refusal",nullptr},{"diagnostic_only",true}};
    } catch(const std::exception& error) {
        return {{"version","paired-continuation-audit-v1"},{"verification",work.diagnostic(true,16)},
            {"root_sampling",roots.diagnostics()},{"refusal",error.what()},{"diagnostic_only",true}};
    }
}
} // namespace spire_search
