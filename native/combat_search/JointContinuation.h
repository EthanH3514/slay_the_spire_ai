#pragma once
#include "PairedContinuation.h"
#include <map>

namespace spire_tree {
inline constexpr auto jointVersion="public-history-tree-v7-joint-continuation";
inline constexpr unsigned jointNodeCap=8;

struct JointPolicy {
    Node* root=nullptr;Edge* first=nullptr;
    std::vector<std::pair<Node*,Edge*>> changes;
    unsigned maxPathChanges=0;
    Edge* select(Node* node) const {
        for(const auto& change:changes)if(change.first==node)return change.second;
        return nullptr;
    }
    J diagnostic(bool detail=false) const {
        J out={{"node_cap",jointNodeCap},{"nodes",changes.size()},{"max_path_changes",maxPathChanges},
            {"carrier_action",first?J::parse(first->action):J()},
            {"nomination","most-visits-then-objective-rule-canonical"},
            {"unknown_observation","charged-rule-continuation"},{"root_conditioned_future",false}};
        if(detail){out["changes"]=J::array();for(const auto& c:changes)
            out["changes"].push_back({{"node",c.first->id},{"depth",c.first->depth},{"action",J::parse(c.second->action)}});}
        return out;
    }
};

struct JointUsage {
    std::array<Node*,jointNodeCap> nodes{};unsigned count=0;
    void add(Node* node) {
        if(!node)throw std::runtime_error("joint_null_override");
        for(unsigned i=0;i<count;++i)if(nodes[i]==node)return;
        if(count==nodes.size())throw std::runtime_error("joint_path_limit");
        nodes[count++]=node;
    }
};

// Public history navigation only. No latent state, outcomes of validation, or
// simulation RNG enters a policy key. Leave the old single-node path untouched.
class JointCandidates {
    struct Link {Node* node=nullptr;Node* parent=nullptr;Edge* via=nullptr;Edge* prior=nullptr;};
    std::vector<Link> links_;Node* root_=nullptr;
public:
    explicit JointCandidates(int cap):links_(cap){}
    void link(Node* node,Node* parent,Edge* via) {
        if(!node)return;auto& item=links_.at(node->id);
        if(item.node&&(item.parent!=parent||item.via!=via))throw std::runtime_error("joint_history_link_changed");
        item.node=node;item.parent=parent;item.via=via;
        if(!parent){if(root_&&root_!=node)throw std::runtime_error("joint_root_changed");root_=node;}
    }
    void prior(Tree& tree,Node* node,const J& action) {
        if(!node)return;auto& item=links_.at(node->id);auto* edge=tree.edge(node,action);
        if(!item.node)throw std::runtime_error("joint_unlinked_prior");
        if(item.prior&&item.prior!=edge)throw std::runtime_error("joint_public_rule_changed");
        item.prior=edge;
    }
    std::size_t bytes() const {return links_.capacity()*sizeof(Link);}
    JointPolicy propose(Tree& tree,const ObjectiveStats& stats,int horizon,unsigned& eligible) const {
        if(!tree.diagnostics().at("frozen").get<bool>())throw std::runtime_error("joint_unfrozen_nomination");
        JointPolicy best;eligible=0;if(!root_)return best;
        // Summarize any edge at most once: tie comparisons must not repeatedly
        // sort a large reference distribution for each alternative.
        std::map<const Edge*,spire_objective::Key> keys;
        auto key=[&](const Edge* e)->const spire_objective::Key& {
            auto it=keys.find(e);if(it==keys.end())it=keys.emplace(e,stats.key(e)).first;return it->second;
        };
        auto choose=[&](Node* n,Edge* prior) {
            Edge* chosen=nullptr;
            for(auto* e=n->edges;e;e=e->next)if(e->visits) {
                if(!chosen||e->visits>chosen->visits||
                    (e->visits==chosen->visits&&(key(e)>key(chosen)||
                        (key(e)==key(chosen)&&((e==prior&&chosen!=prior)||
                            ((e==prior)==(chosen==prior)&&e->action<chosen->action))))))chosen=e;
            }
            return chosen?chosen:prior;
        };
        struct Pending {Node* node;unsigned changes;};
        std::vector<Pending> pending;
        auto append=[&](Edge* e,unsigned changes) {
            std::vector<Node*> children;
            for(auto* n=e->children;n;n=n->sibling)children.push_back(n);
            std::sort(children.begin(),children.end(),[](const Node* a,const Node* b){
                return a->visits!=b->visits?a->visits>b->visits:a->observation<b->observation;
            });
            for(auto i=children.rbegin();i!=children.rend();++i)pending.push_back({*i,changes});
        };
        for(auto* first=root_->edges;first;first=first->next)if(first->visits) {
            JointPolicy candidate;candidate.root=root_;candidate.first=first;pending.clear();append(first,0);
            while(!pending.empty()) {
                auto item=pending.back();pending.pop_back();auto* n=item.node;
                const auto& link=links_.at(n->id);
                if(link.node!=n)throw std::runtime_error("joint_missing_history_link");
                if(n->depth>=unsigned(horizon)||!link.prior)continue;
                auto* selected=candidate.changes.size()<jointNodeCap?choose(n,link.prior):link.prior;
                unsigned pathChanges=item.changes;
                if(selected!=link.prior){candidate.changes.push_back({n,selected});++pathChanges;}
                candidate.maxPathChanges=std::max(candidate.maxPathChanges,pathChanges);
                append(selected,pathChanges);
            }
            if(candidate.maxPathChanges<2)continue;
            ++eligible;
            if(!best.first||key(first)>key(best.first)||(key(first)==key(best.first)&&
                (first->visits>best.first->visits||(first->visits==best.first->visits&&first->action<best.first->action))))
                best=std::move(candidate);
        }
        return best;
    }
};
} // namespace spire_tree

namespace spire_search {
struct JointWork:PairedWork {
    unsigned maxPathChanges=0,multiPaths=0,overrideSteps=0;
    JointWork(){objective=true;}
    void note(const spire_tree::JointUsage& usage) {
        maxPathChanges=std::max(maxPathChanges,usage.count);multiPaths+=usage.count>=2;overrideSteps+=usage.count;
    }
    J diagnostic(bool detail) const {
        auto out=PairedWork::diagnostic(detail);
        out.update({{"max_distinct_overrides",maxPathChanges},{"multi_node_trajectories",multiPaths},
            {"override_steps",overrideSteps},{"future_prefix_conditioning",false}});return out;
    }
};

// Shared by the simulator and finite chance-world tests. The stepper exposes
// only current public actions/rule choice and observations AFTER an action.
template<class Stepper>
J rolloutJointPolicy(spire_tree::Tree& tree,const spire_tree::JointPolicy& policy,bool enabled,
        int horizon,Stepper& world,JointWork& work,Clock::time_point until,
        std::size_t observationBytes,spire_tree::JointUsage& usage,bool chargeSample=true) {
    auto checkTime=[&]{if(Clock::now()>=until)throw std::runtime_error("tree_deadline");};
    checkTime();if(chargeSample)++work.samples;
    if(!policy.first||!policy.root)throw std::runtime_error("joint_missing_carrier");
    auto* node=policy.root;auto legal=world.legal();
    for(int depth=0;depth<horizon;++depth) {
        checkTime();tree.check(node,legal);
        auto* overrideEdge=enabled&&depth?policy.select(node):nullptr;
        J action;
        if(!depth)action=J::parse(policy.first->action);
        else if(overrideEdge){action=J::parse(overrideEdge->action);usage.add(node);}
        else {++work.policies;action=world.prior(legal);}
        if(std::find(legal.begin(),legal.end(),action)==legal.end())throw std::runtime_error("joint_illegal_action");
        auto* via=tree.edge(node,action);
        ++work.steps;auto observed=world.advance(action);checkTime();
        auto key=spire_tree::observationKey(observed,observationBytes);
        if(world.terminal()||depth+1==horizon)break;
        legal=world.legal();node=tree.child(node,via,key,legal,false);
    }
    ++work.values;J row=world.score();
    if(!row.at("value").is_number()||!std::isfinite(row.at("value").get<double>()))
        throw std::runtime_error("joint_nonfinite_value");
    checkTime();return row;
}

template<class Prepare,class Run>
void verifyJointBatch(const spire_tree::JointPolicy& policy,int horizon,int stepCap,int sampleCap,int valueCap,
        JointWork& work,Clock::time_point until,Prepare prepare,Run run) {
    if(!policy.first)return;
    for(int pair=0;pair<spire_tree::verificationPairs;++pair) {
        if(Clock::now()>=until)throw std::runtime_error("tree_deadline");
        // No future-prefix replay/rejection. Every prepared root contributes,
        // even if the desired card is never observed or no override is reached.
        if(stepCap-work.steps<2*horizon||sampleCap-work.samples<2||valueCap-work.values<2){
            work.stop="verification_budget";return;
        }
        ++work.attempts;prepare(pair);
        const bool challengerFirst=pair%2==0;
        J one=run(challengerFirst,pair);J two=run(!challengerFirst,pair);
        const auto& alt=challengerFirst?one:two;const auto& ref=challengerFirst?two:one;
        double delta=std::clamp((alt.at("value").get<double>()+3.)/6.,0.,1.)-
            std::clamp((ref.at("value").get<double>()+3.)/6.,0.,1.);
        work.screen.add(delta);work.objectiveScreen.add(alt,ref);
        work.pairs.push_back({{"challenger",alt},{"rule",ref},{"difference",delta}});
    }
    work.stop="fixed_batch_complete";
}

inline void verifyJointContinuation(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,int seedOffset,
        int& prepared,spire_tree::Tree& tree,const spire_tree::JointPolicy& policy,int horizon,
        int stepCap,int sampleCap,int valueCap,const J& metadata,Clock::time_point until,
        JointWork& work,std::size_t observationBytes) {
    const auto& input=roots.evidence();
    struct PublicStepper {
        ConditionalRoots& roots;SearchRoot world;const J& metadata;J view;std::string objectiveVersion;
        void project(){const auto& in=roots.evidence();
            view=roots.projectView(*world.battle,in.relics,world.knownTop,in.character);}
        J legal(){guard(view,metadata);return actions(*world.battle,true);}
        J prior(const J& choices){return combatChoice(view,choices,metadata);}
        std::vector<std::string> advance(const J& action){auto events=roots.advancePublic(world,action);project();return events;}
        bool terminal() const{return outcome(*world.battle)!=0;}
        J score() const{return spire_objective::record(view,roots.evidence().policy,
            outcome(*world.battle),combatValue(view,outcome(*world.battle)),objectiveVersion);}
    };
    verifyJointBatch(policy,horizon,stepCap,sampleCap,valueCap,work,until,
        [&](int pair){roots.prepare(prepared,seeds.at(seedOffset+pair),until);++prepared;},
        [&](bool enabled,int) {
            // Charge before root rebuilding too. The first owned queue is
            // released on return before the second arm is reconstructed.
            if(Clock::now()>=until)throw std::runtime_error("tree_deadline");
            ++work.samples;PublicStepper world{roots,roots.branch(prepared-1),metadata,{},work.objectiveScreen.objectiveVersion};world.project();
            if(world.view!=input.policy||!roots.sameActions(world.legal(),input.candidates))
                throw std::runtime_error("joint_root_changed");
            spire_tree::JointUsage usage;
            auto row=rolloutJointPolicy(tree,policy,enabled,horizon,world,work,until,observationBytes,usage,false);
            if(enabled)work.note(usage);return row;
        });
}
} // namespace spire_search
