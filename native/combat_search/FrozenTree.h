#pragma once
#include "HistoryTree.h"

namespace spire_tree {
// Owned, bounded copy of PUBLIC history/statistics. No world, RNG, queue, or
// pointers into the mutable construction tree survive this constructor.
class FrozenTree {
    Tree tree_;
    Node* root_;
    static J legal(const Node* n) {
        J out=J::array();for(auto* e=n->edges;e;e=e->next)out.push_back(J::parse(e->action));return out;
    }
    void copy(const Node* src,Node* dst) {
        if(!dst)throw std::runtime_error("frozen_snapshot_arena");
        dst->visits=src->visits;dst->id=src->id;
        auto* de=dst->edges;
        for(auto* se=src->edges;se;se=se->next,de=de->next){
            if(!de||de->action!=se->action)throw std::runtime_error("frozen_snapshot_action");
            de->visits=se->visits;de->sum=se->sum;
            for(auto* sn=se->children;sn;sn=sn->sibling)
                copy(sn,tree_.child(dst,de,sn->observation,legal(sn),true));
        }
    }
public:
    FrozenTree(Limits limits,const Node* source):tree_(limits),root_(tree_.root(legal(source))) {
        copy(source,root_);tree_.freeze();
    }
    const Node* root()const{return root_;}
    const Edge* edge(const Node* n,const J& action)const{
        if(!n)return nullptr;auto key=action.dump();
        for(auto* e=n->edges;e;e=e->next)if(e->action==key)return e;
        throw std::runtime_error("frozen_action_missing");
    }
    const Edge* choose(const Node* n,const J& actions)const{
        tree_.check(n,actions);const Edge* best=nullptr;double score=-1e100;
        if(n)for(auto* e=n->edges;e;e=e->next)if(e->visits){
            const double mean=e->sum/e->visits;
            if(!best||mean>score){best=e;score=mean;}
        }
        return best;
    }
    const Node* child(const Edge* e,std::string_view observation,const J& actions)const{
        if(e)for(auto* n=e->children;n;n=n->sibling)if(n->observation==observation){
            tree_.check(n,actions);return n;
        }
        return nullptr;
    }
    J diagnostic()const{
        auto d=tree_.diagnostics(false);d["root"]=tree_.decisionDiagnostic(root_);
        d["selection"]="visited-scalar-mean-canonical-tie-no-exploration";
        d["unknown_history"]="charged-rule-leaf-no-rejoin";return d;
    }
};
} // namespace spire_tree
