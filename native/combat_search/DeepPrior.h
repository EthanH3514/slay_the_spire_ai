#pragma once
#include "HistoryTree.h"

namespace spire_tree {
inline constexpr auto deepPriorVersion="public-history-tree-v15-deep-prior";
inline constexpr auto deepPriorSelection="public-rule-first-then-positive-prior-PUCT-parent-mean-FPU";

// Public tree statistics only. No retained estimates, pseudo-visits or randomness.
// Caller verifies legal-set identity, rule edge membership and non-frozen owner.
inline Edge* deepPriorChoice(Node* node,Edge* prior) {
    if(!node||!node->depth||!prior)throw std::logic_error("deep_prior_requires_public_node");
    if(!node->visits)return prior;
    unsigned actions=0;double total=0;
    for(auto* e=node->edges;e;e=e->next){++actions;total+=e->sum;}
    const double firstPlay=total/node->visits,scale=std::sqrt(double(node->visits));
    Edge* best=nullptr;double maximum=-std::numeric_limits<double>::infinity();
    for(auto* e=node->edges;e;e=e->next){
        const double q=e->visits?e->sum/e->visits:firstPlay;
        const double probability=.5/actions+(e==prior?.5:0.);
        const double score=q+probability*scale/(1.+e->visits);
        if(!best||score>maximum){best=e;maximum=score;}
    }
    return best;
}
} // namespace spire_tree
