#pragma once
// Post-search public topology only. Does not allocate in or mutate the tree.
#include "HistoryTree.h"

namespace spire_tree {
inline J branchingProfile(const Tree& tree){
    J nodes=J::array(),edges=J::array();
    tree.visitNodes([&](const Node* n){
        unsigned ordinal=0;
        for(const auto* e=n->edges;e;e=e->next,++ordinal){
            if(!e->visits&&!e->children)continue;
            J children=J::array();
            for(const auto* c=e->children;c;c=c->sibling)children.push_back(c->id);
            edges.push_back(J::array({n->id,ordinal,e->visits,children}));
        }
        nodes.push_back(J::array({n->id,n->depth,n->visits,ordinal}));
    });
    return {{"version","public-tree-branching-v1"},{"scope","completed-decision-visits"},
        {"node_columns",{"id","depth","visits","actions"}},{"node_rows",nodes},
        {"edge_columns",{"parent","ordinal","visits","children"}},{"edge_rows",edges}};
}
} // namespace spire_tree
