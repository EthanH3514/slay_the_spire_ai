#pragma once
// Opt-in, completed-path telemetry. Never queried by a policy/selector.
#include "HistoryTree.h"
#include <array>
#include <cstdint>

namespace spire_tree {
inline constexpr auto explorationProfileVersion="completed-tree-exploration-v1";
class ExplorationProfile {
    struct Counts {
        std::uint64_t completed=0,forced=0,firstNode=0,firstEdge=0,repeat=0;
        std::uint64_t firstRevisited=0,repeatUnvisited=0,openOpportunity=0;
        std::uint64_t uct=0,greedySame=0,equalTie=0,lowerMean=0;
        double lowerMeanGap=0;
    };
    std::array<Counts,64> depths_{};
public:
    // V9 backs up immediately after each simulation. No node is visited twice
    // on one path (history depth strictly increases), so these are the exact
    // statistics the selector saw. Interrupted/unbacked paths are excluded.
    void beforeBackup(const Node* n,const Edge* selected){
        if(!n||!selected||n->depth>=depths_.size())throw std::logic_error("profile_path");
        auto& c=depths_[n->depth];++c.completed;
        unsigned count=0,tried=0;const Edge* greedy=nullptr;double best=-1;
        for(auto* e=n->edges;e;e=e->next){
            ++count;tried+=e->visits>0;
            if(e->visits){double q=e->sum/e->visits;if(!greedy||q>best){greedy=e;best=q;}}
        }
        if(count==1){++c.forced;return;}
        if(tried&&tried<count)++c.openOpportunity;
        if(!selected->visits){
            ++c.firstEdge;
            if(!n->visits)++c.firstNode;else ++c.firstRevisited;
            return;
        }
        ++c.repeat;
        if(tried<count)++c.repeatUnvisited;
        // For unchanged V9 all repeat multi-action selections are UCT.
        ++c.uct;
        const double q=selected->sum/selected->visits;
        if(selected==greedy)++c.greedySame;
        else if(q==best)++c.equalTie;
        else {++c.lowerMean;c.lowerMeanGap+=best-q;}
    }
    J result(const Tree& tree)const{
        J counts=J::array(),nodes=J::array();
        for(unsigned d=0;d<depths_.size();++d){const auto& c=depths_[d];if(!c.completed)continue;
            counts.push_back({{"depth",d},{"completed",c.completed},{"forced",c.forced},
                {"first_node",c.firstNode},{"first_edge",c.firstEdge},{"repeat",c.repeat},
                {"first_edge_revisited_node",c.firstRevisited},{"repeat_with_unvisited",c.repeatUnvisited},
                {"open_opportunities",c.openOpportunity},{"uct",c.uct},
                {"uct_greedy_same",c.greedySame},{"uct_equal_mean_tie",c.equalTie},
                {"uct_lower_mean",c.lowerMean},{"uct_lower_mean_gap_sum",c.lowerMeanGap}});
        }
        tree.visitNodes([&](const Node* n){
            unsigned actions=0,tried=0,once=0,children=0,split=0,maximum=0;
            for(auto* e=n->edges;e;e=e->next){
                ++actions;tried+=e->visits>0;once+=e->visits==1;
                unsigned childCount=0;for(auto* child=e->children;child;child=child->sibling)++childCount;
                children+=childCount;split+=childCount>1;maximum=std::max(maximum,childCount);
            }
            // Compact rows bounded by max_nodes. No action/observation strings,
            // sampled worlds or RNG are retained in the profile.
            nodes.push_back(J::array({n->depth,actions,n->visits,tried,once,children,split,maximum}));
        });
        return {{"version",explorationProfileVersion},{"scope","completed-backups-only"},
            {"counter_bytes",sizeof(depths_)},{"by_depth",counts},
            {"node_columns",{"depth","actions","visits","tried","once","children","split_edges","max_children"}},
            {"node_rows",nodes}};
    }
};
} // namespace spire_tree
