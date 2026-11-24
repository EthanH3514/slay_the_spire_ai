#pragma once
#include "ObjectiveTree.h"
#include <array>

namespace spire_tree {
inline constexpr auto focusedVersion="public-history-tree-v8-focused-expansion";
inline constexpr auto deepVersion="public-history-tree-v8-deep-expansion";
inline constexpr auto focusOnlyVersion="public-history-tree-v8-focused-single";

struct ConstructionVariant {
    bool enabled=false,focused=false;unsigned newNodeCap=1;
    static ConstructionVariant parse(const std::string& version) {
        if(version==focusedVersion)return {true,true,3};
        if(version==deepVersion)return {true,false,3};
        if(version==focusOnlyVersion)return {true,true,1};
        return {};
    }
};

// Completed public-history outcomes ONLY. Call before preparing the next world.
// Keeping this scheduler separate makes that information boundary testable.
class FocusedRoots {
    std::vector<Edge*> edges_;std::array<Edge*,2> selected_{};
    unsigned scheduled_=0,exploits_=0,explores_=0;bool frozen_=false;
public:
    explicit FocusedRoots(Node* root) {
        for(auto* e=root->edges;e;e=e->next)edges_.push_back(e);
        std::sort(edges_.begin(),edges_.end(),[](const Edge* a,const Edge* b){return a->action<b->action;});
        if(edges_.size()<2)throw std::invalid_argument("focused_root_action_count");
    }
    Edge* next(const ObjectiveStats& stats) {
        const auto pilot=2*edges_.size();
        if(scheduled_<pilot)return edges_[scheduled_++%edges_.size()];
        if(!frozen_) {
            std::vector<std::pair<Edge*,spire_objective::Key>> ranks;
            for(auto* e:edges_) {
                if(e->visits<2)throw std::runtime_error("focused_pilot_incomplete");
                ranks.push_back({e,stats.key(e)});
            }
            std::sort(ranks.begin(),ranks.end(),[](const auto& a,const auto& b){
                return a.second!=b.second?a.second>b.second:a.first->action<b.first->action;
            });
            selected_={ranks[0].first,ranks[1].first};frozen_=true;
        }
        const auto offset=scheduled_++-pilot;
        if(offset%4==3)return edges_[explores_++%edges_.size()];
        return selected_[exploits_++%selected_.size()];
    }
    J diagnostic() const {
        J out={{"pilot_rounds",2},{"pilot_required",2*edges_.size()},
            {"scheduled",scheduled_},{"pair_frozen",frozen_},{"focused_actions",J::array()},
            {"post_pilot_exploit",exploits_},{"post_pilot_explore",explores_},
            {"root_actions",J::array()},{"selection_before_world",true}};
        if(frozen_)for(auto* e:selected_)out["focused_actions"].push_back(J::parse(e->action));
        for(auto* e:edges_)out["root_actions"].push_back({{"action",J::parse(e->action)},{"completed_visits",e->visits}});
        return out;
    }
};

// Read-only telemetry; do not change the legacy tree's arena layout or scans.
inline J constructionDepths(Node* root) {
    std::array<unsigned,65> visited{},alternatives{};std::vector<Node*> pending{root};
    while(!pending.empty()) {
        auto* n=pending.back();pending.pop_back();unsigned tried=0;
        for(auto* e=n->edges;e;e=e->next){tried+=e->visits>0;
            for(auto* child=e->children;child;child=child->sibling)pending.push_back(child);}
        visited.at(n->depth)+=tried>0;alternatives.at(n->depth)+=tried>=2;
    }
    J out=J::array();
    for(unsigned d=0;d<visited.size();++d)if(visited[d]||alternatives[d])
        out.push_back({{"depth",d},{"visited_nodes",visited[d]},{"multi_action_nodes",alternatives[d]}});
    return out;
}
} // namespace spire_tree
