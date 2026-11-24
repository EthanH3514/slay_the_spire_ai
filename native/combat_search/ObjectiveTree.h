#pragma once
#include "HistoryTree.h"
#include "CombatObjective.h"
#include <map>

namespace spire_tree {
inline constexpr auto objectiveVersion="public-history-tree-v4-objective";
inline constexpr auto efficientVersion="public-history-tree-v5-nomination";
// Side storage keeps the historical tree layout and old arena behavior intact.
// Each retained outcome corresponds to a charged, actually simulated tree step.
class ObjectiveStats {
    std::map<const Edge*,std::vector<spire_objective::Outcome>> rows_;
    std::size_t count_=0,cap_;
    std::string objectiveVersion_;
public:
    explicit ObjectiveStats(std::size_t cap,const std::string& objectiveVersion=spire_objective::version)
        :cap_(cap),objectiveVersion_(objectiveVersion){spire_objective::validateVersion(objectiveVersion);}
    void add(const Edge* edge,const J& row) {
        if(!edge||count_>=cap_)throw std::runtime_error("objective_tree_sample_limit");
        rows_[edge].push_back(spire_objective::Outcome::read(row));++count_;
    }
    spire_objective::Key key(const Edge* edge) const {return spire_objective::summarize(rows_.at(edge));}
    Edge* select(Tree& tree,Node* node,const J& prior,unsigned minimum) const {
        if(!node)return nullptr;auto* ref=tree.edge(node,prior);
        if(ref->visits<minimum)return nullptr;
        auto* best=ref;auto score=key(ref);
        for(auto* e=node->edges;e;e=e->next)if(e->visits>=minimum){auto k=key(e);if(k>score){best=e;score=k;}}
        return best;
    }
    J diagnostics() const {return {{"version",objectiveVersion_},{"retained_outcomes",count_},
        {"outcome_cap",cap_},{"sampled_edges",rows_.size()},
        {"exploration","legacy-UCT-surrogate-only; nomination-and-approval-use-distribution"}};}
};
} // namespace spire_tree
