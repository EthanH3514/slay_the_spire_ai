#pragma once
// Diagnostic-only completed public paths/outcomes. Never consulted by selection.
#include "HistoryTree.h"
#include "CombatObjective.h"

namespace spire_tree {
class ObjectiveAudit {
    struct Record {
        spire_objective::Outcome outcome;
        std::vector<std::pair<unsigned,unsigned>> path;
    };
    unsigned cap_,depth_;
    std::vector<Record> records_;
public:
    ObjectiveAudit(unsigned cap,unsigned depth):cap_(cap),depth_(depth){
        if(!cap||cap>4096||!depth||depth>64)throw std::invalid_argument("objective_audit_bounds");
        records_.reserve(cap);
    }
    void completed(const std::vector<std::pair<Node*,Edge*>>& path,const spire_objective::Outcome& row){
        if(records_.size()>=cap_||path.empty()||path.size()>depth_)
            throw std::logic_error("objective_audit_path_bound");
        Record record{row,{}};record.path.reserve(path.size());
        for(std::size_t d=0;d<path.size();++d){
            const auto* n=path[d].first;const auto* selected=path[d].second;
            if(!n||n->depth!=d||!selected)throw std::logic_error("objective_audit_depth");
            unsigned ordinal=0;auto* edge=n->edges;
            while(edge&&edge!=selected){edge=edge->next;++ordinal;}
            if(!edge)throw std::logic_error("objective_audit_edge");
            record.path.emplace_back(n->id,ordinal);
        }
        records_.push_back(std::move(record));
    }
    J result()const{
        J records=J::array();std::size_t pathEdges=0;
        for(const auto& r:records_){const auto& o=r.outcome;pathEdges+=r.path.size();
            records.push_back({{"path",r.path},{"outcome",{{"value",o.value},{"hp",o.hp},
                {"resource_hp",o.resource},{"outcome",o.terminal}}}});
        }
        return {{"version","completed-objective-audit-v1"},{"scope","completed-backups-only"},
            {"max_records",cap_},{"max_path_edges",depth_},{"path_edges",pathEdges},{"records",records}};
    }
};
} // namespace spire_tree
