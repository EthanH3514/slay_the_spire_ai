#pragma once
// Public history/statistics only. No BattleContext, RNG or hidden-state keys.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "Performance.h"
#include "ResourceUtility.h"

namespace spire_tree {
using J=nlohmann::json;
inline constexpr auto version="public-history-tree-v1";
inline constexpr auto guardedVersion="public-history-tree-v2";
// A minimal repeated-evidence gate, NOT a confidence/significance guarantee.
inline constexpr unsigned minimumContinuationVisits=2;
struct Limits {
    int trainingRollouts=64, trainingHorizon=16, depth=12, nodes=2048;
    std::size_t bytes=8*1024*1024, observationBytes=1024*1024;
    static Limits parse(const J& j) {
        const std::vector<std::string> names={"training_rollouts","training_horizon","max_depth","max_nodes",
                                            "arena_bytes","observation_bytes"};
        if(!j.is_object()||j.size()!=names.size())throw std::invalid_argument("tree_config_fields");
        for(const auto& name:names)if(!j.contains(name)||!j.at(name).is_number_integer()||
            j.at(name).get<std::int64_t>()<1)throw std::invalid_argument("tree_config_positive_integer");
        Limits l;
        if(j.at("training_rollouts")>4096||j.at("training_horizon")>512||j.at("max_depth")>64||
           j.at("max_nodes")>65536||j.at("arena_bytes")>128*1024*1024||j.at("observation_bytes")>4*1024*1024)
            throw std::invalid_argument("tree_config_limit");
        l.trainingRollouts=j.at("training_rollouts");l.trainingHorizon=j.at("training_horizon");
        l.depth=j.at("max_depth");l.nodes=j.at("max_nodes");l.bytes=j.at("arena_bytes");
        l.observationBytes=j.at("observation_bytes");return l;
    }
};
struct Node;
struct Edge {
    std::string_view action;
    Edge* next=nullptr;
    Node* children=nullptr;
    unsigned visits=0;
    double sum=0;
};
struct Node {
    unsigned id=0,depth=0,visits=0;
    Edge* edges=nullptr;
    Node* sibling=nullptr;
    Node* allNext=nullptr;
    std::string_view observation;
};

class Tree {
    Limits limits_;
    std::vector<std::byte> storage_;
    std::pmr::monotonic_buffer_resource arena_;
    Node* all_=nullptr;
    unsigned count_=0,closed_=0,reused_=0,observations_=0,compared_=0;
    bool frozen_=false;
    std::size_t payload_=0;
    template<class T> T* allocate() {
        auto* p=static_cast<T*>(arena_.allocate(sizeof(T),alignof(T)));
        payload_+=sizeof(T);return new(p) T{};
    }
    std::string_view copy(std::string_view s) {
        if(s.empty())return {};
        auto* p=static_cast<char*>(arena_.allocate(s.size(),alignof(char)));
        std::memcpy(p,s.data(),s.size());payload_+=s.size();return {p,s.size()};
    }
    static std::vector<std::string> canonical(const J& actions) {
        if(!actions.is_array()||actions.empty())throw std::invalid_argument("tree_empty_actions");
        std::vector<std::string> out;out.reserve(actions.size());
        for(const auto& a:actions)out.push_back(a.dump());
        std::sort(out.begin(),out.end());
        if(std::adjacent_find(out.begin(),out.end())!=out.end())throw std::invalid_argument("tree_duplicate_action");
        return out;
    }
    Node* make(unsigned depth,std::string_view observation,const J& actions) {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(frozen_||closed_||depth>=unsigned(limits_.depth))return nullptr;
        if(count_>=unsigned(limits_.nodes)){++closed_;return nullptr;}
        auto names=canonical(actions);
        try {
            auto* n=allocate<Node>();n->depth=depth;n->observation=copy(observation);
            Edge** next=&n->edges;
            for(const auto& a:names){auto* e=allocate<Edge>();e->action=copy(a);*next=e;next=&e->next;}
            n->id=count_++;n->allNext=all_;all_=n;return n;
        } catch(const std::bad_alloc&) {++closed_;return nullptr;}
    }
public:
    explicit Tree(Limits limits):limits_(limits),storage_(limits.bytes),
        arena_(storage_.data(),storage_.size(),std::pmr::null_memory_resource()) {}
    Tree(const Tree&)=delete;
    Tree& operator=(const Tree&)=delete;
    Node* root(const J& actions){return make(0,{},actions);}
    void freeze(){frozen_=true;}
    void check(const Node* n,const J& actions) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n)return;
        auto names=canonical(actions);auto* e=n->edges;
        for(const auto& name:names){if(!e||e->action!=name)throw std::runtime_error("tree_public_legal_set_mismatch");e=e->next;}
        if(e)throw std::runtime_error("tree_public_legal_set_mismatch");
    }
    Edge* edge(Node* n,const J& action) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n)return nullptr;
        auto key=action.dump();for(auto* e=n->edges;e;e=e->next)if(e->action==key)return e;
        throw std::runtime_error("tree_action_missing");
    }
    Edge* select(Node* n,bool learn) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n)return nullptr;
        if(learn&&frozen_)throw std::runtime_error("tree_frozen_mutation");
        Edge* best=nullptr;double score=-1e100;
        for(auto* e=n->edges;e;e=e->next) {
            if(!e->visits){if(learn)return e;continue;}
            double q=e->sum/e->visits;
            if(learn)q+=std::sqrt(2.*std::log(double(std::max(1u,n->visits)))/e->visits);
            if(!best||q>score){score=q;best=e;}
        }
        return best;
    }
    Edge* selectGuarded(Node* n,bool learn,const J& ruleAction) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n)return nullptr;
        if(learn&&frozen_)throw std::runtime_error("tree_frozen_mutation");
        auto* prior=edge(n,ruleAction);
        // Real rollouts only: no invented visits, values or pseudo-observations.
        if(learn)return !prior->visits?prior:select(n,true);
        if(prior->visits<minimumContinuationVisits)return nullptr;
        auto* best=prior;
        double score=prior->sum/prior->visits;
        for(auto* e=n->edges;e;e=e->next) {
            if(e->visits<minimumContinuationVisits)continue;
            const double q=e->sum/e->visits;
            // Exact ties keep the rule, then canonical order among alternatives.
            if(q>score){best=e;score=q;}
        }
        return best;
    }
    Edge* selectDeepQuarter(Node* n,const J& ruleAction) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n||!n->depth)throw std::logic_error("quarter_requires_nonroot_node");
        if(frozen_)throw std::runtime_error("tree_frozen_mutation");
        auto* prior=edge(n,ruleAction);
        if(!prior->visits)return prior;
        Edge* best=nullptr;double score=-1e100;
        for(auto* e=n->edges;e;e=e->next){
            if(!e->visits)return e;
            const double q=e->sum/e->visits+
                .25*std::sqrt(2.*std::log(double(std::max(1u,n->visits)))/e->visits);
            if(!best||q>score){score=q;best=e;}
        }
        return best;
    }
    Edge* selectScoreOrdered(Node* n,const J& legal,const J& ruleAction,const std::vector<double>& scores) const {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(!n||!n->depth)throw std::logic_error("score_order_requires_nonroot_node");
        if(frozen_)throw std::runtime_error("tree_frozen_mutation");
        if(!legal.is_array()||legal.empty()||scores.size()!=legal.size())throw std::invalid_argument("score_order_size");
        std::vector<std::pair<std::string,double>> keyed;keyed.reserve(legal.size());
        for(std::size_t i=0;i<legal.size();++i){
            if(std::isnan(scores[i])||scores[i]==std::numeric_limits<double>::infinity())
                throw std::invalid_argument("score_order_invalid_score");
            keyed.emplace_back(legal[i].dump(),scores[i]);
        }
        std::sort(keyed.begin(),keyed.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        Edge* best=nullptr;auto* e=n->edges;double score=-std::numeric_limits<double>::infinity();
        for(const auto& item:keyed){
            if(!e||e->action!=item.first)throw std::invalid_argument("score_order_legal_alignment");
            if(!e->visits&&(!best||item.second>score)){best=e;score=item.second;}
            e=e->next;
        }
        if(e)throw std::invalid_argument("score_order_legal_alignment");
        auto* prior=edge(n,ruleAction);if(!prior->visits)return prior;
        return best?best:select(n,true);
    }
    Node* child(Node* parent,Edge* e,std::string_view observation,const J& actions,bool learn,bool* created=nullptr) {
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(created)*created=false;
        if(!parent||!e)return nullptr;
        if(observation.size()>limits_.observationBytes)throw std::runtime_error("tree_observation_bytes");
        for(auto* n=e->children;n;n=n->sibling) {
            ++compared_;
            if(n->observation==observation){++reused_;check(n,actions);return n;}
        }
        if(!learn)return nullptr;
        auto* n=make(parent->depth+1,observation,actions);
        if(n){n->sibling=e->children;e->children=n;++observations_;if(created)*created=true;}return n;
    }
    void backup(Node* n,Edge* e,double value) {
        if(frozen_)throw std::runtime_error("tree_frozen_mutation");
        if(!n||!e||!std::isfinite(value)||value<0||(value>1&&!spire_resource_utility::linear))throw std::runtime_error("tree_invalid_value");
        ++n->visits;++e->visits;e->sum+=value;
    }
    template<class Visitor> void visitNodes(Visitor&& visit) const {
        for(const Node* n=all_;n;n=n->allNext)visit(n);
    }
    J decisionDiagnostic(const Node* n) const {
        // Read-only public action statistics; no observation or latent-world keys.
        if(!n)return nullptr;
        J edges=J::array();
        for(auto* e=n->edges;e;e=e->next)edges.push_back({{"action",J::parse(e->action)},
            {"visits",e->visits},{"mean",e->visits?J(e->sum/e->visits):J(nullptr)}});
        return {{"id",n->id},{"depth",n->depth},{"visits",n->visits},{"actions",edges}};
    }
    J diagnostics(bool detail=false) const {
        unsigned maxDepth=0,multi=0,nonroot=0;J byDepth=J::object(),rows=J::array();
        for(auto* n=all_;n;n=n->allNext) {
            unsigned tried=0;J edges=J::array();
            for(auto* e=n->edges;e;e=e->next){tried+=e->visits>0;
                if(detail&&rows.size()<32)edges.push_back({{"action",J::parse(e->action)},
                    {"visits",e->visits},{"mean",e->visits?e->sum/e->visits:0.}});}
            if(n->visits)maxDepth=std::max(maxDepth,n->depth+1);
            multi+=tried>1;nonroot+=n->depth>0&&tried>1;
            auto key=std::to_string(n->depth);byDepth[key]=byDepth.value(key,0)+n->visits;
            if(detail&&rows.size()<32)rows.push_back({{"id",n->id},{"depth",n->depth},{"visits",n->visits},{"actions",edges}});
        }
        return {{"version",version},{"nodes",count_},{"arena_bytes",storage_.size()},
            {"retained_payload_bytes",payload_},{"expansion_closed",bool(closed_)},{"max_visited_decision_depth",maxDepth},
            {"multi_action_nodes",multi},{"nonroot_multi_action_nodes",nonroot},{"visits_by_depth",byDepth},
            {"observation_children",observations_},{"history_reuses",reused_},{"child_comparisons",compared_},
            {"frozen",frozen_},{"node_examples",rows}};
    }
};

inline std::string observationKey(const std::vector<std::string>& observations,std::size_t limit) {
    // Length prefixes avoid concatenation collisions; exact bytes, no hash-only identity.
    std::string key;
    for(const auto& o:observations){auto prefix=std::to_string(o.size())+":";
        if(prefix.size()>limit-key.size()||o.size()>limit-key.size()-prefix.size())
            throw std::runtime_error("tree_observation_bytes");
        key+=prefix;key+=o;}
    return key;
}
} // namespace spire_tree
