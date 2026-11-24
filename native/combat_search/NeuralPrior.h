#pragma once
#include "HistoryTree.h"
#include "RootPolicyRecord.h"
#include "ValueMemo.h"
#include "PuctExploration.h"
#include "AllocationProbe.h"
#include "RichCombatPublic.h"
#include "RichCombatModel.h"
#include "ColdLeaf.h"
#include <chrono>
#include <map>
#include <functional>
namespace spire_neural_prior {
using J=nlohmann::json;using spire_tree::Node;using spire_tree::Edge;
inline constexpr auto version="public-neural-prior-v1";
inline constexpr auto memoVersion="exact-public-prior-memo-v1";
inline thread_local bool memoEnabled=false;
inline thread_local std::unique_ptr<spire_rich_model::Model> model;
inline thread_local std::string identity,continuationIdentity;
inline void bind(const J& weights,const J& schema,const std::string& parentDigest,const std::string& id,const std::string& continuation) {
    if(id.size()!=64||continuation.size()!=64)throw std::invalid_argument("prior_weight_identity");
    auto next=std::make_unique<spire_rich_model::Model>(weights,schema,parentDigest);
    model=std::move(next);identity=id;continuationIdentity=continuation;
}
struct PublicContext {J root,relicOrder;};
inline thread_local const PublicContext* active=nullptr;
struct Scope {
    PublicContext context;
    Scope(J root,J order):context{std::move(root),std::move(order)} {
        if(active||!model||spire_cold::identity!=continuationIdentity)throw std::runtime_error("prior_scope_model");
        active=&context;
    }
    ~Scope(){active=nullptr;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
};
inline std::vector<double> probabilities(const std::vector<float>& logits,bool uniform=false) {
    if(logits.empty())throw std::invalid_argument("prior_empty_actions");
    double top=-std::numeric_limits<double>::infinity();
    for(auto v:logits){if(!std::isfinite(v))throw std::invalid_argument("prior_nonfinite");top=std::max(top,double(v));}
    std::vector<double> p;double total=0;
    for(auto v:logits){p.push_back(std::exp(double(v)-top));total+=p.back();}
    for(auto& v:p)v=spire_allocation_probe::uniform(uniform)?1./p.size():.9*v/total+.1/p.size();return p;
}
struct Entry {std::string input;std::vector<std::pair<Edge*,double>> weights;};
inline Edge* choose(const Node* node,const Entry& entry,bool root) {
    if(!node||entry.weights.empty())throw std::invalid_argument("prior_missing_node");
    if(root)for(const auto& p:entry.weights)if(!p.first->visits)return p.first;
    double total=0;for(const auto& p:entry.weights)total+=p.first->sum;
    const double fpu=node->visits?total/node->visits:.5,scale=std::sqrt(double(std::max(1u,node->visits)));
    Edge* best=nullptr;double maximum=-std::numeric_limits<double>::infinity();
    for(const auto& p:entry.weights){auto* e=p.first;const double q=e->visits?e->sum/e->visits:fpu;
        const double c=spire_allocation_probe::coefficient(spire_puct_exploration::coefficient);
        const double score=c==1.?q+p.second*scale/(1.+e->visits):q+c*(p.second*scale/(1.+e->visits));
        if(!best||score>maximum){best=e;maximum=score;}}
    return best;
}
class Cache {
    std::map<const Node*,Entry> entries_;
    std::size_t bytes_=0,slots_=0,byteLimit_,slotLimit_;
public:
    unsigned hits=0,calls=0,choices=0;double inferenceSeconds=0;
    explicit Cache(std::size_t bytes=32*1024*1024,std::size_t slots=1048576):byteLimit_(bytes),slotLimit_(slots){}
    const Entry& get(Node* node,const J& legal,const std::string& key,const std::function<std::vector<double>()>& compute) {
        if(!node||!legal.is_array()||legal.empty())throw std::invalid_argument("prior_cache_input");
        std::vector<std::pair<std::string,unsigned>> order;
        for(unsigned i=0;i<legal.size();++i)order.emplace_back(legal[i].dump(),i);
        std::sort(order.begin(),order.end());auto* edge=node->edges;
        for(const auto& p:order){if(!edge||edge->action!=p.first)throw std::invalid_argument("prior_legal_alignment");edge=edge->next;}
        if(edge)throw std::invalid_argument("prior_legal_alignment");
        auto found=entries_.find(node);
        if(found!=entries_.end()){
            if(found->second.input!=key)throw std::runtime_error("prior_public_cache_mismatch");
            ++hits;return found->second;
        }
        const auto bytes=key.size()+sizeof(Entry)+128+order.size()*sizeof(std::pair<Edge*,double>);
        if(bytes>byteLimit_-bytes_||order.size()>slotLimit_-slots_)throw std::runtime_error("prior_cache_limit");
        auto at=std::chrono::steady_clock::now();auto probabilities=compute();
        inferenceSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();++calls;
        if(probabilities.size()!=legal.size())throw std::invalid_argument("prior_probability_size");
        double sum=0;for(auto p:probabilities){if(!std::isfinite(p)||p<=0||p>1)throw std::invalid_argument("prior_probability");sum+=p;}
        if(std::abs(sum-1)>1e-9)throw std::invalid_argument("prior_probability_sum");
        Entry e{key,{}};e.weights.reserve(order.size());edge=node->edges;
        for(const auto& p:order){e.weights.emplace_back(edge,probabilities[p.second]);edge=edge->next;}
        auto& result=entries_.emplace(node,std::move(e)).first->second;bytes_+=bytes;slots_+=order.size();return result;
    }
    J statistics()const{return {{"nodes",entries_.size()},{"action_slots",slots_},{"retained_bytes",bytes_},
        {"byte_limit",byteLimit_},{"slot_limit",slotLimit_},{"cache_hits",hits},{"prior_batches",calls},
        {"inference_seconds",inferenceSeconds},{"selection_calls",choices}};}
};
class Prior {
    std::string mode_;Cache cache_;spire_root_policy::Record record_;spire_value_memo::Cache memo_{memoEnabled};
public:
    explicit Prior(const J& budget):mode_(budget.value("neural_prior",std::string())) {
        if(mode_.empty())return;
        if((mode_!="learned"&&mode_!="uniform"&&mode_!="probe")||!active||
            budget.value("tree_algorithm","")!="public-history-tree-v9-standalone"||budget.value("leaf_actor","")!="trained")
            throw std::invalid_argument("prior_request_contract");
    }
    bool enabled()const{return !mode_.empty();}
    bool changes()const{return enabled()&&mode_!="probe";}
    const J& root()const{if(!active)throw std::logic_error("prior_no_public_scope");return active->root;}
    J view(const sts::BattleContext& b,const spire_rich::Memory& memory)const{return spire_rich::project(b,active->relicOrder,memory.previous);}
    spire_rich::Memory start(const sts::BattleContext& b)const {
        spire_rich::Memory m;m.observe(b);
        for(unsigned i=0;i<root().at("monsters").size();++i)m.previous[i]=root().at("monsters").at(i).at("previous_intent");
        if(view(b,m)!=root())throw std::runtime_error("prior_root_public_mismatch");return m;
    }
    Edge* select(Node* node,const J& policy,const J& legal,const J& rich) {
        if(!enabled()||!node)return nullptr;
        const auto key=J{{"policy",policy},{"rich",rich}}.dump();
        const auto& e=cache_.get(node,legal,key,[&]{
            if(mode_=="uniform")return probabilities(std::vector<float>(legal.size(),0),true);
            return probabilities(memo_.get(policy,legal,rich,[&]{return model->inspect(policy,legal,rich,false,true);}).at("logits").get<std::vector<float>>());
        });
        record_.observe(node->depth,e.weights);
        if(!changes())return nullptr;++cache_.choices;return choose(node,e,node->depth==0);
    }
    void observe(sts::BattleContext& b,spire_rich::Memory& memory,J& rich,std::vector<std::string>& observations) {
        if(!enabled())return;memory.observe(b);rich=view(b,memory);
    }
    void key(std::vector<std::string>& observations,const J& policy,const J& rich)const {
        if(changes())observations.push_back(J{{"policy",policy},{"rich",rich}}.dump());
    }
    J diagnostics()const {
        auto r=cache_.statistics();if(record_.active())r["root_policy_record"]=record_.diagnostic();r["inference_memo"]=memo_.diagnostic();
        r["inference_memo"]["version"]=memoVersion;r.update({{"version",version},{"mode",mode_},{"weights",identity},
            {"continuation_weights",continuationIdentity},{"temperature",1.},{"uniform_floor",.1},{"puct_c",spire_allocation_probe::coefficient(spire_puct_exploration::coefficient)},
            {"value_network_calls",0},{"root_schedule_before_world",true},{"pseudo_visits",0},
            {"action_pruning",false},{"neural_value_backup",false},{"scope","root-and-expanded-internal-nodes"}});return r;
    }
};
}
