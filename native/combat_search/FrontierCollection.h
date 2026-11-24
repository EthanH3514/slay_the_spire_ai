#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace spire_frontier {
using J=nlohmann::json;
inline constexpr auto version="public-terminal-frontiers-v2-escape";
inline std::uint64_t priority(std::uint64_t salt,unsigned index){
    std::uint64_t x=salt+0x9e3779b97f4a7c15ULL*(std::uint64_t(index)+1);
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
class Collector {
    bool active_;std::uint64_t salt_=0;unsigned index_=0;
    struct Record{std::uint64_t rank;unsigned index;J row;std::size_t bytes;};
    std::vector<Record> records_;J pending_;J eligible_=J::array();
    std::size_t bytes_=0,peak_=0;unsigned completed_=0;
public:
    explicit Collector(const J& budget):active_(budget.contains("frontier_collection")){
        if(!active_)return;
        const auto& c=budget.at("frontier_collection");
        if(!c.is_object()||c.size()!=2||!c.contains("version")||!c.contains("salt")||c.at("version")!=version||
            !c.at("salt").is_number_unsigned()||budget.value("value_leaf","")!="terminal"||
            budget.value("rollout_horizon",0)!=192||budget.value("leaf_actor","")!="trained"||
            budget.at("tree").value("max_depth",0)!=12)
            throw std::invalid_argument("frontier_collection_contract");
        salt_=c.at("salt").get<std::uint64_t>();
    }
    bool enabled()const{return active_;}
    void begin(unsigned index){
        if(!active_)return;
        if(!pending_.is_null()||index>=1024)throw std::runtime_error("frontier_unfinished_simulation");
        index_=index;
    }
    void observe(unsigned depth,const J& view,const J& legal,const J& rich){
        if(!active_||depth!=16)return;
        if(!pending_.is_null()||legal.empty()||eligible_.size()>=1024)throw std::runtime_error("frontier_snapshot_contract");
        const auto rank=priority(salt_,index_);
        eligible_.push_back({{"simulation",index_},{"priority",rank}});
        if(records_.size()==8&&std::make_pair(rank,index_)>=std::make_pair(records_.back().rank,records_.back().index))return;
        pending_={{"simulation",index_},{"priority",rank},{"depth",16},{"view",view},{"legal",legal},{"rich",rich}};
        if(pending_.dump().size()>1024*1024)throw std::runtime_error("frontier_snapshot_byte_limit");
        peak_=std::max(peak_,bytes_+pending_.dump().size());
        if(peak_>8*1024*1024)throw std::runtime_error("frontier_total_byte_limit");
    }
    void finish(const J& terminal){
        if(!active_)return;
        if(terminal.value("version","")!="public-battle-result-v1"||
            (terminal.value("outcome",0)<1||terminal.value("outcome",0)>3)||
            !terminal.contains("hp")||!terminal.contains("potion_reserve"))
            throw std::runtime_error("frontier_requires_observed_terminal");
        ++completed_;if(pending_.is_null())return;
        pending_["terminal"]=terminal;const auto size=pending_.dump().size();
        if(size>1024*1024)throw std::runtime_error("frontier_snapshot_byte_limit");
        if(records_.size()==8){bytes_-=records_.back().bytes;records_.pop_back();}
        records_.push_back({priority(salt_,index_),index_,std::move(pending_),size});pending_=J();bytes_+=size;
        std::sort(records_.begin(),records_.end(),[](const Record& a,const Record& b){return std::make_pair(a.rank,a.index)<std::make_pair(b.rank,b.index);});
        peak_=std::max(peak_,bytes_);if(peak_>8*1024*1024)throw std::runtime_error("frontier_total_byte_limit");
    }
    J report()const{
        J rows=J::array();for(const auto& r:records_)rows.push_back(r.row);
        return {{"version",version},{"salt",salt_},{"eligible",eligible_},{"records",rows},{"completed",completed_},
            {"retained_bytes",bytes_},{"peak_bytes",peak_},{"record_limit",8},{"byte_limit",8*1024*1024},
            {"pending",!pending_.is_null()},{"labels_are_predictions",false}};
    }
};
}
