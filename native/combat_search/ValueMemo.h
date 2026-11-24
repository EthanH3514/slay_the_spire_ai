#pragma once
// Exact memoization of a deterministic model, scoped to one search request.
// Full inputs are compared; no state aggregation, hashes, hidden state or RNG.
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <string>
#include <cstddef>

namespace spire_value_memo {
using J=nlohmann::json;
inline constexpr const char* version="exact-public-value-memo-v1";
inline thread_local bool enabled=false;
class Cache {
    bool enabled_;
    std::size_t bytes_=0, limit_;
    unsigned hits_=0, calls_=0, skipped_=0;
    std::unordered_map<std::string,J> entries_;
public:
    explicit Cache(bool enabled=spire_value_memo::enabled,std::size_t bytes=16*1024*1024)
        :enabled_(enabled),limit_(bytes){}
    template<class Compute> J get(const J& view,const J& legal,const J& rich,Compute compute) {
        if(!enabled_){++calls_;return compute();}
        auto key=J::array({view,legal,rich}).dump();
        auto found=entries_.find(key);
        if(found!=entries_.end()){++hits_;return found->second;}
        ++calls_;auto result=compute();
        const auto size=key.size()+result.dump().size()+128;
        if(entries_.size()<1024&&size<=limit_-bytes_){bytes_+=size;entries_.emplace(std::move(key),result);}
        else ++skipped_;
        return result;
    }
    J diagnostic()const{return {{"version",version},{"enabled",enabled_},{"network_evaluations",calls_},
        {"hits",hits_},{"entries",entries_.size()},{"accounted_bytes",bytes_},{"byte_limit",limit_},
        {"skipped_insertions",skipped_},{"scope","single_search"}};}
};
}
