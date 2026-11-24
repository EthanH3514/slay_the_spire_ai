#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace spire_search {
// Preserve the existing bytewise JSON order (NOT numeric JSON comparison).
// Encode each element once, before moving any value. Scratch is local to this
// projection; nothing survives a state advance or leaks across sampled worlds.
inline void sortByJsonEncoding(nlohmann::json& value) {
    auto& values=value.get_ref<nlohmann::json::array_t&>();
    if(values.size()<2)return;
    std::vector<std::pair<std::string,std::size_t>> keys;keys.reserve(values.size());
    for(std::size_t i=0;i<values.size();++i)keys.emplace_back(values[i].dump(),i);
    std::sort(keys.begin(),keys.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    nlohmann::json::array_t ordered;ordered.reserve(values.size());
    for(const auto& key:keys)ordered.push_back(std::move(values[key.second]));
    values=std::move(ordered);
}
} // namespace spire_search
