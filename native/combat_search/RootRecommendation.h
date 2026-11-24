#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace spire_root_recommendation {
using J=nlohmann::json;
inline constexpr auto version="public-root-recommendation-v1";
inline constexpr auto visitsObjective="completed-visits-then-distribution-objective-v1";
struct Config {
    bool enabled=false,visits=false;
    explicit Config(const J& budget) {
        if(!budget.contains("root_recommendation"))return;
        const auto& j=budget.at("root_recommendation");
        if(!j.is_object()||j.size()!=2||j.value("version","")!=version||
           !j.contains("mode")||!j.at("mode").is_string())
            throw std::invalid_argument("root_recommendation_contract");
        const auto mode=j.at("mode").get<std::string>();
        if(mode!="score"&&mode!="visits")throw std::invalid_argument("root_recommendation_mode");
        if(!budget.contains("boundary_value")||budget.at("boundary_value").value("mode","")!="direct"||
           budget.value("value_leaf","")!="learned"||budget.value("neural_prior","")!="learned")
            throw std::invalid_argument("root_recommendation_requires_network_only");
        enabled=true;visits=mode=="visits";
    }
    J receipt()const {
        return {{"version",version},{"mode",visits?"visits":"score"},
            {"scope","final-root-only"},{"visits","completed-backups-not-distribution-atoms"},
            {"tie_breaker","distribution-key-then-canonical-action"},{"internal_selection_unchanged",true}};
    }
};
// Called only by final recommendation. Tree selection and backup never consult
// this option. Equal keys retain the first edge in canonical action order.
template<class Key>
inline bool better(bool visitsFirst,unsigned visits,const Key& key,unsigned bestVisits,const Key& bestKey) {
    if(visitsFirst&&visits!=bestVisits)return visits>bestVisits;
    return key>bestKey;
}
} // namespace spire_root_recommendation
