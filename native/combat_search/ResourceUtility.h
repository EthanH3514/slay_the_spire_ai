#pragma once
// Utility is mathematical public evidence; no simulator or RNG access.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spire_resource_utility {
inline constexpr auto version="combat-resource-utility-v2-linear";
inline thread_local bool linear=false;

class Scope {
    bool previous_;
public:
    bool enabled;
    explicit Scope(const nlohmann::json& budget):previous_(linear),enabled(budget.contains("resource_utility")) {
        if(enabled&&(budget.at("resource_utility")!=version||budget.value("value_leaf","")!="learned"))
            throw std::invalid_argument("resource_utility_request");
        linear=enabled;
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
    ~Scope(){linear=previous_;}
};

inline double resource(double total,double maxhp) {
    if(!std::isfinite(total)||!std::isfinite(maxhp)||maxhp<=0||(linear&&total<0))
        throw std::invalid_argument("value_resource");
    const double value=total/maxhp;
    if(!std::isfinite(value))throw std::invalid_argument("value_resource_ratio");
    return linear?value:std::clamp(value,0.,1.);
}

inline double backup(double value) {
    if(!std::isfinite(value)||(linear&&value< -3.))throw std::invalid_argument("value_utility");
    const double normalized=(value+3.)/6.;
    return linear?normalized:std::clamp(normalized,0.,1.);
}
}
