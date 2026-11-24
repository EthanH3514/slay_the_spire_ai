#pragma once
#include "CombatObjective.h"

namespace spire_tree {
using J=nlohmann::json;
inline constexpr auto rootRecencyVersion="public-history-tree-v16-root-recency";
inline constexpr auto rootRecencyEstimator="positive-linear-visit-rank-frequency-v1";
// Weighted frequency distribution. No virtual-row expansion or simulator access.
inline spire_objective::Key recencyKey(const std::vector<spire_objective::Outcome>& rows) {
    if(rows.empty()||rows.size()>256)throw std::invalid_argument("root_recency_samples");
    std::vector<std::pair<double,unsigned>> sorted;sorted.reserve(rows.size());
    unsigned weight=0;double losses=0,wins=0,resources=0,values=0;
    for(unsigned i=0;i<rows.size();++i){const auto& r=rows[i];const unsigned w=i+1;
        if(!std::isfinite(r.value)||!std::isfinite(r.hp)||!std::isfinite(r.resource)||r.hp<0||r.terminal<0||r.terminal>3)
            throw std::invalid_argument("root_recency_outcome");
        weight+=w;losses+=w*(r.terminal==2);wins+=w*(r.terminal==1);
        resources+=w*r.resource;values+=w*r.value;sorted.emplace_back(r.resource,w);
    }
    std::sort(sorted.begin(),sorted.end());
    const unsigned tail=std::max(1u,unsigned(std::ceil(weight*.2)));unsigned remaining=tail;double lower=0;
    for(const auto& r:sorted){const unsigned take=std::min(remaining,r.second);lower+=take*r.first;remaining-=take;if(!remaining)break;}
    auto at=[&](unsigned k){for(const auto& r:sorted){if(k<r.second)return r.first;k-=r.second;}
        throw std::logic_error("root_recency_quantile");};
    const double position=(weight-1)*.1;const auto lo=unsigned(position),hi=unsigned(std::ceil(position));
    const double fraction=position-lo,p10=lo==hi?at(lo):at(lo)*(1-fraction)+at(hi)*fraction;
    return {-losses/weight,wins/weight,lower/tail,p10,resources/weight,values/weight};
}
inline J recencyEvidence(const std::vector<spire_objective::Outcome>& rows) {
    if(rows.empty())return J();
    const auto key=recencyKey(rows);const unsigned n=rows.size(),w=n*(n+1)/2,sq=n*(n+1)*(2*n+1)/6;
    return {{"estimator",rootRecencyEstimator},{"actual_samples",n},{"frequency_weight",w},
        {"effective_samples",double(w)*w/sq},{"objective_key",key}};
}
} // namespace spire_tree
