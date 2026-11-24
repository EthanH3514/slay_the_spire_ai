#pragma once
// Versioned distributional combat objective. Public projections/results only.
#include "PublicRules.h"
#include <array>

namespace spire_objective {
using J=nlohmann::json;
inline constexpr auto version="combat-survival-resource-v1";
inline constexpr auto terminalVersion="combat-survival-resource-v2-terminal-hp";
inline void validateVersion(const std::string& requested) {
    if(requested!=version&&requested!=terminalVersion&&requested!="combat-survival-resource-v3-weighted-distribution")throw std::invalid_argument("combat_objective_version");
}
inline std::string name(const J& budget) {
    const auto requested=budget.value("combat_objective",std::string(version));
    validateVersion(requested);return requested;
}
using Key=std::array<double,6>;
inline constexpr std::array<const char*,6> components={"negative_loss_rate","victory_rate",
    "worst_20pct_resource_hp","p10_resource_hp","mean_resource_hp","mean_value"};
struct Outcome {
    double value=0,hp=0,resource=0;int terminal=0;
    static Outcome read(const J& j) {
        Outcome o{j.at("value"),j.at("hp"),j.at("resource_hp"),j.at("outcome")};
        if(o.terminal<0||o.terminal>3||!std::isfinite(o.value)||!std::isfinite(o.hp)||
            !std::isfinite(o.resource)||o.hp<0)throw std::invalid_argument("combat_objective_outcome");
        return o;
    }
};
inline bool enabled(const J& budget) {
    if(!budget.contains("combat_objective"))return false;
    (void)name(budget);
    return true;
}
inline J applyVersion(J row,const std::string& requested) {
    validateVersion(requested);
    (void)Outcome::read(row);
    if((requested==terminalVersion||requested=="combat-survival-resource-v3-weighted-distribution")&&row.at("outcome")==2)
        row["resource_hp"]=row.at("resource_hp").get<double>()-row.at("hp").get<double>();
    return row;
}
inline J record(const J& view,const J& root,int terminal,double value,const std::string& requested=version) {
    validateVersion(requested);
    const int hp=spire_search::number(view.at("player"),"current_hp");
    // Preserve raw public HP. Only failed-world surviving-HP credit changes;
    // inventory deltas and victory/escape/unfinished semantics stay unchanged.
    J row={{"value",value},{"hp",hp},{"outcome",terminal},
        {"resource_hp",hp+spire_search::inventoryReserve(view)-spire_search::inventoryReserve(root)}};
    return applyVersion(std::move(row),requested);
}
inline Key fromEstimate(const J& e) {
    auto resource=[&](const char* field,const char* raw) {
        return e.contains(field)&&!e.at(field).is_null()?e.at(field).get<double>():e.at(raw).get<double>();
    };
    Key k={-e.at("loss_rate").get<double>(),e.at("victory_rate").get<double>(),
        resource("worst_20pct_resource_hp","worst_20pct_mean_ending_hp"),resource("p10_resource_hp","p10_ending_hp"),
        resource("mean_resource_hp","mean_ending_hp"),e.at("mean_value").get<double>()};
    for(double x:k)if(!std::isfinite(x))throw std::invalid_argument("non-finite risk estimate");
    return k;
}
inline Key summarize(const std::vector<Outcome>& rows,int omit=-1) {
    std::vector<double> resources;resources.reserve(rows.size());
    double losses=0,wins=0,sum=0,values=0;
    for(std::size_t i=0;i<rows.size();++i)if(int(i)!=omit) {
        const auto& o=rows[i];resources.push_back(o.resource);sum+=o.resource;values+=o.value;
        losses+=o.terminal==2;wins+=o.terminal==1;
    }
    if(resources.empty())throw std::invalid_argument("combat_objective_empty_samples");
    std::sort(resources.begin(),resources.end());const double n=double(resources.size());
    const std::size_t tail=std::max<std::size_t>(1,std::size_t(std::ceil(n*.2)));
    double lower=0;for(std::size_t i=0;i<tail;++i)lower+=resources[i];
    const double position=(n-1)*.1;const auto lo=std::size_t(position),hi=std::size_t(std::ceil(position));
    // Preserve Python result aggregation's operation order, including exact
    // endpoints; algebraically equivalent interpolation can change strict ties.
    const double fraction=position-lo;
    const double p10=lo==hi?resources[lo]:resources[lo]*(1-fraction)+resources[hi]*fraction;
    return {-losses/n,wins/n,lower/tail,p10,sum/n,values/n};
}
inline int firstDifference(const Key& a,const Key& b) {
    for(int i=0;i<6;++i)if(a[i]!=b[i])return i;return -1;
}
// Fixed independent pairs. Jackknife uncertainty on the FIRST differing
// lexicographic component; this is an empirical screen, not a confidence bound.
struct PairScreen {
    std::string objectiveVersion=version;
    std::vector<Outcome> challenger,reference;
    void add(const J& alt,const J& ref) {
        if(challenger.size()>=64)throw std::runtime_error("objective_pair_limit");
        challenger.push_back(Outcome::read(alt));reference.push_back(Outcome::read(ref));
    }
    J diagnostic(unsigned required) const {
        const auto n=challenger.size();
        J out={{"version",objectiveVersion},{"complete_pairs",n},{"required_pairs",required},
            {"approved",false},{"calibrated_confidence_bound",false},{"decisive_component",nullptr},
            {"mean_difference",nullptr},{"standard_error",nullptr},{"screen_margin",nullptr}};
        if(!n)return out;
        const auto a=summarize(challenger),b=summarize(reference);const int k=firstDifference(a,b);
        out["challenger_key"]=a;out["rule_key"]=b;
        if(k<0)return out;
        const double delta=a[k]-b[k];double se=0;
        if(n>1){std::vector<double> leave;double mean=0;
            for(std::size_t i=0;i<n;++i){double d=summarize(challenger,int(i))[k]-summarize(reference,int(i))[k];
                leave.push_back(d);mean+=d/n;}
            double ss=0;for(double d:leave)ss+=(d-mean)*(d-mean);se=std::sqrt((n-1.)/n*ss);
        }
        out.update({{"decisive_component",components[k]},{"mean_difference",delta},{"standard_error",se},
            {"screen_margin",delta-2*se},{"approved",n==required&&delta-2*se>1e-12}});
        return out;
    }
    bool approve(unsigned required) const {return diagnostic(required).at("approved");}
};
} // namespace spire_objective
