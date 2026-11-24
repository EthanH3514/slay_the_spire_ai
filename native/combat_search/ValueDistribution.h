#pragma once
// Public mathematical evidence only: no simulator or hidden-state access.
#include "CombatObjective.h"
#include "ResourceUtility.h"
namespace spire_value_distribution {
using J=nlohmann::json;
inline constexpr auto version="combat-survival-resource-v3-weighted-distribution";
struct Atom {double mass,resource,value;int outcome;};
struct Distribution {
    bool predicted=false;std::vector<Atom> atoms;J evidence;
    double meanValue()const{double v=0;for(const auto& a:atoms)v+=a.mass*a.value;return v;}
    void check()const {
        if(atoms.empty()||atoms.size()>63)throw std::invalid_argument("value_atom_count");
        double mass=0;for(const auto& a:atoms){
            if(!std::isfinite(a.mass)||!std::isfinite(a.resource)||!std::isfinite(a.value)||a.mass<0||a.outcome<1||a.outcome>3)
                throw std::invalid_argument("value_atom");mass+=a.mass;}
        if(std::abs(mass-1)>1e-8)throw std::invalid_argument("value_mass");
    }
};
inline double scalar(int outcome,double total,double maxhp) {
    if(!std::isfinite(total)||!std::isfinite(maxhp)||maxhp<=0||(spire_resource_utility::linear&&total<0))throw std::invalid_argument("value_resource");
    if(outcome==2)return -3.;
    if(outcome!=1&&outcome!=3)throw std::invalid_argument("value_not_terminal");
    return (outcome==1?2.:.5)+spire_resource_utility::resource(total,maxhp);
}
inline Distribution terminal(J row,double rootReserve,double maxhp) {
    const auto o=spire_objective::Outcome::read(row);
    const double value=scalar(o.terminal,o.resource+rootReserve,maxhp);
    row["value"]=value;row["source"]="terminal";
    if(spire_resource_utility::linear)row.update({{"resource_utility",spire_resource_utility::version},
        {"root_reserve",rootReserve},{"max_hp",maxhp}});
    Distribution d{false,{{1.,o.resource,value,o.terminal}},row};d.check();return d;
}
inline Distribution prediction(const J& raw,double reserve,double maxhp) {
    if(!std::isfinite(reserve)||reserve<0||!std::isfinite(maxhp)||maxhp<=0)throw std::invalid_argument("value_root_resource");
    const auto& logits=raw.at("outcome_logits");const auto& q=raw.at("resource_quantiles");
    if(!logits.is_array()||logits.size()!=3||!q.is_array()||q.size()!=3)throw std::invalid_argument("value_shape");
    std::array<double,3> p={logits[0],logits[1],logits[2]};
    for(double x:p)if(!std::isfinite(x))throw std::invalid_argument("value_logits");
    double top=*std::max_element(p.begin(),p.end()),total=0;
    for(double& x:p){x=std::exp(x-top);total+=x;}for(double& x:p)x/=total;
    Distribution d;d.predicted=true;
    d.evidence={{"source","model"},{"probabilities",p},{"resource_quantiles",q},{"root_reserve",reserve},{"max_hp",maxhp}};
    if(spire_resource_utility::linear)d.evidence["resource_utility"]=spire_resource_utility::version;
    for(int k=0;k<3;++k){
        if(!q[k].is_array()||q[k].size()!=21)throw std::invalid_argument("value_quantile_shape");
        double last=0;for(const auto& x:q[k]){double v=x;
            if(!std::isfinite(v)||v<last)throw std::invalid_argument("value_quantile_order");last=v;
            const int outcome=std::array<int,3>{2,1,3}[k];d.atoms.push_back({p[k]/21.,v*100.-reserve,scalar(outcome,v*100.,maxhp),outcome});}
    }
    d.check();return d;
}
inline spire_objective::Key summarize(const std::vector<Distribution>& rows) {
    if(rows.empty())throw std::invalid_argument("value_no_distributions");
    std::vector<std::pair<double,double>> atoms;double losses=0,wins=0,sum=0,values=0;
    for(const auto& r:rows)for(const auto& a:r.atoms){if(a.mass==0)continue;
        atoms.emplace_back(a.resource,a.mass);losses+=a.mass*(a.outcome==2);wins+=a.mass*(a.outcome==1);
        sum+=a.mass*a.resource;values+=a.mass*a.value;}
    std::sort(atoms.begin(),atoms.end());const double n=rows.size();
    double remaining=.2*n,lower=0,cumulative=0,p10=0;bool found=false;
    for(const auto& a:atoms){double take=std::min(remaining,a.second);lower+=take*a.first;remaining-=take;cumulative+=a.second;
        if(!found&&cumulative>=.1*n-1e-12){p10=a.first;found=true;}}
    if(remaining>1e-8||!found)throw std::invalid_argument("value_tail_mass");
    return {-losses/n,wins/n,lower/(.2*n),p10,sum/n,values/n};
}
}
