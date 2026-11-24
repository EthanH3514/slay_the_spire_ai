#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

// Opt-in, one-factor allocation experiments. No model, backup or final-selector
// changes. The default path preserves the frozen controller's arithmetic.
namespace spire_allocation_probe {
using J=nlohmann::json;
inline constexpr auto version="public-allocation-probe-v1";
inline thread_local std::string mode;
inline thread_local bool active=false;
inline bool uniform(bool original) {return original||mode=="uniform";}
inline double coefficient(double original) {return mode=="lower_c"?.1:original;}
struct Scope {
    bool enabled=false;
    explicit Scope(const J& budget) {
        if(active)throw std::invalid_argument("allocation_probe_nested");
        std::string next;
        if(budget.contains("allocation_probe")) {
            const auto& c=budget.at("allocation_probe");
            if(!c.is_object()||c.size()!=2||c.value("version","")!=version||
               !c.contains("mode")||!c.at("mode").is_string())
                throw std::invalid_argument("allocation_probe_contract");
            next=c.at("mode").get<std::string>();
            if(next!="control"&&next!="lower_c"&&next!="uniform")
                throw std::invalid_argument("allocation_probe_mode");
            if(!budget.contains("max_seconds")||!budget.at("max_seconds").is_null()||
               budget.value("neural_prior","")!="learned"||budget.value("value_leaf","")!="learned"||
               budget.at("boundary_value").value("mode","")!="direct"||
               budget.at("root_recommendation").value("mode","")!="visits"||
               budget.at("puct_exploration").at("coefficient")!=.5)
                throw std::invalid_argument("allocation_probe_requires_frozen_network_controller");
            enabled=true;
        }
        mode=next;active=true;
    }
    ~Scope(){mode.clear();active=false;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
    J receipt()const {return {{"version",version},{"mode",mode},
        {"coefficient",coefficient(.5)},{"uniform_prior",mode=="uniform"},
        {"scope","root-and-internal-PUCT"},{"weights_changed",false},
        {"backup_changed",false},{"final_selector_changed",false}};}
};
}
