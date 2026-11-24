#pragma once

// A new artifact envelope separates target provenance from the old actor ABI.
// It does not change inference, PUCT, backups, proof eligibility or budgets.
namespace spire_search_value {
using J=nlohmann::json;
inline constexpr auto abi="public-search-value-v1";
inline thread_local J target=nullptr;

inline bool sha256(const J& s) {
    if(!s.is_string())return false;
    const auto& x=s.get_ref<const std::string&>();
    return x.size()==64&&std::all_of(x.begin(),x.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}

inline J binding(const J& e) {
    if(e.at("version")!="public-search-value-v1") {
        spire_policy_value::keys(e,{"version","model","continuation"},{"version","model","continuation"});
        const auto& c=e.at("continuation");
        spire_policy_value::keys(c,{"kind","weights_sha256"},{"kind","weights_sha256"});
        if(e.at("version")!="public-greedy-value-v2-escape"||c.at("kind")!="actor_greedy")
            throw std::invalid_argument("value_legacy_target");
        target=nullptr;return c;
    }
    spire_policy_value::keys(e,{"version","model","binding","target"},{"version","model","binding","target"});
    const auto& b=e.at("binding");const auto& t=e.at("target");
    spire_policy_value::keys(b,{"kind","weights_sha256"},{"kind","weights_sha256"});
    spire_policy_value::keys(t,{"kind","controller","controller_sha256"},{"kind","controller","controller_sha256"});
    if(b.at("kind")!="legacy-actor-abi-only"||!sha256(b.at("weights_sha256"))||
       t.at("kind")!="frozen-closed-loop-search-terminal-v1"||!sha256(t.at("controller_sha256"))||!t.at("controller").is_object())
        throw std::invalid_argument("value_search_target");
    const auto& c=t.at("controller");
    if(c.value("steps",0)!=16384||c.value("simulations",0)!=1024||c.value("coefficient",0.)!=.5||
       c.value("old_actor_execution",true)||c.value("terminal_proof",std::string())!="public-terminal-proof-v1"||
       c.value("fallback",std::string())!="refuse")throw std::invalid_argument("value_search_teacher");
    target=t;return b;
}
}
