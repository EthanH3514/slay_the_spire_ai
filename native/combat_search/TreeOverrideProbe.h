#pragma once
// Diagnostic-only contract. Never passed by a production search policy.
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace spire_tree {
inline constexpr auto overrideProbeVersion="tree-first-override-probe-v1";
struct OverrideProbe {
    using J=nlohmann::json;
    bool enabled=false, useRule=false, ruleSuffix=false;
    int suffixSteps=0;
    J expected;
    static OverrideProbe parse(const J* input,bool detail,bool guarded,const J& seconds) {
        OverrideProbe p;if(!input)return p;
        if(!detail||!guarded||!seconds.is_null())
            throw std::invalid_argument("override_probe_requires_guarded_trace_and_external_watchdog");
        const auto& j=*input;
        if(!j.is_object()||j.size()!=5||j.at("version")!=overrideProbeVersion||
           !j.at("expected").is_object()||!j.at("max_suffix_steps").is_number_integer()||
           j.at("max_suffix_steps")<1||j.at("max_suffix_steps")>512||
           (j.at("first_action")!="accepted"&&j.at("first_action")!="rule")||
           (j.at("continuation")!="frozen"&&j.at("continuation")!="rule"))
            throw std::invalid_argument("invalid_override_probe");
        p.enabled=true;p.expected=j.at("expected");p.suffixSteps=j.at("max_suffix_steps");
        p.useRule=j.at("first_action")=="rule";p.ruleSuffix=j.at("continuation")=="rule";
        return p;
    }
};
} // namespace spire_tree
