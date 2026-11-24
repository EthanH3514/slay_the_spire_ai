#pragma once
// Explicit experimental model binding; never part of the production default.
#include "LeafActorFast.h"
namespace spire_cold {
inline constexpr auto version="public-cold-leaf-v1";
inline thread_local std::unique_ptr<spire_actor_fast::Actor> actor;
inline thread_local std::string identity;
inline void bind(const nlohmann::json& weights,const nlohmann::json& schema,const std::string& id){
    if(id.size()!=64)throw std::invalid_argument("cold_weight_identity");
    auto next=std::make_unique<spire_actor_fast::Actor>(weights,schema);
    actor=std::move(next);identity=id;
}
inline nlohmann::json choose(const nlohmann::json& view,const nlohmann::json& legal){
    if(!actor)throw std::runtime_error("cold_actor_not_loaded");return actor->choose(view,legal);
}
inline nlohmann::json leaf(const nlohmann::json& view,const nlohmann::json& legal,
        const nlohmann::json& prior,const std::string& mode,spire_actor::Counts& counts){
    if(mode!="trained"&&mode!="probe")throw std::invalid_argument("cold_actor_mode");
    ++counts.calls;counts.selections+=view.at("input")=="CARD_SELECT";counts.paused+=view.at("resolved")==false;
    const auto action=choose(view,legal);if(mode=="probe")return prior;
    counts.changed+=action!=prior;return action;
}
}
