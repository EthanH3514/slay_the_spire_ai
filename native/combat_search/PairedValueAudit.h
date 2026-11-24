#pragma once
// Diagnostic-only ownership of sampled worlds. No real RNG or private state is
// exported. Never clone a paused queue: MOVE the completed search branch.
#include "ConditionalRoots.h"
#include "ColdLeaf.h"
#include "FrontierCollection.h"
#include <map>
namespace spire_value_audit {
using J=nlohmann::json;
inline constexpr auto version="paired-value-frontiers-v1";
class Audit {
    struct Entry {
        std::uint64_t priority; unsigned index; J row;
        spire_search::SearchRoot world;
    };
    bool active_; std::uint64_t salt_=0; unsigned seen_=0;
    std::map<std::string,std::vector<Entry>> groups_;
    std::map<std::string,unsigned> ordinals_;
    J eligible_=J::array();
public:
    explicit Audit(const J& budget):active_(budget.contains("paired_value_audit")) {
        if(!active_)return;
        const auto& a=budget.at("paired_value_audit");
        if(a.size()!=5||a.value("version","")!=version||!a.at("salt").is_number_unsigned()||
            a.value("per_action",0)!=4||a.value("action_limit",0)!=32||a.value("continuation_cap",0)!=768||
            budget.value("value_leaf","")!="learned"||!budget.at("max_seconds").is_null()||
            budget.value("rollout_horizon",0)!=16||budget.value("leaf_actor","")!="trained")
            throw std::invalid_argument("paired_value_audit_contract");
        salt_=a.at("salt").get<std::uint64_t>();
    }
    bool enabled()const{return active_;}
    void observe(unsigned index,const std::string& action,const J& prediction,const J& view,
                 const J& legal,const J& rich,spire_search::SearchRoot& world) {
        if(!active_)return;
        const unsigned ordinal=ordinals_[action]++;
        if(prediction.at("source")!="model")return;
        if(++seen_>1024||legal.empty()||spire_search::outcome(*world.battle))
            throw std::runtime_error("paired_value_audit_nonfrontier");
        const auto rank=spire_frontier::priority(salt_,index);
        eligible_.push_back({{"simulation",index},{"action",J::parse(action)},{"ordinal",ordinal},{"priority",rank}});
        auto& group=groups_[action];
        if(groups_.size()>32)throw std::runtime_error("paired_value_audit_action_limit");
        if(group.size()==4 && std::make_pair(rank,index)>=std::make_pair(group.back().priority,group.back().index))return;
        J row={{"simulation",index},{"action",J::parse(action)},{"ordinal",ordinal},{"priority",rank},
               {"view",view},{"legal",legal},{"rich",rich},{"prediction",prediction}};
        if(row.dump().size()>1024*1024)throw std::runtime_error("paired_value_audit_public_byte_limit");
        if(group.size()==4)group.pop_back();
        group.push_back({rank,index,std::move(row),std::move(world)});
        std::sort(group.begin(),group.end(),[](const Entry& a,const Entry& b){
            return std::make_pair(a.priority,a.index)<std::make_pair(b.priority,b.index);});
    }
    // Called only AFTER search recommendation and all search evidence are saved.
    // All extra actions are counted here, never in the production step budget.
    J finish(spire_search::ConditionalRoots& roots,const J& metadata) {
        J rows=J::array();unsigned steps=0,censored=0; auto started=std::chrono::steady_clock::now();
        const auto& input=roots.evidence(); spire_actor::Counts counts;
        for(auto& group:groups_)for(auto& e:group.second) {
            auto& w=e.world; J row=e.row;unsigned used=0;
            try {
                auto project=[&]{return roots.projectView(*w.battle,input.relics,w.knownTop,input.character);};
                auto view=project();
                if(view!=row.at("view")||spire_search::actions(*w.battle,true)!=row.at("legal"))
                    throw std::runtime_error("paired_value_audit_snapshot_changed");
                while(!spire_search::outcome(*w.battle)&&used<768) {
                    auto legal=spire_search::actions(*w.battle,true);
                    spire_search::guard(view,metadata);
                    auto prior=spire_search::combatChoice(view,legal,metadata);
                    auto action=spire_cold::leaf(view,legal,prior,"trained",counts);
                    if(std::find(legal.begin(),legal.end(),action)==legal.end())
                        throw std::runtime_error("paired_value_audit_illegal_continuation");
                    roots.advancePublic(w,action);++used;++steps;view=project();
                }
                const int terminal=spire_search::outcome(*w.battle);
                row["terminal"]=terminal?J{{"version","public-battle-result-v1"},{"outcome",terminal},
                    {"hp",spire_search::number(view.at("player"),"current_hp",0)},
                    {"potion_reserve",spire_search::inventoryReserve(view)}}:J();
                row["status"]=terminal?"terminal":"censored";censored+=!terminal;
            }catch(const std::exception& error){row["status"]="failed";row["error"]=error.what();}
            row["continuation_steps"]=used;rows.push_back(std::move(row));
        }
        return {{"version",version},{"salt",salt_},{"eligible",eligible_},{"records",rows},
            {"per_action",4},{"record_limit",128},{"continuation_cap",768},{"censored",censored},
            {"continuation_steps",steps},{"actor_counts",counts.json()},{"continuation_weights",spire_cold::identity},
            {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},
            {"labels_used_by_search",false},{"same_sampled_world",true}};
    }
};
}
