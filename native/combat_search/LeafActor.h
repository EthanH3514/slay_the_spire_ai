#pragma once
// Experimental immutable public-input actor. No simulator, RNG or Python API.
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace spire_actor {
using J=nlohmann::json;
inline const std::vector<std::string> stateGroups={"global","hand","draw","discard","exhaust","held","top_candidates","selection_candidates","monsters","player_powers","relics","potions"};
inline const std::vector<std::string> actionGroups={"action","cards","potion","target"};
inline const std::set<std::string> forbidden={"seed","seed_numeric","seed_text","reset_seed","audit_state","backend_features","rng","rng_state","rng_streams","streams","world_words","source","run_group","root_evidence","future","label","outcome"};
inline std::string key(const std::string& value){std::string s;for(unsigned char c:value){if(c>=128)throw std::invalid_argument("actor_ascii_identifier");if(std::isalnum(c))s+=char(std::tolower(c));}return s;}
inline void publicOnly(const J& value){
    if(value.is_object())for(auto i=value.begin();i!=value.end();++i){if(forbidden.count(i.key()))throw std::invalid_argument("actor_private_channel");publicOnly(i.value());}
    else if(value.is_array())for(const auto& x:value)publicOnly(x);
    else if(value.is_number_float()&&!std::isfinite(value.get<double>()))throw std::invalid_argument("actor_nonfinite_input");
}
inline void keys(const J& value,const std::set<std::string>& allowed,const std::set<std::string>& required){
    if(!value.is_object())throw std::invalid_argument("actor_object");
    for(auto i=value.begin();i!=value.end();++i)if(!allowed.count(i.key()))throw std::invalid_argument("actor_unknown_field:"+i.key());
    for(const auto& k:required)if(!value.contains(k))throw std::invalid_argument("actor_missing_field:"+k);
}
inline int index(const J& value,const J& array){
    if(!value.is_number_integer()||!array.is_array()||value.get<long long>()<0||value.get<unsigned long long>()>=array.size())throw std::invalid_argument("actor_action_reference");
    return value.get<int>();
}
struct Row {
    std::array<int,4> categories{};std::array<float,32> values{},known{};
    bool operator<(const Row& r)const{return std::tie(categories,values,known)<std::tie(r.categories,r.values,r.known);}
    J json()const{return J{{"categories",categories},{"values",values},{"known",known}};}
};
using Group=std::vector<Row>;
struct Encoded {
    std::vector<Group> state;std::vector<std::vector<Group>> candidates;
    J unknown=J::object();unsigned rows=0;
    J json(const std::string& schema)const{
        auto groups=[](const std::vector<Group>& g,const std::vector<std::string>& names){J out=J::object();for(std::size_t i=0;i<g.size();++i){out[names[i]]=J::array();for(const auto& r:g[i])out[names[i]].push_back(r.json());}return out;};
        J cs=J::array();for(const auto& c:candidates)cs.push_back(groups(c,actionGroups));
        return J{{"schema_id",schema},{"state",groups(state,stateGroups)},{"candidates",cs},
            {"mask",std::vector<bool>(candidates.size(),true)},{"unknown",unknown},{"rows",rows}};
    }
};

class Actor {
    J schema_;std::map<std::string,int> tokens_;
    std::map<std::string,std::vector<float>> weights_;
    static constexpr unsigned W=32;
    using V=std::array<float,W>;
    int token(const std::string& category,const std::string& value,Encoded& out)const{
        if(value.empty())return 0;
        const auto name=category+":"+(category=="role"?value:key(value));auto it=tokens_.find(name);
        if(it!=tokens_.end())return it->second;
        out.unknown[name]=out.unknown.value(name,0)+1;return 1;
    }
    Row row(const std::string& role,const std::vector<std::pair<std::string,std::string>>& cats,const J& source,Encoded& out)const{
        if(++out.rows>32768)throw std::invalid_argument("actor_row_cap");
        Row r;r.categories[0]=token("role",role,out);
        for(std::size_t i=0;i<cats.size();++i)r.categories[i+1]=token(cats[i].first,cats[i].second,out);
        int slot=0;for(const auto& field:schema_.at("fields").at(role)){
            const auto name=field.at(0).get<std::string>();
            if(source.contains(name)){
                const auto& x=source.at(name);if(!x.is_number()&&!x.is_boolean())throw std::invalid_argument("actor_numeric_scalar");
                const double v=(x.is_boolean()?(x.get<bool>()?1.:0.):x.get<double>())/field.at(1).get<double>();
                if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<float>::max())throw std::invalid_argument("actor_numeric_range");
                r.values.at(slot)=float(v);r.known.at(slot)=1.f;
            }++slot;
        }return r;
    }
    Row card(const J& c,Encoded& out,int order=-1)const{
        J data=c;if(order>=0)data["order"]=order;
        return row("card",{{"card",c.at("id")},{"type",c.at("type")}},data,out);
    }
    Group monster(const J& m,Encoded& out)const{
        Group result{row("monster",{},m,out)};
        for(auto p=m.at("powers").begin();p!=m.at("powers").end();++p)
            result.push_back(row("monster_power",{{"power",p.key()}},J{{"amount",p.value()}},out));
        return result;
    }
    std::vector<float> linear(const std::string& name,const std::vector<float>& x,unsigned count,bool tanh)const{
        const auto& w=weights_.at(name+".weight");const auto& b=weights_.at(name+".bias");
        if(w.size()!=count*x.size()||b.size()!=count)throw std::runtime_error("actor_linear_shape");
        std::vector<float> y(count);
        for(unsigned i=0;i<count;++i){float sum=b[i];for(std::size_t j=0;j<x.size();++j)sum+=w[i*x.size()+j]*x[j];
            y[i]=tanh?std::tanh(sum):sum;if(!std::isfinite(y[i]))throw std::runtime_error("actor_nonfinite_logit");}
        return y;
    }
public:
    Actor(const J& artifact,const J& expected):schema_(expected){
        const std::set<std::string> fields={"version","schema","tensors","dtype","architecture","critic"};keys(artifact,fields,fields);
        if(artifact.at("version")!="public-leaf-actor-weights-v1"||artifact.at("schema")!=expected||
           artifact.at("dtype")!="float32"||artifact.at("architecture")!="combat_entity_candidate_actor_v1"||
           !artifact.at("critic").is_boolean()||artifact.at("critic").get<bool>())throw std::invalid_argument("actor_artifact_contract");
        if(expected.at("version")!="public-leaf-entities-v1"||expected.at("numbers")!=32||expected.at("categories")!=4||
           expected.at("width")!=32||expected.at("embedding_width")!=8||expected.at("state_groups")!=stateGroups||expected.at("action_groups")!=actionGroups||
           expected.at("max_rows")!=32768||expected.at("max_actions")!=4096)throw std::invalid_argument("actor_schema_contract");
        for(const auto& t:expected.at("vocab")){const auto name=t.get<std::string>();if(!tokens_.emplace(name,int(tokens_.size())).second)throw std::invalid_argument("actor_duplicate_token");}
        if(tokens_.at("<pad>")!=0||tokens_.at("<unknown>")!=1)throw std::invalid_argument("actor_padding_contract");
        const std::map<std::string,std::vector<int>> shapes={{"embedding.weight",{int(tokens_.size()),8}},
            {"row.weight",{32,96}},{"row.bias",{32}},{"context.weight",{32,396}},{"context.bias",{32}},
            {"head.weight",{32,164}},{"head.bias",{32}},{"output.weight",{1,32}},{"output.bias",{1}}};
        if(artifact.at("tensors").size()!=shapes.size())throw std::invalid_argument("actor_tensor_names");
        for(const auto& p:shapes){const auto& t=artifact.at("tensors").at(p.first);keys(t,{"shape","values"},{"shape","values"});
            if(t.at("shape")!=p.second||!t.at("values").is_array())throw std::invalid_argument("actor_tensor_shape");
            unsigned n=1;for(int k:p.second)n*=k;if(t.at("values").size()!=n)throw std::invalid_argument("actor_tensor_size");
            auto& dst=weights_[p.first];dst.reserve(n);for(const auto& v:t.at("values")){
                if(!v.is_number()||!std::isfinite(v.get<double>())||std::abs(v.get<double>())>std::numeric_limits<float>::max())throw std::invalid_argument("actor_tensor_value");
                dst.push_back(v.get<float>());}
        }
        for(int i=0;i<8;++i)if(weights_.at("embedding.weight")[i]!=0)throw std::invalid_argument("actor_pad_embedding");
    }
    const J& schema()const{return schema_;}
    Encoded encode(const J& v,const J& legal)const{
        publicOnly(v);publicOnly(legal);
        const std::set<std::string> viewKeys={"bomb_schema","bottom_tracked","character","discard","draw","exhaust","hand","held","input","limbo_empty",
            "monsters","opening_priority_ids","opening_schema","opening_uncertain","player","potions","power_records","powers_complete","relics","resolved","selection",
            "selection_any","selection_candidates","selection_copies","selection_max","selection_options","selection_pick","selection_type","times_damaged","top_candidates","uncertain_order"};
        keys(v,viewKeys,viewKeys);
        if((v.at("input")!="PLAYER_NORMAL"&&v.at("input")!="CARD_SELECT")||v.at("character")!="IRONCLAD"||
            !v.at("resolved").is_boolean()||(v.at("resolved")!=true&&v.at("input")!="CARD_SELECT")||
            v.at("powers_complete")!=true||v.at("limbo_empty")!=true)throw std::invalid_argument("actor_public_scope:"+v.at("input").dump()+":"+v.at("resolved").dump());
        if(!legal.is_array()||legal.empty()||legal.size()>4096)throw std::invalid_argument("actor_legal_capacity");
        const std::set<std::string> cardRequired={"base_cost","cost","damage","free","id","special","type","uid","up"};auto cardKeys=cardRequired;cardKeys.insert("playable");
        for(const auto& name:std::vector<std::string>{"hand","draw","discard","exhaust","held","top_candidates","selection_candidates"}){
            if(!v.at(name).is_array())throw std::invalid_argument("actor_pile");
            for(const auto& c:v.at(name)){keys(c,cardKeys,cardRequired);if(!c.at("uid").is_number_integer()||c.at("uid")!=-1)throw std::invalid_argument("actor_copy_id");}}
        const std::set<std::string> playerKeys={"block","current_hp","energy","max_hp","panache_damage","powers"};keys(v.at("player"),playerKeys,playerKeys);
        const std::set<std::string> monsterKeys={"block","current_hp","half_dead","is_gone","max_hp","move_base_damage","move_hits","powers"};
        for(const auto& m:v.at("monsters"))keys(m,monsterKeys,monsterKeys);
        for(const auto& p:v.at("power_records"))keys(p,{"id","amount","damage","hp_loss"},{"id","amount"});
        Encoded out;out.state.resize(stateGroups.size());J globals=v;globals.update(v.at("player"));
        out.state[0].push_back(row("global",{{"input",v.at("input")},{"task",v.at("selection")},{"type",v.at("selection_type")}},globals,out));
        for(unsigned g=1;g<8;++g){J pile=v.at(stateGroups[g]);
            if(g==2||g==6){for(auto& c:pile)c.erase("uid");std::stable_sort(pile.begin(),pile.end(),[](const J&a,const J&b){return a.dump()<b.dump();});}
            for(const auto& c:pile)out.state[g].push_back(card(c,out));}
        for(const auto& m:v.at("monsters")){auto rows=monster(m,out);out.state[8].insert(out.state[8].end(),rows.begin(),rows.end());}
        int order=0;for(auto p:v.at("power_records")){p["order"]=order++;out.state[9].push_back(row("player_power",{{"power",p.at("id")}},p,out));}
        order=0;for(const auto& r:v.at("relics"))out.state[10].push_back(row("relic",{{"relic",r}},J{{"order",order++}},out));
        for(const auto& p:v.at("potions"))out.state[11].push_back(row("potion",{{"potion",p}},J::object(),out));
        std::set<std::string> seen;
        for(const auto& a:legal){
            if(!seen.insert(a.dump()).second)throw std::invalid_argument("actor_duplicate_action");
            const std::string kind=a.at("kind"),task=v.at("selection");J cards=J::array(),potion,target;bool skip=false;
            if(v.at("input")=="CARD_SELECT"){
                keys(a,{"kind","choice_index"},{"kind","choice_index"});if(kind!="choose")throw std::invalid_argument("actor_selection_action");
                const auto choice=a.at("choice_index");const std::set<std::string> multi={"FORETHOUGHT","EXHAUST_MANY","GAMBLE","LIQUID_MEMORIES_POTION"};
                if(multi.count(task)){
                    const auto& selected=v.at("selection_options").at(index(choice,v.at("selection_options")));
                    const auto& pile=v.at(task=="LIQUID_MEMORIES_POTION"?"selection_candidates":"hand");std::set<int> unique;
                    if(!selected.is_array())throw std::invalid_argument("actor_selection_group");
                    for(const auto& i:selected){int n=index(i,pile);if(!unique.insert(n).second)throw std::invalid_argument("actor_duplicate_selection");cards.push_back(pile[n]);}
                }else{
                    const std::map<std::string,std::string> fields={{"HEADBUTT","discard"},{"WARCRY","hand"},{"EXHAUST_ONE","hand"},
                        {"ARMAMENTS","selection_candidates"},{"EXHUME","selection_candidates"},{"SECRET_TECHNIQUE","selection_candidates"},
                        {"SECRET_WEAPON","selection_candidates"},{"DUAL_WIELD","selection_candidates"},{"DISCOVERY","selection_candidates"},{"CODEX","selection_candidates"}};
                    if(!fields.count(task))throw std::invalid_argument("actor_selection_task");
                    if(task=="CODEX"&&choice.is_number_integer()&&choice==3)skip=true;
                    else {const auto& pile=v.at(fields.at(task));cards.push_back(pile.at(index(choice,pile)));}
                }
            }else if(kind=="play_card"){
                keys(a,{"kind","card_index","target_index"},{"kind","card_index"});cards.push_back(v.at("hand").at(index(a.at("card_index"),v.at("hand"))));
            }else if(kind=="use_potion"||kind=="discard_potion"){
                auto allowed=std::set<std::string>{"kind","potion_index"};if(kind=="use_potion")allowed.insert("target_index");keys(a,allowed,{"kind","potion_index"});
                potion=v.at("potions").at(index(a.at("potion_index"),v.at("potions")));
            }else if(kind=="end_turn")keys(a,{"kind"},{"kind"});
            else throw std::invalid_argument("actor_action_kind");
            if(a.contains("target_index"))target=v.at("monsters").at(index(a.at("target_index"),v.at("monsters")));
            std::vector<Group> cg(4);cg[0].push_back(row("action",{{"action",kind}},J{{"skip",skip},{"selected_count",cards.size()},{"has_target",!target.is_null()}},out));
            order=0;for(const auto& c:cards)cg[1].push_back(card(c,out,order++));
            if(!potion.is_null())cg[2].push_back(row("potion",{{"potion",potion}},J::object(),out));
            if(!target.is_null())cg[3]=monster(target,out);out.candidates.push_back(std::move(cg));
        }return out;
    }
    std::vector<float> forward(const Encoded& encoded)const{
        // Per-decision memoization only: exact same semantic row, same weights.
        // No cache survives a state advance or introduces cross-branch ownership.
        std::map<Row,V> cache;
        auto pool=[&](const Group& group){std::vector<float> out(33,0);for(const auto& r:group){auto it=cache.find(r);
            if(it==cache.end()){std::vector<float>x; x.reserve(96);const auto& e=weights_.at("embedding.weight");
                for(int token:r.categories){if(token<0||std::size_t(token)*8+8>e.size())throw std::invalid_argument("actor_token_range");for(int k=0;k<8;++k)x.push_back(e[token*8+k]);}
                x.insert(x.end(),r.values.begin(),r.values.end());x.insert(x.end(),r.known.begin(),r.known.end());
                const auto y=linear("row",x,32,true);V a;std::copy(y.begin(),y.end(),a.begin());it=cache.emplace(r,a).first;}
            for(int i=0;i<32;++i)out[i]+=it->second[i];}
            if(!group.empty()){for(int i=0;i<32;++i)out[i]/=float(group.size());out[32]=float(std::log1p(double(group.size())));}return out;};
        std::vector<float> pooled;for(const auto& group:encoded.state){auto p=pool(group);pooled.insert(pooled.end(),p.begin(),p.end());}
        const auto context=linear("context",pooled,32,true);std::vector<float> logits;logits.reserve(encoded.candidates.size());
        for(const auto& c:encoded.candidates){auto x=context;for(const auto& group:c){auto p=pool(group);x.insert(x.end(),p.begin(),p.end());}
            logits.push_back(linear("output",linear("head",x,32,true),1,false)[0]);}return logits;
    }
    J choose(const J& view,const J& legal)const{
        const auto logits=forward(encode(view,legal));unsigned best=0;
        for(unsigned i=1;i<logits.size();++i)if(logits[i]>logits[best]||(logits[i]==logits[best]&&legal[i].dump()<legal[best].dump()))best=i;
        return legal.at(best);
    }
};

// Experiment-only thread-local call binding; never loaded by production builds.
inline thread_local const Actor* active=nullptr;
struct Counts {unsigned calls=0,changed=0,selections=0,paused=0;J json()const{return J{{"calls",calls},{"changed",changed},{"selections",selections},{"paused",paused}};}};
inline J leaf(const J& view,const J& legal,const J& prior,const std::string& mode,Counts& counts){
    if(!active)throw std::runtime_error("actor_not_bound");++counts.calls;
    counts.selections+=view.at("input")=="CARD_SELECT";counts.paused+=view.at("resolved")==false;
    const auto action=active->choose(view,legal);
    if(mode=="probe")return prior;
    if(mode!="random")throw std::runtime_error("actor_mode");counts.changed+=action!=prior;return action;
}
}
