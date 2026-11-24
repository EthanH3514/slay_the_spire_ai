#pragma once
// Pure immutable inference: accepts only the two public envelopes, never a simulator.
#include "CombatPolicyValue.h"
namespace spire_rich_model {
using namespace spire_actor;
using Vec=std::vector<float>;
inline void append(Vec& a,const Vec& b){a.insert(a.end(),b.begin(),b.end());}
struct Tensor {std::vector<int> shape;Vec values;};
struct EncodedRich {J json;};
class Model {
    J spec_;
    int outcomes_=2;
    bool playerValue_=false;
    bool criticState_=false;
    spire_actor_fast::Actor old_;
    std::map<std::string,Tensor> t_;
    std::map<std::string,std::map<std::string,int>> tokens_;
    Vec linear(const std::string& name,const Vec& x,bool activation=false) const {
        const auto& w=t_.at(name+".weight");
        if(w.shape.size()!=2||w.shape[1]!=int(x.size()))throw std::invalid_argument("rich_linear_shape");
        Vec y(w.shape[0]);auto b=t_.find(name+".bias");
        for(int i=0;i<w.shape[0];++i){float s=b==t_.end()?0:b->second.values[i];
            for(unsigned j=0;j<x.size();++j)s+=w.values[i*x.size()+j]*x[j];
            y[i]=activation?std::tanh(s):s;if(!std::isfinite(y[i]))throw std::invalid_argument("rich_nonfinite_output");}
        return y;
    }
    Vec embedding(const std::string& name,int id)const{
        const auto& t=t_.at(name+".weight");if(id<0||id>=t.shape[0])throw std::invalid_argument("rich_token_range");
        return Vec(t.values.begin()+id*8,t.values.begin()+(id+1)*8);
    }
    int token(const std::string& field,const J& value)const{
        if(!value.is_string())throw std::invalid_argument("rich_token_type");
        const auto& table=tokens_.at(field);auto found=table.find(value.get<std::string>());
        if(found==table.end())throw std::invalid_argument("rich_unknown_token");return found->second;
    }
    static Vec numbers(const J& row,const J& fields){
        Vec values,known;
        for(const auto& f:fields){const auto name=f.at(0).get<std::string>();const bool present=row.contains(name)&&!row.at(name).is_null();
            const auto x=present?row.at(name):J(0);
            if(!x.is_number()&&!x.is_boolean())throw std::invalid_argument("rich_numeric_type");
            const double v=x.is_boolean()?(x.get<bool>()?1:0):x.get<double>();
            if(!std::isfinite(v)||std::abs(v)>1e9)throw std::invalid_argument("rich_numeric_range");
            values.push_back(float(v/f.at(1).get<double>()));known.push_back(present?1.f:0.f);}
        append(values,known);return values;
    }
    static Vec pool(const std::vector<Vec>& rows){
        Vec out(33,0);for(const auto& r:rows)for(int i=0;i<32;++i)out[i]+=r[i];
        if(!rows.empty()){for(int i=0;i<32;++i)out[i]/=float(rows.size());out[32]=float(std::log1p(double(rows.size())));}return out;
    }
    std::pair<Vec,std::vector<Vec>> richContext(const J& e,const std::string& prefix)const{
        std::vector<Vec> monsters,relics;
        for(unsigned i=0;i<e.at("monster_categories").size();++i){const auto& c=e.at("monster_categories").at(i);
            Vec x=embedding(prefix+".monster_embedding",c.at(0));
            append(x,embedding(prefix+".intent_embedding",c.at(1)));append(x,embedding(prefix+".intent_embedding",c.at(2)));
            append(x,e.at("monster_numbers").at(i).get<Vec>());monsters.push_back(linear(prefix+".monster",x,true));}
        for(unsigned i=0;i<e.at("relic_categories").size();++i){Vec x=embedding(prefix+".relic_embedding",e.at("relic_categories").at(i).at(0));
            append(x,e.at("relic_numbers").at(i).get<Vec>());relics.push_back(linear(prefix+".relic",x,true));}
        Vec x=pool(monsters);append(x,pool(relics));append(x,e.at("global_numbers").get<Vec>());
        return {linear(prefix+".context",x,true),monsters};
    }
public:
    Model(const J& value,const J& expectedSchema,const std::string& parentActorDigest)
        :spec_(expectedSchema),old_(value.at("parent").at("actor"),value.at("parent").at("actor").at("schema")){
        const std::set<std::string> fields={"version","schema","parent","tensors","semantics"};keys(value,fields,fields);
        criticState_=value.at("version")=="public-combat-policy-value-v5-critic-state";
        playerValue_=criticState_||value.at("version")=="public-combat-policy-value-v4-player";
        outcomes_=(playerValue_||value.at("version")=="public-combat-policy-value-v3-escape")?3:2;
        if((outcomes_==2&&value.at("version")!="public-combat-policy-value-v2-rich")||value.at("schema")!=spec_)throw std::invalid_argument("rich_artifact_schema");
        spire_policy_value::Model checked(value.at("parent"),value.at("parent").at("actor"),parentActorDigest);
        J levels=J::array();for(int i=0;i<21;++i)levels.push_back((i+.5)/21);
        J semantics={{"target","terminal-outcome-resource-quantiles-v1"},{"value_to_actor_gradient",false},
            {"parent_frozen",true},{"full_belief",false},{"resource_scale",100.},{"quantile_levels",levels}};
        if(outcomes_==3){semantics["target"]="terminal-outcome-resource-quantiles-v2-escape";semantics["outcome_order"]={"loss","win","escape"};}
        if(playerValue_){semantics["value_player_input"]="direct-public-player-v1";
            semantics["player_fields"]=J::array({J::array({"current_hp",100}),J::array({"max_hp",100}),
                J::array({"block",100}),J::array({"energy",10})});}
        if(criticState_){semantics["value_state_encoder"]="independent-value-state-encoder-v1";
            semantics["value_state_schema_id"]=value.at("parent").at("actor").at("schema").at("id");}
        if(value.at("semantics")!=semantics)throw std::invalid_argument("rich_semantics");
        for(const auto& name:{"monsters","relics","intents","powers"}){
            auto& map=tokens_[name];for(const auto& token:spec_.at(name))if(!map.emplace(token.get<std::string>(),int(map.size())).second)
                throw std::invalid_argument("rich_duplicate_vocab");}
        std::map<std::string,std::vector<int>> shapes;
        auto dense=[&](std::string n,int in,int out,bool bias=true){shapes[n+".weight"]={out,in};if(bias)shapes[n+".bias"]={out};};
        for(std::string p:{"actor_extra","critic_extra"}){
            shapes[p+".monster_embedding.weight"]={int(tokens_["monsters"].size()),8};
            shapes[p+".intent_embedding.weight"]={int(tokens_["intents"].size()),8};
            shapes[p+".relic_embedding.weight"]={int(tokens_["relics"].size()),8};
            dense(p+".monster",40+2*tokens_["powers"].size(),32);dense(p+".relic",20,32);dense(p+".context",78,32);}
        dense("actor_hidden",229,32);dense("actor_output",32,1);dense("value_extra",32,64,false);
        dense("value_hidden",32,64);dense("outcome",64,outcomes_);dense("resource",64,21*outcomes_);
        if(playerValue_)dense("player_value",4,64,false);
        if(criticState_){shapes["critic_state.embedding.weight"]={int(value.at("parent").at("actor").at("schema").at("vocab").size()),8};
            dense("critic_state.row",96,32);dense("critic_state.context",396,32);}
        if(value.at("tensors").size()!=shapes.size())throw std::invalid_argument("rich_tensor_names");
        auto add=[&](const std::string& n,const J& x,const std::vector<int>& shape){
            keys(x,{"shape","values"},{"shape","values"});unsigned size=1;for(int d:shape)size*=d;
            if(x.at("shape")!=shape||!x.at("values").is_array()||x.at("values").size()!=size)throw std::invalid_argument("rich_tensor_shape");
            Tensor t{shape,{}};for(const auto& v:x.at("values")){
                if(!v.is_number()||!std::isfinite(v.get<double>())||std::abs(v.get<double>())>3e38)throw std::invalid_argument("rich_tensor_finite");
                t.values.push_back(v.get<float>());}t_[n]=std::move(t);};
        for(const auto& p:shapes)add(p.first,value.at("tensors").at(p.first),p.second);
        if(criticState_)for(int i=0;i<8;++i)if(t_.at("critic_state.embedding.weight").values[i]!=0)
            throw std::invalid_argument("critic_state_padding_embedding");
        const auto& actor=value.at("parent").at("actor").at("tensors");
        for(auto p=actor.begin();p!=actor.end();++p)add("base."+p.key(),p.value(),p.value().at("shape"));
    }
    J encode(const J& v,const J& legal,const J& r)const{
        publicOnly(r);const std::set<std::string> fields={"version","monsters","relics","turn","ascension","cards_played","attacks_played","skills_played","discarded"};keys(r,fields,fields);
        if(r.at("version")!="public-combat-entities-v2"||!r.at("monsters").is_array()||!r.at("relics").is_array()||
            r.at("monsters").empty()||r.at("monsters").size()>5||r.at("monsters").size()!=v.at("monsters").size()||r.at("relics").size()>256)
            throw std::invalid_argument("rich_entity_count");
        std::vector<std::string> a=v.at("relics").get<std::vector<std::string>>(),b;
        for(const auto& x:r.at("relics"))b.push_back(x.at("id"));std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        if(a!=b||std::find(a.begin(),a.end(),"runicdome")!=a.end()||std::find(a.begin(),a.end(),"prismaticshard")!=a.end())throw std::invalid_argument("rich_relic_alignment");
        J e={{"monster_categories",J::array()},{"monster_numbers",J::array()},{"relic_categories",J::array()},
             {"relic_numbers",J::array()},{"global_numbers",numbers(r,spec_.at("global_fields"))},{"targets",J::array()}};
        for(unsigned i=0;i<r.at("monsters").size();++i){const auto& m=r.at("monsters").at(i);const auto& base=v.at("monsters").at(i);
            keys(m,{"id","slot","intent","previous_intent"},{"id","slot","intent","previous_intent"});
            if(!m.at("slot").is_number_integer()||m.at("slot")!=i)throw std::invalid_argument("rich_slot");
            if(m.at("id")=="invalid0") {
                if(!base.at("is_gone").get<bool>()||base.at("half_dead").get<bool>()||!base.at("powers").empty()||
                   m.at("intent")!="UNKNOWN"||(!m.at("previous_intent").is_null()&&m.at("previous_intent")!="UNKNOWN"))throw std::invalid_argument("rich_empty_slot");
                for(const auto* f:{"current_hp","max_hp","block","move_base_damage","move_hits"})
                    if(base.at(f).get<double>()!=0)throw std::invalid_argument("rich_empty_slot");
            }
            e["monster_categories"].push_back({token("monsters",m.at("id")),token("intents",m.at("intent")),
                m.at("previous_intent").is_null()?0:token("intents",m.at("previous_intent"))});
            J values=base;values["slot"]=i;Vec ns=numbers(values,spec_.at("monster_fields"));
            J powers=J::object(),powerFields=J::array();for(auto p=base.at("powers").begin();p!=base.at("powers").end();++p)token("powers",p.key());
            for(const auto& name:spec_.at("powers")){const auto s=name.get<std::string>();powers[s]=base.at("powers").value(s,0);powerFields.push_back({s,100});}
            append(ns,numbers(powers,powerFields));e["monster_numbers"].push_back(ns);}
        for(unsigned i=0;i<r.at("relics").size();++i){const auto& relic=r.at("relics").at(i);
            keys(relic,{"id","order","counter","available","attack_played","skill_played","power_played"},{"id","order"});
            if(!relic.at("order").is_number_integer()||relic.at("order")!=i)throw std::invalid_argument("rich_relic_order");
            e["relic_categories"].push_back({token("relics",relic.at("id"))});e["relic_numbers"].push_back(numbers(relic,spec_.at("relic_fields")));}
        for(const auto& action:legal){const J t=action.value("target_index",J(-1));if(!t.is_number_integer()||t.get<int>() < -1||t.get<int>()>=int(r.at("monsters").size()))throw std::invalid_argument("rich_target");
            if(t.get<int>()>=0&&r.at("monsters").at(t.get<int>()).at("id")=="invalid0")throw std::invalid_argument("rich_empty_target");
            e["targets"].push_back(t);}
        return e;
    }
    J inspect(const J& v,const J& legal,const J& rich,bool diagnostic=false,bool policyOnly=false,bool valueOnly=false)const{
        const auto old=old_.encode(v,legal);const auto e=encode(v,legal,rich);
        std::map<Row,Vec> cache;
        auto oldPool=[&](const Group& group){std::vector<Vec> rows;
            for(const auto& r:group){auto it=cache.find(r);if(it==cache.end()){
                Vec x;for(int c:r.categories)append(x,embedding("base.embedding",c));
                x.insert(x.end(),r.values.begin(),r.values.end());x.insert(x.end(),r.known.begin(),r.known.end());
                it=cache.emplace(r,linear("base.row",x,true)).first;}rows.push_back(it->second);}return pool(rows);};
        Vec context;
        // Value-only v5 inference needs its own encoder, not an unused old
        // policy context. Legacy and policy-only operation order stays intact.
        if(!valueOnly||!criticState_){Vec joined;for(const auto& g:old.state)append(joined,oldPool(g));context=linear("base.context",joined,true);}
        Vec logits;
        if(!valueOnly){const auto ac=richContext(e,"actor_extra");
        for(unsigned i=0;i<old.candidates.size();++i){Vec x=context;for(const auto& g:old.candidates[i])append(x,oldPool(g));
            float score=linear("base.output",linear("base.head",x,true))[0];append(x,ac.first);
            const int target=e.at("targets").at(i);append(x,target<0?Vec(32,0):ac.second.at(target));x.push_back(target<0?0:1);
            logits.push_back(score+linear("actor_output",linear("actor_hidden",x,true))[0]);}
        }
        if(policyOnly)return J{{"logits",logits}};
        const auto vc=richContext(e,"critic_extra");
        Vec valueContext=context;
        if(criticState_){
            // Separate weights and call-local cache: policy pooling never sees
            // trainable critic features. No cross-call learned-feature cache.
            std::map<Row,Vec> criticCache;Vec joinedCritic;
            for(const auto& group:old.state){std::vector<Vec> rows;
                for(const auto& r:group){auto it=criticCache.find(r);if(it==criticCache.end()){
                    Vec x;for(int c:r.categories)append(x,embedding("critic_state.embedding",c));
                    x.insert(x.end(),r.values.begin(),r.values.end());x.insert(x.end(),r.known.begin(),r.known.end());
                    it=criticCache.emplace(r,linear("critic_state.row",x,true)).first;}rows.push_back(it->second);}
                append(joinedCritic,pool(rows));}
            valueContext=linear("critic_state.context",joinedCritic,true);
        }
        auto hidden=linear("value_hidden",valueContext),extra=linear("value_extra",vc.first);
        if(playerValue_){
            Vec player;const auto& row=v.at("player");
            for(const auto& f:J::array({J::array({"current_hp",100}),J::array({"max_hp",100}),
                    J::array({"block",100}),J::array({"energy",10})})){
                const auto& x=row.at(f.at(0).get<std::string>());
                if(!x.is_number())throw std::invalid_argument("player_value_numeric");
                const double n=x.get<double>();
                if(!std::isfinite(n)||n<0||n>1e9)throw std::invalid_argument("player_value_range");
                player.push_back(float(n/f.at(1).get<double>()));
            }
            const auto direct=linear("player_value",player);
            for(unsigned i=0;i<hidden.size();++i)hidden[i]=std::tanh((hidden[i]+extra[i])+direct[i]);
        } else for(unsigned i=0;i<hidden.size();++i)hidden[i]=std::tanh(hidden[i]+extra[i]);
        const auto outcome=linear("outcome",hidden),raw=linear("resource",hidden);J q=J::array();
        for(int k=0;k<outcomes_;++k){Vec row;float sum=0;for(int i=0;i<21;++i){float x=raw[k*21+i];sum+=x>20?x:std::log1p(std::exp(x));row.push_back(sum);}q.push_back(row);}
        J result={{"logits",logits},{"outcome_logits",outcome},{"resource_quantiles",q}};
        if(diagnostic){result["encoded"]=e;result["old_encoded"]=old.json(old_.schema().at("id"));}
        return result;
    }
};
}
