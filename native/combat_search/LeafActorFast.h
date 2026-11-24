#pragma once
// Same sealed actor contract and float operation order, fewer transient objects.
// All caches below are call-local; model objects only hold immutable weights/schema.
#include "LeafActor.h"

namespace spire_actor_fast {
using namespace spire_actor;
class Actor : public spire_actor::Actor {
    enum Role {GLOBAL,CARD,MONSTER,PLAYER_POWER,MONSTER_POWER,RELIC,POTION,ACTION};
    struct Field {std::string name;double scale;};
    struct RoleInfo {int token;std::vector<Field> fields;};
    std::array<RoleInfo,8> roles_;
    std::map<std::string,int> tokens_;
    std::vector<float> embedding_;
    template<unsigned I,unsigned O> struct Linear {
        std::array<float,I*O> weight;std::array<float,O> bias;
        void load(const J& tensors,const std::string& name){
            const auto w=tensors.at(name+".weight").at("values").get<std::vector<float>>();
            const auto b=tensors.at(name+".bias").at("values").get<std::vector<float>>();
            std::copy(w.begin(),w.end(),weight.begin());std::copy(b.begin(),b.end(),bias.begin());
        }
        std::array<float,O> run(const std::array<float,I>& x,bool activate)const{
            std::array<float,O> y;
            for(unsigned i=0;i<O;++i){float sum=bias[i];for(unsigned j=0;j<I;++j)sum+=weight[i*I+j]*x[j];
                y[i]=activate?std::tanh(sum):sum;if(!std::isfinite(y[i]))throw std::runtime_error("actor_nonfinite_logit");}
            return y;
        }
    };
    Linear<96,32> row_;Linear<396,32> context_;Linear<164,32> head_;Linear<32,1> output_;

    int token(const std::string& category,const std::string& value,Encoded& out)const{
        if(value.empty())return 0;const auto name=category+":"+key(value);const auto it=tokens_.find(name);
        if(it!=tokens_.end())return it->second;out.unknown[name]=out.unknown.value(name,0)+1;return 1;
    }
    Row row(Role role,std::initializer_list<std::pair<std::string,std::string>> cats,const J& source,
            Encoded& out,int order=-1,const J* overlay=nullptr)const{
        if(++out.rows>32768)throw std::invalid_argument("actor_row_cap");
        Row r;r.categories[0]=roles_[role].token;unsigned c=1;
        for(const auto& p:cats)r.categories[c++]=token(p.first,p.second,out);
        unsigned slot=0;
        for(const auto& field:roles_[role].fields){
            const J* x=nullptr;
            if(overlay){auto p=overlay->find(field.name);if(p!=overlay->end())x=&*p;}
            if(!x){auto p=source.find(field.name);if(p!=source.end())x=&*p;}
            if(order>=0&&field.name=="order"){r.values.at(slot)=float(double(order)/field.scale);r.known.at(slot)=1.f;}
            else if(x){
                if(!x->is_number()&&!x->is_boolean())throw std::invalid_argument("actor_numeric_scalar");
                const double v=(x->is_boolean()?(x->get<bool>()?1.:0.):x->get<double>())/field.scale;
                if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<float>::max())throw std::invalid_argument("actor_numeric_range");
                r.values.at(slot)=float(v);r.known.at(slot)=1.f;
            }++slot;
        }return r;
    }
    Row card(const J& c,Encoded& out,int order=-1)const{return row(CARD,{{"card",c.at("id")},{"type",c.at("type")}},c,out,order);}
    Group monster(const J& m,Encoded& out)const{
        Group rows;rows.reserve(1+m.at("powers").size());rows.push_back(row(MONSTER,{},m,out));
        for(auto p=m.at("powers").begin();p!=m.at("powers").end();++p)
            rows.push_back(row(MONSTER_POWER,{{"power",p.key()}},J{{"amount",p.value()}},out));
        return rows;
    }
public:
    Actor(const J& artifact,const J& expected):spire_actor::Actor(artifact,expected){
        // Base constructor preserves every envelope/schema/tensor/finite check.
        for(const auto& t:expected.at("vocab"))tokens_.emplace(t.get<std::string>(),int(tokens_.size()));
        const std::array<std::string,8> names={"global","card","monster","player_power","monster_power","relic","potion","action"};
        for(unsigned i=0;i<names.size();++i){auto& r=roles_[i];r.token=tokens_.at("role:"+names[i]);
            for(const auto& field:expected.at("fields").at(names[i]))r.fields.push_back({field.at(0),field.at(1)});}
        const auto& t=artifact.at("tensors");embedding_=t.at("embedding.weight").at("values").get<std::vector<float>>();
        row_.load(t,"row");context_.load(t,"context");head_.load(t,"head");output_.load(t,"output");
    }
    Encoded encode(const J& v,const J& legal)const{
        publicOnly(v);publicOnly(legal);
        static const std::set<std::string> viewKeys={"bomb_schema","bottom_tracked","character","discard","draw","exhaust","hand","held","input","limbo_empty",
            "monsters","opening_priority_ids","opening_schema","opening_uncertain","player","potions","power_records","powers_complete","relics","resolved","selection",
            "selection_any","selection_candidates","selection_copies","selection_max","selection_options","selection_pick","selection_type","times_damaged","top_candidates","uncertain_order"};
        keys(v,viewKeys,viewKeys);
        if((v.at("input")!="PLAYER_NORMAL"&&v.at("input")!="CARD_SELECT")||v.at("character")!="IRONCLAD"||
            !v.at("resolved").is_boolean()||(v.at("resolved")!=true&&v.at("input")!="CARD_SELECT")||
            v.at("powers_complete")!=true||v.at("limbo_empty")!=true)throw std::invalid_argument("actor_public_scope:"+v.at("input").dump()+":"+v.at("resolved").dump());
        if(!legal.is_array()||legal.empty()||legal.size()>4096)throw std::invalid_argument("actor_legal_capacity");
        static const std::set<std::string> cardRequired={"base_cost","cost","damage","free","id","special","type","uid","up"};
        static const std::set<std::string> cardKeys=[](){auto k=cardRequired;k.insert("playable");return k;}();
        for(unsigned g=1;g<8;++g){const auto& pile=v.at(stateGroups[g]);if(!pile.is_array())throw std::invalid_argument("actor_pile");
            for(const auto& c:pile){keys(c,cardKeys,cardRequired);if(!c.at("uid").is_number_integer()||c.at("uid")!=-1)throw std::invalid_argument("actor_copy_id");}}
        static const std::set<std::string> playerKeys={"block","current_hp","energy","max_hp","panache_damage","powers"};keys(v.at("player"),playerKeys,playerKeys);
        static const std::set<std::string> monsterKeys={"block","current_hp","half_dead","is_gone","max_hp","move_base_damage","move_hits","powers"};
        for(const auto& m:v.at("monsters"))keys(m,monsterKeys,monsterKeys);
        static const std::set<std::string> powerKeys={"id","amount","damage","hp_loss"},powerRequired={"id","amount"};
        for(const auto& p:v.at("power_records"))keys(p,powerKeys,powerRequired);
        Encoded out;out.state.resize(stateGroups.size());out.candidates.reserve(legal.size());
        out.state[0].push_back(row(GLOBAL,{{"input",v.at("input")},{"task",v.at("selection")},{"type",v.at("selection_type")}},v,out,-1,&v.at("player")));
        for(unsigned g=1;g<8;++g){const auto& pile=v.at(stateGroups[g]);auto& dst=out.state[g];dst.reserve(pile.size());
            if(g==2||g==6){
                // Serialize once/card, without UID, instead of twice/comparison.
                std::vector<std::pair<std::string,const J*>> sorted;sorted.reserve(pile.size());
                for(const auto& c:pile){J anonymous=c;anonymous.erase("uid");sorted.emplace_back(anonymous.dump(),&c);}
                std::stable_sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){return a.first<b.first;});
                for(const auto& c:sorted)dst.push_back(card(*c.second,out));
            }else for(const auto& c:pile)dst.push_back(card(c,out));
        }
        for(const auto& m:v.at("monsters")){auto rows=monster(m,out);out.state[8].insert(out.state[8].end(),rows.begin(),rows.end());}
        int order=0;for(const auto& p:v.at("power_records"))out.state[9].push_back(row(PLAYER_POWER,{{"power",p.at("id")}},p,out,order++));
        order=0;for(const auto& r:v.at("relics"))out.state[10].push_back(row(RELIC,{{"relic",r}},J::object(),out,order++));
        for(const auto& p:v.at("potions"))out.state[11].push_back(row(POTION,{{"potion",p}},J::object(),out));
        std::set<std::string> seen;
        for(const auto& a:legal){
            if(!seen.insert(a.dump()).second)throw std::invalid_argument("actor_duplicate_action");
            const std::string kind=a.at("kind"),task=v.at("selection");std::vector<const J*> cards;const J *potion=nullptr,*target=nullptr;bool skip=false;
            if(v.at("input")=="CARD_SELECT"){
                static const std::set<std::string> allowed={"kind","choice_index"};keys(a,allowed,allowed);
                if(kind!="choose")throw std::invalid_argument("actor_selection_action");
                const auto& choice=a.at("choice_index");static const std::set<std::string> multi={"FORETHOUGHT","EXHAUST_MANY","GAMBLE","LIQUID_MEMORIES_POTION"};
                if(multi.count(task)){
                    const auto& selected=v.at("selection_options").at(index(choice,v.at("selection_options")));
                    const auto& pile=v.at(task=="LIQUID_MEMORIES_POTION"?"selection_candidates":"hand");std::set<int> unique;
                    if(!selected.is_array())throw std::invalid_argument("actor_selection_group");cards.reserve(selected.size());
                    for(const auto& i:selected){int n=index(i,pile);if(!unique.insert(n).second)throw std::invalid_argument("actor_duplicate_selection");cards.push_back(&pile[n]);}
                }else{
                    static const std::map<std::string,std::string> fields={{"HEADBUTT","discard"},{"WARCRY","hand"},{"EXHAUST_ONE","hand"},
                        {"ARMAMENTS","selection_candidates"},{"EXHUME","selection_candidates"},{"SECRET_TECHNIQUE","selection_candidates"},
                        {"SECRET_WEAPON","selection_candidates"},{"DUAL_WIELD","selection_candidates"},{"DISCOVERY","selection_candidates"},{"CODEX","selection_candidates"}};
                    if(!fields.count(task))throw std::invalid_argument("actor_selection_task");
                    if(task=="CODEX"&&choice.is_number_integer()&&choice==3)skip=true;
                    else {const auto& pile=v.at(fields.at(task));cards.push_back(&pile.at(index(choice,pile)));}
                }
            }else if(kind=="play_card"){
                static const std::set<std::string> allowed={"kind","card_index","target_index"},required={"kind","card_index"};keys(a,allowed,required);
                cards.push_back(&v.at("hand").at(index(a.at("card_index"),v.at("hand"))));
            }else if(kind=="use_potion"||kind=="discard_potion"){
                static const std::set<std::string> allowed={"kind","potion_index","target_index"},required={"kind","potion_index"};
                keys(a,kind=="use_potion"?allowed:required,required);
                potion=&v.at("potions").at(index(a.at("potion_index"),v.at("potions")));
            }else if(kind=="end_turn"){
                static const std::set<std::string> allowed={"kind"};keys(a,allowed,allowed);
            }else throw std::invalid_argument("actor_action_kind");
            if(a.contains("target_index"))target=&v.at("monsters").at(index(a.at("target_index"),v.at("monsters")));
            std::vector<Group> cg(4);cg[0].push_back(row(ACTION,{{"action",kind}},J{{"skip",skip},{"selected_count",cards.size()},{"has_target",target!=nullptr}},out));
            cg[1].reserve(cards.size());order=0;for(const auto* c:cards)cg[1].push_back(card(*c,out,order++));
            if(potion)cg[2].push_back(row(POTION,{{"potion",*potion}},J::object(),out));
            if(target)cg[3]=monster(*target,out);out.candidates.push_back(std::move(cg));
        }return out;
    }
    std::vector<float> forward(const Encoded& encoded)const{
        std::map<Row,std::array<float,32>> rows;
        auto pool=[&](const Group& group){std::array<float,33> out{};
            for(const auto& r:group){auto it=rows.find(r);
                if(it==rows.end()){
                    std::array<float,96>x;unsigned slot=0;
                    for(int token:r.categories){if(token<0||std::size_t(token)*8+8>embedding_.size())throw std::invalid_argument("actor_token_range");
                        for(unsigned i=0;i<8;++i)x[slot++]=embedding_[token*8+i];}
                    for(float v:r.values)x[slot++]=v;for(float v:r.known)x[slot++]=v;
                    it=rows.emplace(r,row_.run(x,true)).first;
                }for(unsigned i=0;i<32;++i)out[i]+=it->second[i];
            }
            if(!group.empty()){for(unsigned i=0;i<32;++i)out[i]/=float(group.size());out[32]=float(std::log1p(double(group.size())));}return out;
        };
        if(encoded.state.size()!=12)throw std::runtime_error("actor_linear_shape");
        std::array<float,396> pooled;unsigned offset=0;
        for(const auto& group:encoded.state){const auto p=pool(group);std::copy(p.begin(),p.end(),pooled.begin()+offset);offset+=33;}
        const auto context=context_.run(pooled,true);std::vector<float> logits;logits.reserve(encoded.candidates.size());
        // Head input duplicates are exact, including multiplicity/order of rows.
        std::map<std::array<float,164>,float> candidates;
        for(const auto& c:encoded.candidates){if(c.size()!=4)throw std::runtime_error("actor_linear_shape");
            std::array<float,164>x;std::copy(context.begin(),context.end(),x.begin());offset=32;
            for(const auto& group:c){const auto p=pool(group);std::copy(p.begin(),p.end(),x.begin()+offset);offset+=33;}
            auto found=candidates.find(x);
            if(found==candidates.end())found=candidates.emplace(x,output_.run(head_.run(x,true),false)[0]).first;
            logits.push_back(found->second);
        }return logits;
    }
    J choose(const J& view,const J& legal)const{
        const auto logits=forward(encode(view,legal));unsigned best=0;
        for(unsigned i=1;i<logits.size();++i)if(logits[i]>logits[best]||(logits[i]==logits[best]&&legal[i].dump()<legal[best].dump()))best=i;
        return legal.at(best);
    }
};
inline thread_local const Actor* active=nullptr;
inline J leaf(const J& view,const J& legal,const J& prior,const std::string& mode,spire_actor::Counts& counts){
    if(!active)throw std::runtime_error("actor_not_bound");++counts.calls;
    counts.selections+=view.at("input")=="CARD_SELECT";counts.paused+=view.at("resolved")==false;
    const auto action=active->choose(view,legal);
    if(mode=="probe")return prior;if(mode!="random")throw std::runtime_error("actor_mode");counts.changed+=action!=prior;return action;
}
}
