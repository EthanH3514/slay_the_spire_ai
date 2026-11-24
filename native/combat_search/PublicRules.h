#pragma once
#include "Performance.h"
// First-party public rule policy (combat-recipe-rule-v56).
// This file MUST NOT include a simulator header: rules see owned public JSON only.
// Deliberately retain the reference heuristic, including its approximations.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace spire_search {
inline constexpr auto abiVersion="public-flat-search-v6";
inline constexpr auto ruleVersion="combat-recipe-rule-v56";
inline constexpr auto valueVersion="combat-local-heuristic-v3-escape";
inline constexpr auto algorithmVersion="combat-flat-rollout-v3-potion-aware";
inline constexpr auto recipeVersion="combat-card-recipes-v48";
using J = nlohmann::json;
struct NotPorted : std::runtime_error { using std::runtime_error::runtime_error; };
// A reviewed distinction: lack of combination coverage is not corrupt public
// state. Only this typed refusal may select the explicit native-rule fallback.
struct ScopeGap : NotPorted { using NotPorted::NotPorted; };
inline std::string key(const std::string &s) {
    std::string r;
    for (unsigned char c : s) if (std::isalnum(c)) r += char(std::tolower(c));
    return r;
}
inline int number(const J &j, const char *s, int fallback=0) { return j.value(s, fallback); }
inline int power(const J &j, const std::string &s) { return j.at("powers").value(s, 0); }
inline bool live(const J &m) { return !m.at("is_gone").get<bool>() && number(m,"current_hp") > 0; }
inline bool has(const J &xs, const std::string &s) { return std::find(xs.begin(), xs.end(), s) != xs.end(); }
inline int attack(const J &m, const J &p, bool weak, int strength, bool intangible) {
    int base=number(m,"move_base_damage",-1), hits=number(m,"move_hits");
    if (base < 0 || hits <= 0) return 0;
    int d=std::max(0,base+strength);
    if (weak) d=int(d*.75);
    if (power(p,"vulnerable")>0) d=int(d*1.5);
    return (intangible ? std::min(1,d) : d)*hits;
}
inline int incoming(const J &v, bool intangible) {
    const auto &p=v.at("player"); int n=0;
    for (const auto &m:v.at("monsters")) if(live(m))
        n+=attack(m,p,power(m,"weak")>0,power(m,"strength"),intangible);
    return std::max(0,n-number(p,"block"));
}

// Explicit heuristic opportunity cost in HP units, not a learned future value.
// Survival dominates it. Kept potions retain value even at a terminal victory.
inline double potionReserve(const std::string &id) {
    if(id=="emptypotionslot"||id=="emptypotionid"||id=="potionslot"||id=="invalid")return 0.;
    if(id=="fairypotion")return 12.;
    if(id=="smokebomb")return 10.;
    if(id=="entropicbrew")return 8.;
    if(id=="fruitjuice")return 4.;
    return 3.;
}
inline double inventoryReserve(const J &view) {
    double value=0.;
    for(const auto &id:view.value("potions",J::array()))value+=potionReserve(id.get<std::string>());
    return value;
}

// Error-free partial sums, including the final half-even correction. Match
// Python math.fsum rather than changing ties with an epsilon or long-double sum.
inline double preciseSum(const std::vector<double> &values) {
    std::vector<double> partials;
    for(double x:values) {
        std::size_t n=0;
        for(double y:partials) {
            if(std::abs(x)<std::abs(y))std::swap(x,y);
            const double hi=x+y, lo=y-(hi-x);
            if(lo!=0)partials[n++]=lo;
            x=hi;
        }
        partials.resize(n);
        if(x!=0)partials.push_back(x);
    }
    if(partials.empty())return 0.;
    double hi=partials.back(),lo=0.;partials.pop_back();
    while(!partials.empty()) {
        const double x=hi,y=partials.back();partials.pop_back();
        hi=x+y;lo=y-(hi-x);
        if(lo!=0)break;
    }
    if(!partials.empty()&&((lo<0&&partials.back()<0)||(lo>0&&partials.back()>0))) {
        double y=lo*2.,x=hi+y;
        if(y==x-hi)hi=x;
    }
    return hi;
}

// Exactly itertools.combinations order; Forethought adds the reverse of each
// multi-card group. Bounded subsets, not factorial permutations or recursion.
inline J choiceGroups(int size,int limit,bool reverse=false,bool optional=true) {
    if(size<0||size>10)throw NotPorted("selection hand size");
    J out=J::array();if(optional)out.push_back(J::array());
    for(int count=1;count<=std::min(size,limit);++count) {
        std::vector<int> group(count);for(int i=0;i<count;++i)group[i]=i;
        while(true) {
            out.push_back(group);
            if(reverse&&count>1)out.push_back(std::vector<int>(group.rbegin(),group.rend()));
            int i=count-1;while(i>=0&&group[i]==size-count+i)--i;
            if(i<0)break;
            ++group[i];for(int j=i+1;j<count;++j)group[j]=group[j-1]+1;
        }
    }
    return out;
}

inline J retrievalGroups(int size,int count) {
    if(size<1 || (count!=1 && count!=2) || count>size)throw NotPorted("public retrieval count");
    J out=J::array();
    for(int i=0;i<size;++i) {
        if(count==1)out.push_back({i});
        else for(int j=0;j<size;++j)if(j!=i)out.push_back({i,j});
    }
    return out;
}

inline bool bombId(const std::string &s) {
    return s.rfind("thebomb",0)==0&&std::all_of(s.begin()+7,s.end(),[](unsigned char c){return std::isdigit(c);});
}
inline J bombPackets(const J &v) {
    J packets=J::array();
    for(const auto &p:v.at("power_records"))if(bombId(p.at("id")))
        packets.push_back({{"turns",number(p,"amount")},{"damage",number(p,"damage")}});
    return packets;
}
inline bool hasBomb(const J &v) {
    if(!bombPackets(v).empty())return true;
    for(const char *pile:{"hand","draw","discard","exhaust"})for(const auto &c:v.at(pile))
        if(c.at("id")=="thebomb")return true;
    return false;
}
inline bool bombScope(const J &v,bool prospective=false,bool ruleStateOnly=false) {
    if(!prospective&&!hasBomb(v))return true;
    if(number(v,"bomb_schema")!=1||!v.at("powers_complete").get<bool>()||!v.at("limbo_empty").get<bool>())return false;
    if((!v.at("resolved").get<bool>()||v.at("input")!="PLAYER_NORMAL") &&
        v.at("input")!="CARD_SELECT")return false;
    auto packets=bombPackets(v);
    for(const auto &p:packets)if(number(p,"turns")<1||number(p,"turns")>3
        ||number(p,"damage")<=0)return false;
    std::set<std::string> seen;
    for(const auto &p:v.at("power_records")) {
        auto id=p.at("id").get<std::string>();
        if(!p.at("amount").is_number_integer())return false;
        int n=number(p,"amount");
        if(bombId(id))continue;
        if(id.empty()||!seen.insert(id).second)return false;
        if(id=="combust") {
            int hp=number(p,"hp_loss");
            if(hp<1)return false;
        }
    }
    return true;
}
inline double bombPressure(const J &v,int turns=0,int damage=0) {
    if(!bombScope(v,turns>0))return 0.;
    J packets=bombPackets(v);
    if(turns) {
        if(turns<1||turns>3||damage<=0)return 0.;
        packets.push_back({{"turns",turns},{"damage",damage}});
    }
    std::stable_sort(packets.begin(),packets.end(),[](const J&a,const J&b){return a.at("turns")<b.at("turns");});
    double score=0.;
    for(const auto &m:v.at("monsters")) {
        int hp=number(m,"current_hp"),block=number(m,"block");
        if(!live(m)||m.at("half_dead").get<bool>()||block<0)continue;
        for(const auto &p:packets) {
            int hit=power(m,"intangible")>0?1:number(p,"damage"),absorbed=std::min(block,hit);
            block-=absorbed;int useful=std::min(hp,hit-absorbed);hp-=useful;
            score+=useful*std::pow(.5,number(p,"turns"));
        }
    }
    return score;
}

// One mechanism inventory, not a per-card permission list. Imported metadata
// grants no evidence; both search and public-history evidence must be present.
inline const std::set<std::string> &portedPrimitives() {
    static const std::set<std::string> p={
        "DIRECT_DAMAGE","AREA_DAMAGE","CURRENT_BLOCK_DAMAGE","DYNAMIC_DAMAGE",
        "GROW_PLAYED_DAMAGE","GAIN_BLOCK","DRAW_CARDS","GAIN_ENERGY","GAIN_STRENGTH",
        "APPLY_LOSE_STRENGTH","GAIN_CURRENT_STRENGTH","LOSE_HP","HEAL","APPLY_WEAK",
        "APPLY_VULNERABLE","ATTACK_SEQUENCE","GENERATE_CARD","EXHAUST_RANDOM_HAND",
        "ON_EXHAUST_ENERGY","ON_HP_LOSS_COST_REDUCTION","REGISTER_DRAW_TRIGGER",
        "REGISTER_EXHAUST_TRIGGER","REGISTER_CORRUPTION","REGISTER_ATTACK_REPEAT",
        "APPLY_NO_DRAW","APPLY_NO_BLOCK","GAIN_ARTIFACT","GAIN_INTANGIBLE",
        "ENEMY_STRENGTH_LOSS","RETAIN_BLOCK","CURRENT_BLOCK_GAIN","END_TURN_BLOCK",
        "TEMP_RETALIATION","SELF_VULNERABLE","SHUFFLE_DISCARD_INTO_DRAW",
        "AREA_LIFESTEAL","FATAL_ATTACK","REGISTER_EVENT_TRIGGER","MODIFY_HAND_COST",
        "EXHAUST_CHOSEN_HAND","EXHAUST_HAND_GROUP","REGISTER_TURN_TRIGGER",
        "UPGRADE_CARDS","MOVE_SELECTED_CARD","COPY_SELECTED_CARD","SELECT_BOTTOM_FREE",
        "EXHAUST_UP_TO_HAND","RETRIEVE_RANDOM_ATTACKS","GENERATE_CHOICE_HAND","GENERATE_X_HAND",
        "GENERATE_RANDOM_HAND","GENERATE_RANDOM_DRAW","AUTOPLAY_TOP_CARD","REGISTER_DELAYED_DAMAGE"
    };
    return p;
}
inline void guard(const J &v,const J &meta,J *ruleScopeGaps=nullptr) {
    spire_perf::Scope timing(spire_perf::Phase::Guard);
    if(!v.contains("opening_schema")||!v.at("opening_schema").is_number_integer()||
       !v.contains("opening_priority_ids")||!v.at("opening_priority_ids").is_array()||
       !v.contains("opening_uncertain")||!v.at("opening_uncertain").is_boolean())
        throw NotPorted("invalid public opening-order schema");
    const auto &opening=v.at("opening_priority_ids");
    int openingSchema=v.at("opening_schema").get<int>(),last=-1;
    if((openingSchema!=0&&openingSchema!=1)||(openingSchema==0&&!opening.empty())||
       (opening.empty()&&v.at("opening_uncertain").get<bool>()))throw NotPorted("invalid public opening-order schema");
    for(const auto &uid:opening) {
        if(!uid.is_number_integer()||uid.get<int>()<=last||
           std::count_if(v.at("draw").begin(),v.at("draw").end(),
               [&](const J &c){return c.at("uid")==uid;})!=1)
            throw NotPorted("invalid public opening-priority identities");
        last=uid.get<int>();
    }
    J gaps=J::array();
    auto bombCheck=[&](bool prospective,const char *reason) {
        if(bombScope(v,prospective))return;
        if(!bombScope(v,prospective,true))throw NotPorted(reason);
        gaps.push_back(reason);
    };
    static const std::set<std::string> selections={"EXHAUST_ONE","EXHAUST_MANY","FORETHOUGHT","ARMAMENTS",
        "HEADBUTT","WARCRY","EXHUME","SECRET_TECHNIQUE","SECRET_WEAPON","DUAL_WIELD","DISCOVERY",
        "GAMBLE","CODEX","LIQUID_MEMORIES_POTION"};
    if(v.at("input")!="PLAYER_NORMAL" && !(v.at("input")=="CARD_SELECT" && selections.count(v.value("selection",""))))
        throw NotPorted("selection window");
    if(has(v.at("relics"),"prismaticshard") || has(v.at("relics"),"runicdome"))
        throw NotPorted("unsupported policy scope");
    // Persistent powers can survive their source card. Do not infer absence
    // from pile contents or omit the reference's specialized admission checks.
    for(const char *p:{"masterreality"})
        if(power(v.at("player"),p)!=0) throw NotPorted(std::string("persistent ")+p);
    bombCheck(false,"The Bomb public scope");
    const auto &p=v.at("player");
    if(p.at("powers").contains("panache") && (power(p,"panache")<1 || power(p,"panache")>5 || number(p,"panache_damage")<=0))
        throw NotPorted("invalid public Panache state");
    for(const char *pile:{"hand","draw","discard","exhaust","held"}) for(const auto &c:v.at(pile)) {
        const auto id=c.at("id").get<std::string>();
        if(!meta.contains(id) || !meta.at(id).at("audited").get<bool>())
            throw NotPorted("card evidence: "+id);
        const auto &r=meta.at(id);
        for(const auto &e:r.at("effects")) {
            const auto primitive=e.at("primitive").get<std::string>();
            if(!portedPrimitives().count(primitive)) throw NotPorted("primitive: "+primitive);
        }
        if(number(c,"up")<0) throw NotPorted("invalid upgrade count");
        if(id=="rampage" && number(c,"damage")<8) throw NotPorted("invalid Rampage state");
        if(id=="ritualdagger" && (number(c,"damage")<15 || number(c,"damage")>32767))
            throw NotPorted("invalid Ritual Dagger state");
        if(id=="forethought"&&!v.at("bottom_tracked").get<bool>())throw NotPorted("public bottom knowledge");
    }
    bool generation=power(p,"magnetism")!=0||v.value("selection","")=="DISCOVERY";
    bool classPool=v.value("selection","")=="DISCOVERY",damageHistory=classPool;
    auto contracts=[&](const J &e) {
        for(const char *field:{"random_hand_generation","random_draw_generation","discovery_generation"}) {
            if(!e.contains(field)||e.at(field).is_null())continue;
            auto contract=e.at(field).get<std::string>();generation=true;
            classPool=classPool||contract.rfind("IRONCLAD",0)==0;
            damageHistory=damageHistory||contract=="IRONCLAD_ATTACK_TURN_ZERO"||contract=="IRONCLAD_THREE";
        }
    };
    int mayhem=power(p,"mayhem"),copies=0;
    std::vector<std::string> future;
    for(const char *pile:{"draw","hand","discard","exhaust"})for(const auto &c:v.at(pile)) {
        auto id=c.at("id").get<std::string>();future.push_back(id);copies+=id=="mayhem";
        if(std::string(pile)=="exhaust")continue;
        for(const auto &e:meta.at(id).at("effects")) {
            contracts(e);
            if(!e.at("turn_trigger").is_null())for(const auto &t:e.at("turn_trigger").at("effects"))contracts(t);
        }
    }
    if(generation&&((classPool&&v.value("character","")!="IRONCLAD")||(damageHistory&&number(v,"times_damaged",-1)<0)))
        throw NotPorted("random generation public prerequisites");
    if(p.at("powers").contains("mayhem")&&mayhem<1)throw NotPorted("Mayhem stack state");
    if(mayhem||copies) {
        if(power(p,"magnetism")!=0)for(const auto &id:meta.at("$pools").at("COLORLESS_TURN_START"))future.push_back(id);
        std::set<std::string> visited;
        for(std::size_t i=0;i<future.size();++i) {
            auto id=future[i];if(!visited.insert(id).second)continue;
            if(!meta.contains(id)||!meta.at(id).at("audited").get<bool>())throw NotPorted("Mayhem future evidence");
            if(id=="thebomb")bombCheck(true,"Mayhem future Bomb scope");
            auto enqueue=[&](const J &e) {
                for(const char *field:{"random_hand_generation","random_draw_generation","discovery_generation"})
                    if(e.contains(field)&&!e.at(field).is_null())for(const auto &x:meta.at("$pools").at(e.at(field).get<std::string>()))future.push_back(x);
            };
            for(const auto &e:meta.at(id).at("effects")) {
                if(!e.at("generated_card_id").is_null())future.push_back(key(e.at("generated_card_id")));
                enqueue(e);
                if(!e.at("turn_trigger").is_null())for(const auto &t:e.at("turn_trigger").at("effects"))enqueue(t);
            }
        }
    }
    if(v.at("input")=="CARD_SELECT") {
        const auto task=v.value("selection","");
        if(task=="GAMBLE" && v.at("selection_options")!=choiceGroups(int(v.at("hand").size()),int(v.at("hand").size())))
            throw NotPorted("public redraw subsets");
        if(task=="CODEX"||task=="LIQUID_MEMORIES_POTION") {
            const auto &candidates=v.at("selection_candidates");
            if(candidates.empty()||(task=="CODEX"&&candidates.size()!=3))throw NotPorted("public offered candidates");
            for(const auto &c:candidates) {
                auto id=c.at("id").get<std::string>();
                if(!meta.contains(id)||!meta.at(id).at("audited").get<bool>())throw NotPorted("offered card evidence: "+id);
            }
            if(task=="LIQUID_MEMORIES_POTION") {
                if(v.at("selection_options")!=retrievalGroups(int(candidates.size()),number(v,"selection_pick",1)))
                    throw NotPorted("public retrieval choices");
                auto expected=v.at("discard");auto actual=candidates;
                // The Python transport may supply candidate playability; pile
                // entries intentionally omit that hand-only property. Compare
                // the full card identity and cost state, not this presentation field.
                for(auto &c:actual)c.erase("playable");
                auto less=[](const J&a,const J&b){return a.dump()<b.dump();};
                std::sort(expected.begin(),expected.end(),less);std::sort(actual.begin(),actual.end(),less);
                if(expected!=actual)throw NotPorted("public discard retrieval candidates");
            }
        }
        if(task=="ARMAMENTS") {
            J expected=J::array();
            for(const auto &c:v.at("hand"))if(c.at("type")!="CURSE"&&c.at("type")!="STATUS"
                &&(number(c,"up")==0||c.at("id")=="searingblow"))expected.push_back(c);
            if(expected.empty()||v.at("selection_candidates")!=expected)
                throw NotPorted("Armaments public candidate mapping");
        }
        if(task=="FORETHOUGHT" && (v.at("hand").empty()||!v.at("bottom_tracked").get<bool>()||
            v.at("selection_options")!=choiceGroups(int(v.at("hand").size()),v.at("selection_any").get<bool>()?10:1,true,v.at("selection_any").get<bool>())))
            throw NotPorted("Forethought public ordered choices");
        if(task=="EXHAUST_MANY"&&((number(v,"selection_max")<1||number(v,"selection_max")>10)||
            v.at("selection_options")!=choiceGroups(int(v.at("hand").size()),number(v,"selection_max"))))
            throw NotPorted("public exhaust subsets");
        if(task=="DUAL_WIELD"&&number(v,"selection_copies")!=1&&number(v,"selection_copies")!=2)throw NotPorted("public copy count");
        if(task=="DISCOVERY") {
            if((number(v,"selection_copies")!=1&&number(v,"selection_copies")!=2)||v.at("selection_candidates").size()!=3)throw NotPorted("public Discovery count");
            auto type=v.value("selection_type","INVALID");
            if(type!="INVALID"&&type!="ATTACK"&&type!="SKILL"&&type!="POWER"&&type!="STATUS")throw NotPorted("public Discovery pool type");
            const auto &pool=meta.at("$pools").at(type=="STATUS"?"COLORLESS_TURN_START":"IRONCLAD_THREE");
            std::set<std::string> ids;
            for(const auto &c:v.at("selection_candidates")) {
                auto id=c.at("id").get<std::string>();
                if(!ids.insert(id).second||number(c,"up")!=0||!has(pool,id)||
                   (type!="INVALID"&&type!="STATUS"&&c.at("type")!=type))throw NotPorted("public Discovery candidates");
            }
        }
    }
    // Finish every hard validation before reporting a recoverable gap: a scope
    // warning must never mask a malformed selection/card/instance later on.
    if(ruleScopeGaps)*ruleScopeGaps=gaps;
    else if(!gaps.empty())throw ScopeGap(gaps[0].get<std::string>());
}

// Reconstructing an independent world needs more evidence than scoring a
// public action. Random insertion after known-top/bottom placement leaves a
// correlated order posterior which the root sampler does not represent yet.
// Keep this restriction at the resampling boundary, not at later rule choices
// inside an already independently sampled branch.
inline void guardResamplingRoot(const J &v,const J &meta) {
    if(v.at("uncertain_order").get<bool>()) throw NotPorted("uncertain public draw order");
    guard(v,meta);
}

class Rules {
    const J &v, &meta, &p, &hand, &monsters;
    int threat;
    std::vector<int> living;
    bool relic(const char *id) const { return has(v.at("relics"),id); }
    const J &recipe(const J &c) const { return meta.at(c.at("id").get<std::string>()); }
    bool up(const J &c) const { return number(c,"up")>0; }
    bool attackCard(const J &c) const { return c.at("type")=="ATTACK"; }
    bool skill(const J &c) const { return c.at("type")=="SKILL"; }
    bool disposable(const J &c) const { return c.at("type")=="CURSE" || c.at("type")=="STATUS"; }
    bool corrupt(const J &c) const { return power(p,"corruption")!=0 && skill(c); }
    int cost(const J &c) const { int t=number(c,"cost"); return c.at("free").get<bool>() && t>=0 ? 0 : t; }
    int fixed(const J &e,bool u) const { const auto &n=e.at(u?"upgraded_amount":"base_amount"); return n.is_null()?0:n.get<int>(); }
    int dynamic(const J &e,const J &c) const {
        const auto source=e.at("dynamic_damage").at("source").get<std::string>();
        if(source=="DRAW_PILE_COUNT") return int(v.at("draw").size());
        if(source=="UPGRADE_COUNT") {int n=number(c,"up"); return 12+n*(n+7)/2;}
        if(source=="STRIKE_COUNT") {
            static const std::set<std::string> strikes={"strikered","strikegreen","strikeblue","strikepurple",
                "twinstrike","wildstrike","pommelstrike","perfectedstrike","swiftstrike","thunderstrike",
                "meteorstrike","sneakystrike","windmillstrike"};
            int n=0; for(const char *s:{"hand","draw","discard"}) for(const auto &x:v.at(s))
                n+=strikes.count(x.at("id").get<std::string>())!=0;
            return 6+(up(c)?3:2)*n;
        }
        return number(c,"damage");
    }
    double cardValue(const J &c) const {
        if(disposable(c)) return -20;
        if(!c.value("playable",true)) return -5;
        const auto &r=recipe(c); double d=std::max(0,number(r.at("variants").at(up(c)?1:0),"base_damage"));
        for(const auto &e:r.at("effects")) {
            if(!e.at("delayed_damage").is_null())d=fixed(e,up(c))*std::pow(.5,number(e.at("delayed_damage"),"turns"));
            if(!e.at("dynamic_damage").is_null())d=dynamic(e,c);
        }
        return (c.at("type")=="POWER"?3.:((attackCard(c)||skill(c))?2.:.5))+d*.25+(up(c)?.5:0.);
    }
    double intrinsic(const J &c) const {
        double n=0; for(const auto &e:recipe(c).at("effects")) if(e.at("primitive")=="ON_EXHAUST_ENERGY") n+=fixed(e,up(c))*3.;
        return n;
    }
    double consume(const J &c) const {
        const auto &copy=recipe(c).at("self_copy");
        if(!copy.is_null() && copy.at("timing")=="EXHAUST") return intrinsic(c);
        return (disposable(c)||!c.value("playable",true) ? 3. : -std::max(1.,cardValue(c)*.5))+intrinsic(c);
    }
    int adjustedBlock(int base) const {
        if(power(p,"noblock")>0) return 0;
        int n=std::max(0,base+power(p,"dexterity"));
        return power(p,"frail")>0?int(n*.75):n;
    }
    double blockTrigger(int events) const {
        int n=std::max(0,power(p,"juggernaut"));
        if(!n || living.empty() || events<=0) return 0;
        int total=0; for(int idx:living) {const auto &m=monsters[idx];
            total+=std::min(std::max(0,(power(m,"intangible")?std::min(n,1):n)-number(m,"block")),number(m,"current_hp"));}
        return double(events)*total/living.size();
    }
    double block(int base) const {
        int adjusted=adjustedBlock(base), useful=std::min(adjusted,threat);
        return (threat>=number(p,"current_hp") && useful>0 ?100.:0.)+useful*1.5+blockTrigger(adjusted>0);
    }
    double nonCardBlock(int n,int events=1) const {
        int gain=std::min(std::max(0,n),std::max(0,999-std::max(0,number(p,"block"))));
        int useful=std::min(gain,threat), hp=number(p,"current_hp");
        int retained=power(p,"barricade")!=0?std::max(0,gain-useful):0;
        return (threat>=hp && hp>threat-useful?100.:0.)+useful*1.5+std::min(12.,retained*.5)+blockTrigger(n>0?events:0);
    }
    double draw(int n,int freed=0) const {
        if(power(p,"nodraw")!=0) return 0;
        return 4.*std::min({n,int(v.at("draw").size()+v.at("discard").size()),std::max(0,10-std::max(0,int(hand.size())-1-freed))});
    }
    double nonAttack(const J &m,int n) const {
        if(power(m,"intangible")>0) n=std::min(n,1);
        return std::min(std::max(0,n-number(m,"block")),std::max(0,number(m,"current_hp")));
    }
    int monsterAttack(const J &m,bool weak,int strength) const {
        return attack(m,p,weak,strength,power(p,"intangible")>0);
    }
    double upgradeValue(const J &c,bool inHand,int energy) const {
        const auto &r=recipe(c);bool repeat=false;
        for(const auto &e:r.at("effects"))repeat=repeat||(!e.at("dynamic_damage").is_null()&&e.at("dynamic_damage").at("source")=="UPGRADE_COUNT");
        if((up(c)&&!repeat)||disposable(c))return 0.;
        int old=number(c,"base_cost"),turn=number(c,"cost"),base=number(r.at("variants")[0],"energy_cost"),next=number(r.at("variants")[1],"energy_cost");
        bool blood=false;for(const auto &e:r.at("effects"))blood=blood||e.at("primitive")=="ON_HP_LOSS_COST_REDUCTION";
        if(blood) {if(old<base)next=std::max(0,old-1);} else if(base==next)next=old;
        int nextTurn=turn>0?std::max(0,next+turn-old):turn;
        bool playable=inHand&&nextTurn>=0&&nextTurn<=energy;
        double result=std::max(0,old-next)*2.;if(playable&&turn>energy)result+=3.;
        for(const auto &e:r.at("effects")) {
            if(!e.at("dynamic_damage").is_null()) {
                J upgraded=c;upgraded["up"]=number(c,"up")+1;
                result+=std::max(0,dynamic(e,upgraded)-dynamic(e,c))*(playable?1.:.5);continue;
            }
            if(e.at("base_amount").is_null()||e.at("upgraded_amount").is_null())continue;
            int delta=fixed(e,true)-fixed(e,false);const auto primitive=e.at("primitive");
            if(primitive=="GAIN_BLOCK") {
                int gain=std::max(0,adjustedBlock(fixed(e,true))-adjustedBlock(fixed(e,false)));
                result+=playable?std::min(threat,gain)*3.:gain*.5;
            } else if(primitive=="DIRECT_DAMAGE"||primitive=="AREA_DAMAGE")result+=std::max(0,delta)*(playable?1.:.5);
            else if(!e.at("delayed_damage").is_null())result+=std::max(0,delta)*std::pow(.5,number(e.at("delayed_damage"),"turns"));
            else if(primitive=="DRAW_CARDS"||primitive=="GAIN_ENERGY"||primitive=="GAIN_STRENGTH"||primitive=="ENEMY_STRENGTH_LOSS")result+=std::max(0,delta)*2.;
            else if(!e.at("combat_upgrade").is_null()&&e.at("combat_upgrade").at("base_scope")!=e.at("combat_upgrade").at("upgraded_scope"))result+=2.;
        }
        if(r.at("exhaust")[0].get<bool>()&&!r.at("exhaust")[1].get<bool>())result+=1.;
        return result;
    }
    double optionalExhaust(const J &cards,const J &indices,int handCount) const {
        double result=0.;for(const auto &i:indices)result+=consume(cards.at(i.get<int>()));
        int count=int(indices.size()),block=std::max(0,power(p,"feelnopain"))*count;
        int draws=power(p,"nodraw")>0?0:std::min({std::max(0,power(p,"darkembrace"))*count,
            int(v.at("draw").size()+v.at("discard").size()),std::max(0,10-handCount+count)});
        return result+std::min(threat,block)*3+draws*4;
    }
    double retrieval(const J &c,int energy) const {
        double result=cardValue(c);int price=cost(c);
        if(!((price>=0&&price<=energy)||corrupt(c)))return result*.25;
        result+=1.;
        for(const auto &e:recipe(c).at("effects"))if(e.at("primitive")=="GAIN_BLOCK")result+=std::min(threat,adjustedBlock(fixed(e,up(c))))*3.;
        return result;
    }
    double redraw(const J &indices) const {
        // Expected immediate utility over a PUBLIC multiset, never latent order.
        // Without draw/replenishment (No Draw), retaining the hand wins ties.
        if(indices.empty()||power(p,"nodraw")>0)return indices.empty()?0.:-1.;
        J pool=v.at("draw");
        if(pool.empty()) {pool=v.at("discard");for(const auto&i:indices)pool.push_back(hand.at(i.get<int>()));}
        if(pool.empty())return -1.;
        auto utility=[&](const J &c){return disposable(c)?-4.:retrieval(c,number(p,"energy"));};
        std::vector<double> values;for(const auto&c:pool)values.push_back(utility(c));
        const double replacement=preciseSum(values)/values.size();
        double score=0.;for(const auto&i:indices)score+=replacement-utility(hand.at(i.get<int>()))-.01;
        return score;
    }
    double copyValue(const J &c,int count,int energy,int slots) const {
        double result=0.;int price=cost(c);
        for(int i=0;i<std::min(count,std::max(0,slots));++i) {
            result+=retrieval(c,energy);if(price>=0&&price<=energy)energy-=price;
        }
        return result;
    }
    J preview(const std::string &id,bool upgraded=false) const {
        const auto &r=meta.at(id);int price=number(r.at("variants")[upgraded?1:0],"energy_cost");
        return J{{"id",id},{"type",r.at("type")},{"up",upgraded?1:0},{"base_cost",price},{"cost",price},{"free",false},{"damage",0}};
    }
    double discovery(J c,int energy,int slots,bool fromPool=false) const {
        int base=number(c,"base_cost");
        if(fromPool&&c.at("id")=="bloodforblood")base=std::max(0,base-number(v,"times_damaged"));
        c["base_cost"]=base;c["cost"]=base>=0?0:base;c["playable"]=true;
        double result=base==-1?cardValue(c)*std::max(0,energy):retrieval(c,energy);
        return result*(slots<=0?.25:1.);
    }
    double bottomValue(const J &cards,const J &indices) const {
        double result=0.;int position=0;
        for(const auto &i:indices) {
            const auto &c=cards.at(i.get<int>());int base=number(c,"base_cost"),turn=cost(c);
            double future=c.at("free").get<bool>()?0.:std::max(0,base)*2.;
            future/=1+(v.at("draw").size()+position++)/5.;
            bool usable=turn>=0&&turn<=number(p,"energy")&&c.value("playable",true);
            double opportunity=std::max(0.,cardValue(c))*(usable?1.:.2);
            if(usable&&threat>0)for(const auto &e:recipe(c).at("effects"))if(e.at("primitive")=="GAIN_BLOCK")
                opportunity+=std::min(threat,adjustedBlock(fixed(e,up(c))))*3;
            result+=future-opportunity-.25;
        }
        return result;
    }
    double generation(const J &step,int amount,const J &played,bool delayed=false) const {
        bool intoDraw=step.contains("primitive")&&step.at("primitive")=="GENERATE_RANDOM_DRAW";
        const auto contract=step.at(intoDraw?"random_draw_generation":"random_hand_generation").get<std::string>();
        const auto &pool=meta.at("$pools").at(contract);if(pool.empty())return 0.;
        std::vector<double> values;
        for(const auto &id:pool) {
            bool upgraded=contract=="COLORLESS_X_TURN_ZERO"&&up(played);
            J c=preview(id,upgraded);double value=std::max(0.,cardValue(c))*.5;
            int price=number(c,"base_cost");
            if(!intoDraw&&id=="bloodforblood")price=std::max(0,price-number(v,"times_damaged"));
            if((intoDraw||contract=="IRONCLAD_ATTACK_TURN_ZERO"||contract=="COLORLESS_X_TURN_ZERO")&&price>0)value+=std::min(3,price)*.5;
            values.push_back(value);
        }
        double expected=preciseSum(values)/pool.size();
        if(delayed)return expected*amount*.5;
        if(intoDraw)return expected*amount*std::min(.5,5./std::max(1,int(v.at("draw").size())+amount));
        int slots=std::max(0,10-std::max(0,int(hand.size())-1)),inHand=std::min(amount,slots);
        return expected*(inHand+(amount-inHand)*.25);
    }
    double autoplay(bool future=false) const {
        J candidates=v.at("top_candidates");
        if(future) {candidates=v.at("draw");for(const char *pile:{"hand","discard"})for(const auto &c:v.at(pile))candidates.push_back(c);}
        std::vector<double> values;
        for(auto c:candidates) {
            if(future&&c.at("id")=="mayhem")continue;
            const auto &r=recipe(c);J playable=c;playable["playable"]=true;
            double value=0.;
            if(future) {
                value=std::max(0.,cardValue(playable))*.5;
                for(const auto &e:r.at("effects")) {
                    if(e.at("primitive")=="GAIN_BLOCK")value+=fixed(e,up(c))*.2;
                    else if(e.at("primitive")=="LOSE_HP")value-=fixed(e,up(c))*2.;
                }
            }else {
                if(disposable(c))value=2.;
                else {
                    value=cardValue(playable)*.5;
                    if(attackCard(c)&&power(p,"entangled")>0)value=-1.;
                    else for(const auto &e:r.at("effects")) {
                        if(e.at("primitive")=="GAIN_BLOCK")value+=block(fixed(e,up(c)));
                        else if(e.at("primitive")=="LOSE_HP")value-=fixed(e,up(c))*2;
                    }
                    if(c.at("type")!="POWER")value-=std::max(0.,cardValue(c))*.25;
                }
                if(c.at("type")!="POWER") {
                    value+=std::max(0,power(p,"feelnopain"))*.5;
                    value+=std::max(0,power(p,"darkembrace"))*1.5;
                    value+=intrinsic(c);
                }
            }
            values.push_back(value);
        }
        std::sort(values.begin(),values.end());
        if(values.empty())return 0.;
        // Without a representable order posterior, do not invent uniform top
        // probabilities. This is the minimum candidate HEURISTIC utility, not
        // an expected value or a guarantee on battle outcome/HP.
        return v.at("uncertain_order").get<bool>()?values.front():preciseSum(values)/values.size();
    }
public:
    Rules(const J &view,const J &metadata):v(view),meta(metadata),p(v.at("player")),hand(v.at("hand")),monsters(v.at("monsters")),
        threat(incoming(v,power(p,"intangible")>0)) {
        for(int i=0;i<int(monsters.size());++i) if(live(monsters[i])) living.push_back(i);
    }
    double score(const J &a) const {
        if(a.at("kind")=="use_potion"||a.at("kind")=="discard_potion")return potionScore(a);
        if(v.at("input")=="CARD_SELECT" && a.at("kind")=="choose") {
            const auto task=v.value("selection","");int idx=a.at("choice_index");
            if(task=="FORETHOUGHT")return bottomValue(hand,v.at("selection_options").at(idx));
            if(task=="EXHAUST_MANY")return optionalExhaust(hand,v.at("selection_options").at(idx),int(hand.size()));
            if(task=="GAMBLE")return redraw(v.at("selection_options").at(idx));
            if(task=="CODEX") {
                if(idx==3)return 0.;
                const auto &c=v.at("selection_candidates").at(idx);
                std::vector<double> values;
                for(const char*pile:{"hand","draw","discard"})for(const auto&x:v.at(pile))values.push_back(cardValue(x));
                const double baseline=values.empty()?0.:preciseSum(values)/values.size();
                return (cardValue(c)-baseline)*.5-.01; // Optional delayed addition, not a free hand card.
            }
            if(task=="LIQUID_MEMORIES_POTION") {
                double value=0.;int available=10-int(hand.size());
                for(int i:v.at("selection_options").at(idx)) {
                    if(available--<=0)break;
                    auto c=v.at("selection_candidates").at(i);
                    if(number(c,"cost")>=0)c["cost"]=0;
                    value+=retrieval(c,number(p,"energy"));
                }
                return value;
            }
            if(task=="HEADBUTT")return cardValue(v.at("discard").at(idx));
            if(task=="DISCOVERY")return number(v,"selection_copies")*discovery(v.at("selection_candidates").at(idx),number(p,"energy"),10-int(hand.size()));
            if(task=="DUAL_WIELD")return copyValue(v.at("selection_candidates").at(idx),number(v,"selection_copies"),number(p,"energy"),10-int(hand.size()));
            if(task=="EXHUME"||task=="SECRET_TECHNIQUE"||task=="SECRET_WEAPON")return retrieval(v.at("selection_candidates").at(idx),number(p,"energy"));
            if(task=="ARMAMENTS")return upgradeValue(v.at("selection_candidates").at(idx),true,number(p,"energy"));
            const auto &c=hand.at(idx);
            if(task=="WARCRY") {
                double value=cardValue(c);
                if(number(c,"cost")<=number(p,"energy"))for(const auto &e:recipe(c).at("effects"))if(e.at("primitive")=="GAIN_BLOCK")
                    value+=std::min(threat,adjustedBlock(fixed(e,up(c))))*3;
                return -value;
            }
            if(task!="EXHAUST_ONE")throw NotPorted("selection score");
            if(disposable(c))return 100.;
            if(!c.value("playable",true))return 10.;
            return intrinsic(c);
        }
        if(a.at("kind")!="play_card") return a.at("kind")=="end_turn"?-1.:-.5;
        int idx=a.at("card_index"), target=a.value("target_index",-1);
        const auto &c=hand.at(idx), &r=recipe(c); bool u=up(c);
        const auto &effects=r.at("effects");
        std::map<std::pair<int,std::string>,int> delta; std::map<int,int> artifactUsed;
        int projectedHP=number(p,"current_hp"), buffer=power(p,"buffer"), strength=0, strengthLost=0, artifact=power(p,"artifact");
        double weakPrevented=0; bool rescued=false;
        double total= !r.at("self_copy").is_null() && r.at("self_copy").at("timing")=="EXHAUST" ? -1.1 : .1;
        auto status=[&](int i,const std::string &s) { return power(monsters[i],s)+delta[{i,s}]; };
        auto loss=[&](int n) -> std::pair<double,bool> {
            if(n<=0) return {0,false};
            if(buffer>0) {--buffer;return {-( (threat>=projectedHP?100.:0.)+std::max(1.,std::min(double(threat),8.)) ),false};}
            int effective=power(p,"intangible")>0?1:n;
            if(relic("tungstenrod")) effective=std::max(0,effective-1);
            if(effective<=0) return {0,false};
            int before=projectedHP; projectedHP=std::max(0,projectedHP-effective);
            double value=-effective*2.+std::max(0,power(p,"rupture"))*2.;
            if(projectedHP<=0) return {value-1000,true};
            if(threat<before && threat>=projectedHP) value-=100;
            return {value,false};
        };
        auto heal=[&](int raw,int maxGain=0) {
            if(relic("markofthebloom")) return 0.;
            int n=relic("magicflower")?(raw*3+1)/2:raw;
            int actual=std::min(n,std::max(0,number(p,"max_hp")+maxGain-projectedHP));
            int before=projectedHP; projectedHP+=actual;
            return actual*2.+(before<=threat && threat<projectedHP?100.:0.);
        };
        if(power(p,"panache")==1) for(int i:living) total+=nonAttack(monsters[i],number(p,"panache_damage"));
        if(attackCard(c)) total+=nonCardBlock(std::max(0,power(p,"rage")));
        for(int i=0;i<int(hand.size());++i) if(i!=idx) {
            const auto &resident=recipe(hand[i]).at("resident_rule");
            if(!resident.is_null() && resident.at("kind")=="OTHER_PLAY_HP_LOSS") {
                auto x=loss(number(resident,"amount")); total+=x.first;
                if(x.second) return -std::numeric_limits<double>::infinity();
            }
        }
        for(const auto &e:effects) {
            std::string primitive=e.at("primitive"), scope=(u&&!e.at("upgraded_target").is_null()?e.at("upgraded_target"):e.at("target"));
            std::vector<int> affected;
            if(scope=="ALL_ENEMIES" || scope=="RANDOM_ENEMY") affected=living;
            else if(scope=="TARGET_ENEMY" && target>=0 && target<int(monsters.size()) && live(monsters[target])) affected.push_back(target);
            if(!e.at("condition").is_null()) {
                std::string condition=e.at("condition");
                if(condition=="TARGET_HAS_VULNERABLE" && (target<0 || status(target,"vulnerable")<=0)) continue;
                if(condition=="TARGET_ATTACK_INTENT" && (target<0 || number(monsters[target],"move_hits")<=0)) continue;
                if(condition=="HAND_HAS_NO_ATTACK") {
                    bool found=false; for(int i=0;i<int(hand.size());++i) if(i!=idx&&attackCard(hand[i])) found=true;
                    if(found) continue;
                }
            }
            int n=fixed(e,u);
            if(e.at("amount_source")=="CURRENT_BLOCK") n=number(p,"block");
            if(e.at("amount_source")=="CURRENT_STRENGTH") n=power(p,"strength")+strength;
            if(!e.at("dynamic_damage").is_null()) n=dynamic(e,c);
            if(primitive=="UPGRADE_CARDS") {
                auto scope=e.at("combat_upgrade").at(u?"upgraded_scope":"base_scope").get<std::string>();
                int energy=std::max(0,number(p,"energy")-std::max(0,number(c,"cost")));
                double value=0.;bool found=false;
                for(int i=0;i<int(hand.size());++i)if(i!=idx) {
                    double x=upgradeValue(hand[i],true,energy);
                    if(scope=="CHOSEN_HAND") {value=found?std::max(value,x):x;found=true;} else value+=x;
                }
                total+=value;
                if(scope=="ALL_COMBAT_PILES") {
                    double future=0.;for(const char *pile:{"draw","discard"})for(const auto &x:v.at(pile))future+=upgradeValue(x,false,0)*.5;
                    total+=future;
                }
            }
            else if(primitive=="SELECT_BOTTOM_FREE"||primitive=="EXHAUST_UP_TO_HAND") {
                J others=J::array();for(int i=0;i<int(hand.size());++i)if(i!=idx)others.push_back(hand[i]);
                J groups=primitive=="SELECT_BOTTOM_FREE"?choiceGroups(int(others.size()),u?10:1,true,u):choiceGroups(int(others.size()),n);
                double best=-std::numeric_limits<double>::infinity();
                for(const auto &g:groups)best=std::max(best,primitive=="SELECT_BOTTOM_FREE"?bottomValue(others,g):optionalExhaust(others,g,int(others.size())));
                if(!groups.empty())total+=best;
            }
            else if(primitive=="COPY_SELECTED_CARD") {
                int energy=std::max(0,number(p,"energy")-std::max(0,cost(c)));double best=0.;bool found=false;
                for(int i=0;i<int(hand.size());++i)if(i!=idx&&has(e.at("selected_card_copy").at("card_types"),hand[i].at("type"))) {
                    double x=copyValue(hand[i],n,energy,11-int(hand.size()));best=found?std::max(best,x):x;found=true;
                }
                if(found)total+=best;
            }
            else if(primitive=="MOVE_SELECTED_CARD") {
                if(!e.at("selected_card_filter").is_null()) {
                    auto filter=e.at("selected_card_filter").get<std::string>();
                    const auto &pile=v.at(e.at("selected_card_source")=="DRAW_PILE"?"draw":"exhaust");
                    int energy=number(p,"energy")-std::max(0,cost(c));bool found=false;double best=0.;
                    for(const auto &x:pile)if(filter=="NON_EXHUME"?x.at("id")!="exhume":x.at("type")==J(filter=="SKILL"?"SKILL":"ATTACK")) {
                        double value=retrieval(x,energy);best=found?std::max(best,value):value;found=true;
                    }
                    if(found&&hand.size()<=10)total+=best;
                } else if(e.at("selected_card_source")=="HAND") {
                    if(draw(1)>0)total-=4.*n;
                    else {double best=0.;bool found=false;for(int i=0;i<int(hand.size());++i)if(i!=idx) {
                        double x=cardValue(hand[i]);best=found?std::min(best,x):x;found=true;}
                        if(found)total-=std::max(1.,best)*n;}
                } else if(n>0&&!v.at("discard").empty()) {
                    double best=-std::numeric_limits<double>::infinity();for(const auto &x:v.at("discard"))best=std::max(best,cardValue(x));total+=best*n;
                }
            }
            else if(primitive=="RETRIEVE_RANDOM_ATTACKS") {
                int energy=std::max(0,number(p,"energy")-std::max(0,cost(c)));std::vector<double> vals;
                for(const auto &x:v.at("draw"))if(attackCard(x))vals.push_back(retrieval(x,energy));
                int moved=std::min({n,int(vals.size()),std::max(0,11-int(hand.size()))});
                if(moved)total+=.5*moved*preciseSum(vals)/vals.size();
            }
            else if(primitive=="GENERATE_RANDOM_HAND"||primitive=="GENERATE_RANDOM_DRAW")total+=generation(e,n,c);
            else if(primitive=="GENERATE_X_HAND")total+=generation(e,std::max(0,number(p,"energy"))+(relic("chemicalx")?2:0),c);
            else if(primitive=="GENERATE_CHOICE_HAND") {
                int energy=std::max(0,number(p,"energy")-(corrupt(c)?0:std::max(0,cost(c))));std::vector<double> vals,weighted;
                for(const auto &id:meta.at("$pools").at("IRONCLAD_THREE"))vals.push_back(discovery(preview(id),energy,11-int(hand.size()),true));
                std::sort(vals.begin(),vals.end());
                for(int i=2;i<int(vals.size());++i)weighted.push_back((i*(i-1)/2)*vals[i]);
                auto count=vals.size();if(count>=3)total+=.5*preciseSum(weighted)/(count*(count-1)*(count-2)/6);
            }
            else if(primitive=="AUTOPLAY_TOP_CARD")total+=autoplay();
            else if(primitive=="REGISTER_DELAYED_DAMAGE")total+=std::max(0.,bombPressure(v,number(e.at("delayed_damage"),"turns"),n)-bombPressure(v));
            else if(primitive=="GAIN_BLOCK") total+=block(n);
            else if(primitive=="DRAW_CARDS") total+=draw(n);
            else if(primitive=="GAIN_ENERGY") total+=std::max(0,n)*3.;
            else if(primitive=="GAIN_STRENGTH") strength+=n;
            else if(primitive=="APPLY_LOSE_STRENGTH") {if(n>0&&artifact>0) --artifact; else strengthLost+=n;}
            else if(primitive=="GAIN_CURRENT_STRENGTH") {
                if(n<0&&artifact>0) --artifact;
                else {int current=power(p,"strength")+strength; strength+=std::max(-999,std::min(999,current+n))-current;}
            }
            else if(primitive=="LOSE_HP") {auto x=loss(n);total+=x.first;if(x.second) break;}
            else if(primitive=="HEAL") total+=heal(n);
            else if(primitive=="DIRECT_DAMAGE" || primitive=="AREA_DAMAGE" || primitive=="CURRENT_BLOCK_DAMAGE" || primitive=="DYNAMIC_DAMAGE") {
                for(int i:affected) {const auto &m=monsters[i]; int d=std::max(0,n+power(p,"strength"));
                    if(power(p,"weak")>0) d=int(d*.75);
                    if(status(i,"vulnerable")>0) d=int(d*1.5);
                    int loss=std::max(0,d-number(m,"block")),hp=number(m,"current_hp");
                    // Reference direct-damage heuristic intentionally has no Intangible cap.
                    total+=(hp>0&&loss>=hp?100:0)+std::min(loss,std::max(0,hp));}
            }
            else if(primitive=="ATTACK_SEQUENCE" || primitive=="AREA_LIFESTEAL" || primitive=="FATAL_ATTACK") {
                int hits=1,multiplier=1;
                if(primitive=="ATTACK_SEQUENCE") {const auto &s=e.at("attack_sequence");
                    hits=s.at("count_source")=="ENERGY_ON_USE"?std::max(0,number(p,"energy"))+(relic("chemicalx")?2:0):number(s,u?"upgraded_hits":"base_hits");
                    multiplier=number(s,u?"upgraded_strength_multiplier":"base_strength_multiplier",1);}
                bool random=scope=="RANDOM_ENEMY"&&affected.size()>1; int healed=0;
                for(int i:affected) {const auto &m=monsters[i]; double hit=std::max(0,n+power(p,"strength")*multiplier);
                    if(power(p,"weak")>0) hit*=.75;
                    if((primitive=="ATTACK_SEQUENCE"?status(i,"vulnerable"):power(m,"vulnerable"))>0) hit*=1.5;
                    int d=int(hit); if(power(m,"intangible")>0) d=std::min(1,d);
                    double effective=std::max(0.,d*(random?double(hits)/affected.size():hits)-number(m,"block"));
                    int hp=std::max(0,number(m,"current_hp")); bool killed=hp>0&&effective>=hp;
                    total+=std::min(effective,double(hp))+(!random&&killed?100:0);
                    healed+=int(std::min(effective,double(hp)));
                    if(!e.at("fatal_reward").is_null()&&killed&&!m.at("half_dead").get<bool>()&&power(m,"minion")==0&&power(m,"regrow")==0) {
                        const auto &rw=e.at("fatal_reward"); int reward=number(rw,u?"upgraded_amount":"base_amount");
                        if(rw.at("kind")=="MAX_HP") total+=reward*4.+heal(reward,reward);
                        else if(rw.at("kind")=="GOLD"&&!relic("ectoplasm")) total+=reward*.2;
                    }
                }
                if(primitive=="AREA_LIFESTEAL") total+=heal(healed);
            }
            else if(primitive=="APPLY_WEAK" || primitive=="APPLY_VULNERABLE" || primitive=="ENEMY_STRENGTH_LOSS") {
                for(int i:affected) {const auto &m=monsters[i];
                    if(power(m,"artifact")-artifactUsed[i]>0) {++artifactUsed[i];continue;}
                    double trigger=nonAttack(m,std::max(0,power(p,"sadistic")));
                    std::string name=primitive=="APPLY_WEAK"?"weak":(primitive=="APPLY_VULNERABLE"?"vulnerable":"strength");
                    int before=status(i,name);
                    if(name=="strength") {
                        int after=std::max(-999,std::min(999,before-n)); delta[{i,name}]=after-power(m,name);
                        bool weak=status(i,"weak")>0;
                        int prevented=std::max(0,monsterAttack(m,weak,before)-monsterAttack(m,weak,after));
                        double remaining=std::max(0.,threat-weakPrevented),useful=std::min(remaining,double(prevented));
                        if(!rescued&&remaining>=projectedHP&&projectedHP>remaining-useful) {total+=100;rescued=true;}
                        weakPrevented+=useful; total+=trigger+useful*1.5;
                        if(e.at("strength_reduction").at("restoration")=="NONE") total+=(before-after)*.5;
                    } else {
                        delta[{i,name}]+=n;
                        if(name=="vulnerable") {total+=n+trigger;continue;}
                        int immediate=before<=0?std::max(0,monsterAttack(m,false,power(m,"strength"))-monsterAttack(m,true,power(m,"strength"))):0;
                        double prevented=std::min(double(immediate),double(threat));
                        double old=std::max(0.,threat-weakPrevented); weakPrevented=std::min(double(threat),weakPrevented+prevented);
                        double after=std::max(0.,threat-weakPrevented);
                        double rescue=!rescued&&old>=number(p,"current_hp")&&after<number(p,"current_hp")?100.:0.;
                        rescued=rescued||rescue>0; total+=rescue+prevented*1.5+n*.5+trigger;
                    }
                }
            }
            else if(primitive=="CURRENT_BLOCK_GAIN") total+=nonCardBlock(n);
            else if(primitive=="END_TURN_BLOCK") total+=nonCardBlock(n)+std::min(4.,n*.5);
            else if(primitive=="RETAIN_BLOCK") {
                if(power(p,"barricade")==0) {int raw=0;for(int i:living) raw+=monsterAttack(monsters[i],power(monsters[i],"weak")>0,power(monsters[i],"strength"));
                    total+=2.+std::min(12.,std::max(0,number(p,"block")-raw)*.5);}
            }
            else if(primitive=="TEMP_RETALIATION") {
                for(int i:living) {const auto &m=monsters[i];int hits=number(m,"move_hits");if(hits<=0) continue;
                    int hit=power(m,"intangible")>0?std::min(n,1):n;
                    total+=std::min(std::max(0,hits*hit-number(m,"block")),std::max(0,number(m,"current_hp")))*.5;}
            }
            else if(primitive=="GAIN_INTANGIBLE") {
                int remaining=incoming(v,true); total+=std::max(0,threat-remaining)*1.5;
                if(threat>=number(p,"current_hp")&&number(p,"current_hp")>remaining) total+=100;
                total+=1./(1+std::max(0,power(p,"intangible")));
            }
            else if(primitive=="SELF_VULNERABLE") {
                if(artifact>0) --artifact;
                else if(power(p,"vulnerable")<=0) {int extra=0,raw=0;
                    for(int i:living) {const auto &m=monsters[i];int dmg=monsterAttack(m,power(m,"weak")>0,power(m,"strength"));raw+=dmg;
                        int hits=number(m,"move_hits"); if(hits>0) {int hit=dmg/hits;extra+=(int(hit*1.5)-hit)*hits;}}
                    int now=std::max(0,raw+extra-number(p,"block")), before=std::max(0,raw-number(p,"block"));
                    total-=(now-before)*2.+n*.5;if(before<projectedHP&&projectedHP<=now)total-=100;}
            }
            else if(primitive=="APPLY_NO_DRAW") {
                if(artifact>0) {--artifact;total-=1;}
                else if(power(p,"nodraw")==0) for(int i=0;i<int(hand.size());++i) if(i!=idx&&number(hand[i],"cost")<=number(p,"energy"))
                    for(const auto &s:recipe(hand[i]).at("effects")) if(s.at("primitive")=="DRAW_CARDS") total-=draw(fixed(s,up(hand[i])));
            }
            else if(primitive=="APPLY_NO_BLOCK") {
                if(artifact>0) {--artifact;total-=1;}
                else {int defensive=0,count=0;for(const char *pile:{"draw","discard"})for(const auto &other:v.at(pile)) {
                    ++count;bool found=false;for(const auto &s:recipe(other).at("effects")) found=found||s.at("primitive")=="GAIN_BLOCK";defensive+=found;}
                    total-=2.+(count?6.*defensive/count:0.);}
            }
            else if(primitive=="GAIN_ARTIFACT") {
                total+=n*2./(1+std::max(0,artifact));
                if(artifact==0) for(int i=0;i<int(hand.size());++i) if(i!=idx&&cost(hand[i])<=number(p,"energy")) {
                    const auto &es=recipe(hand[i]).at("effects");bool found=false;
                    for(const auto &s:es)found=found||s.at("primitive")=="APPLY_NO_BLOCK";
                    if(found) {for(const auto &s:es)if(s.at("primitive")=="GAIN_BLOCK")total+=block(fixed(s,up(hand[i])));break;}
                }
                artifact+=n;
            }
            else if(primitive=="EXHAUST_RANDOM_HAND" || primitive=="EXHAUST_CHOSEN_HAND") {
                std::vector<double> vals;for(int i=0;i<int(hand.size());++i)if(i!=idx)vals.push_back(consume(hand[i]));
                if(n>0&&!vals.empty()) {
                    int count=std::min(n,int(vals.size()));double sum=0;
                    if(primitive=="EXHAUST_CHOSEN_HAND") {std::sort(vals.begin(),vals.end(),std::greater<double>());for(int i=0;i<count;++i)sum+=vals[i];total+=sum;}
                    else {for(auto value:vals)sum+=value;total+=count*sum/vals.size();}
                }
            }
            else if(primitive=="EXHAUST_HAND_GROUP") {
                const auto &group=e.at("hand_exhaust");int count=0;
                double consumedValue=0;
                for(int i=0;i<int(hand.size());++i)if(i!=idx&&(group.at("filter")=="ALL"||!attackCard(hand[i]))) {++count;consumedValue+=consume(hand[i]);}
                total+=consumedValue;
                int events=count+int(r.at("exhaust").at(u?1:0).get<bool>()||corrupt(c));
                int amount=events*std::max(0,power(p,"feelnopain")),blockEvents=power(p,"feelnopain")>0?events:0;
                if(group.at("payoff")=="BLOCK") {amount+=count*adjustedBlock(n);blockEvents+=adjustedBlock(n)>0?count:0;}
                total+=nonCardBlock(amount,blockEvents);total+=draw(events*std::max(0,power(p,"darkembrace")),count);
                if(group.at("payoff")=="ATTACK")for(int i:affected) {
                    const auto &m=monsters[i];double hit=std::max(0,n+power(p,"strength"));
                    if(power(p,"weak")>0)hit*=.75;if(status(i,"vulnerable")>0)hit*=1.5;
                    int damage=int(hit);if(power(m,"intangible")>0)damage=std::min(damage,1);
                    int loss=std::max(0,count*damage-number(m,"block")),hp=std::max(0,number(m,"current_hp"));
                    total+=std::min(loss,hp)+(hp&&loss>=hp?100:0);
                }
            }
            else if(primitive=="REGISTER_TURN_TRIGGER") {
                const auto &trigger=e.at("turn_trigger");
                for(const auto &t:trigger.at("effects")) {
                    int value=fixed(t,u);
                    if(t.at("kind")=="SELF_HP_LOSS") {
                        if(trigger.at("timing")=="END") {auto x=loss(value);total+=x.first;if(x.second)break;}
                        else {int effective=std::max(0,value-int(relic("tungstenrod")));total-=effective*2.;if(projectedHP<=effective)total-=100;}
                    }else if(t.at("kind")=="GENERATE_RANDOM_HAND")total+=generation(t,value,c,true);
                    else if(t.at("kind")=="AUTOPLAY_TOP")total+=value*autoplay(true);
                    else if(t.at("kind")=="NON_ATTACK_AREA_DAMAGE") {
                        for(int i:living)total+=nonAttack(monsters[i],value)*.5;
                    }else if(t.at("kind")=="DRAW"||t.at("kind")=="STRENGTH"||t.at("kind")=="ENERGY")total+=value*(t.at("kind")=="DRAW"?2.:1.5);
                    else throw NotPorted("turn trigger scoring");
                }
            }
            else if(primitive=="GENERATE_CARD") {
                const J *generated=&c;
                if(e.at("generated_card_source")=="FIXED_BASE") {auto id=key(e.at("generated_card_id"));generated=meta.contains(id)?&meta.at(id):nullptr;}
                double per=generated?((generated->at("type")=="CURSE"||generated->at("type")=="STATUS")?-3.:1.):0.;
                if(n>0) {
                    if(e.at("generated_card_destination")=="HAND") {int slots=std::max(0,10-std::max(0,int(hand.size())-1)), h=std::min(n,slots);total+=per*(h*.5+n-h);}
                    else total+=n*per*(e.at("generated_card_destination")=="DRAW_PILE_RANDOM"?1.5:1.);
                }
            }
            else if(primitive=="REGISTER_DRAW_TRIGGER") {
                const auto &t=e.at("draw_trigger");int matches=0;
                for(const char *pile:{"draw","discard"})for(const auto &x:v.at(pile)) matches+=has(t.at("card_types"),x.at("type"));
                int opportunities=std::min(2,matches);
                total+=t.at("effect")=="DRAW"?double(opportunities*n*3):std::min(12.,opportunities*n*.25*living.size());
            }
            else if(primitive=="REGISTER_EXHAUST_TRIGGER") {
                int sources=0;for(const char *pile:{"hand","draw","discard"})for(const auto &x:v.at(pile)) {
                    const auto &rx=recipe(x);bool found=rx.at("exhaust").at(up(x)?1:0).get<bool>()||rx.at("ethereal").at(up(x)?1:0).get<bool>()||corrupt(x);
                    for(const auto &s:rx.at("effects")) found=found||s.at("primitive")=="EXHAUST_RANDOM_HAND"||s.at("primitive")=="EXHAUST_CHOSEN_HAND"||s.at("primitive")=="EXHAUST_HAND_GROUP"||s.at("primitive")=="EXHAUST_UP_TO_HAND";
                    sources+=found;
                }
                total+=std::min(12.,std::min(2,sources)*n*(e.at("exhaust_trigger").at("effect")=="DRAW"?3.:.75));
            }
            else if(primitive=="REGISTER_CORRUPTION") {
                if(power(p,"corruption")!=0)total-=2;
                else {int saved=0,count=0;for(int i=0;i<int(hand.size());++i)if(i!=idx&&skill(hand[i])){saved+=std::max(0,cost(hand[i]));++count;}
                    int opportunities=std::min(3,count);total+=std::max(0,saved-std::max(0,cost(c)))*3+opportunities;
                    total+=opportunities*(std::max(0,power(p,"feelnopain"))*.5+std::max(0,power(p,"darkembrace"))*2);}
            }
            else if(primitive=="REGISTER_ATTACK_REPEAT") {
                int energy=std::max(0,number(p,"energy")-(corrupt(c)?0:std::max(0,cost(c))));
                std::vector<int> choices;for(int i=0;i<int(hand.size());++i)if(i!=idx&&attackCard(hand[i]))choices.push_back(i);
                std::stable_sort(choices.begin(),choices.end(),[&](int a,int b){return cardValue(hand[a])>cardValue(hand[b]);});
                std::vector<double> values;for(int i:choices){int cost_=cost(hand[i]);if(cost_<0||cost_>energy||!hand[i].value("playable",true))continue;
                    energy-=cost_;values.push_back(std::max(0.,cardValue(hand[i])*4-8));}
                int start=std::max(0,power(p,"doubletap"));for(int i=start;i<std::min(int(values.size()),start+n);++i)total+=values[i];
            }
            else if(primitive=="MODIFY_HAND_COST") {
                const auto &change=e.at("hand_cost");
                if(change.at("selection")=="RANDOM_POSITIVE") {
                    std::vector<int> chosen;for(int i=0;i<int(hand.size());++i)if(i!=idx&&cost(hand[i])>0)chosen.push_back(i);
                    if(chosen.empty())for(int i=0;i<int(hand.size());++i)if(i!=idx&&number(hand[i],"base_cost")>0)chosen.push_back(i);
                    double sum=0;for(int i:chosen)sum+=std::max(0,cost(hand[i]))*3+std::max(0,number(hand[i],"base_cost"))*.5;
                    if(!chosen.empty())total+=sum/chosen.size();
                } else {for(int i=0;i<int(hand.size());++i)if(i!=idx)total+=std::max(0,cost(hand[i])-n)*3;
                    if(change.at(u?"upgraded_duration":"base_duration")=="COMBAT")for(int i=0;i<int(hand.size());++i)if(i!=idx)total+=std::max(0,number(hand[i],"base_cost")-n)*.5;}
            }
            else if(primitive=="REGISTER_EVENT_TRIGGER") {
                const auto event=e.at("event_trigger").at("event").get<std::string>(); int sources=0;
                for(int i=0;i<int(hand.size());++i)if(i!=idx) {
                    const auto &x=hand[i];bool found=false;
                    if(event=="FIFTH_CARD_PLAYED")sources+=x.value("playable",true);
                    else if(event=="ATTACK_PLAYED")sources+=attackCard(x)&&x.value("playable",true);
                    else {for(const auto &s:recipe(x).at("effects")) {
                        if(event=="ENEMY_DEBUFF_APPLIED")sources+=s.at("primitive")=="APPLY_WEAK"||s.at("primitive")=="APPLY_VULNERABLE"||s.at("primitive")=="ENEMY_STRENGTH_LOSS";
                        else if(event=="SELF_HP_LOST")found=found||s.at("primitive")=="LOSE_HP";
                        else found=found||s.at("primitive")=="GAIN_BLOCK"||s.at("primitive")=="CURRENT_BLOCK_GAIN";
                    }sources+=found;}
                }
                if(event=="FIFTH_CARD_PLAYED")total+=n*std::min(1.,sources/5.)*living.size()*.5;
                else if(event=="ENEMY_DEBUFF_APPLIED")total+=n*std::min(3,sources)*.5;
                else if(event=="ATTACK_PLAYED")total+=std::min(threat,n*sources)*.5;
                else if(event=="SELF_HP_LOST")total+=n*std::min(3,sources+int(power(p,"brutality")>0)+int(power(p,"combust")>0))*1.5;
                else total+=n*std::min(3,sources+int(power(p,"rage")>0)+int(power(p,"feelnopain")>0))*.5;
            }
            else if(primitive=="SHUFFLE_DISCARD_INTO_DRAW") {if(!v.at("discard").empty()&&relic("theabacus"))total+=nonCardBlock(6);}
            else if(primitive!="ON_EXHAUST_ENERGY"&&primitive!="ON_HP_LOSS_COST_REDUCTION"&&primitive!="GROW_PLAYED_DAMAGE")
                throw NotPorted("scorer: "+primitive);
        }
        bool medical=relic("medicalkit")&&has(r.at("traits"),"UNPLAYABLE_STATUS");
        bool candle=relic("bluecandle")&&(has(r.at("traits"),"UNPLAYABLE_CURSE")||has(r.at("traits"),"PLAYABLE_CURSE"));
        if(candle) {auto x=loss(1);total+=x.first;if(x.second)return total;
            if(!r.at("ethereal").at(u?1:0).get<bool>())total+=std::max(0.,consume(c))*.25;}
        bool groupExhaust=false;for(const auto &e:effects)groupExhaust=groupExhaust||e.at("primitive")=="EXHAUST_HAND_GROUP";
        if(!groupExhaust&&((r.at("exhaust").at(u?1:0).get<bool>()&&has(r.at("traits"),"PLAYABLE_CURSE"))||corrupt(c)||medical||candle)) {
            total+=nonCardBlock(std::max(0,power(p,"feelnopain")));total+=draw(std::max(0,power(p,"darkembrace")));}
        if(medical)total+=consume(c);
        if(strength) {int attacks=0;for(int i=0;i<int(hand.size());++i)if(i!=idx&&attackCard(hand[i]))++attacks;
            int lasting=std::max(0,strength-strengthLost),temporary=strength-lasting;
            total+=strength*std::min(3,attacks)*3.+lasting*2.5+temporary*.25;}
        return total;
    }
    double potionScore(const J &a) const {
        const int slot=number(a,"potion_index",-1);
        if(!v.contains("potions")||slot<0||slot>=int(v.at("potions").size()))throw NotPorted("public potion slot");
        const auto id=v.at("potions")[slot].get<std::string>();
        if(a.at("kind")=="discard_potion")return -potionReserve(id)-1.;
        const auto &contracts=meta.at("$potions");
        if(!contracts.contains(id))throw NotPorted("potion mechanism: "+id);
        const auto &c=contracts.at(id);const auto effect=c.at("effect").get<std::string>();
        const int n=number(c,"amount")*(relic("sacredbark")?2:1),target=number(a,"target_index",-1);
        const int room=10-int(hand.size()),energy=number(p,"energy"),missing=number(p,"max_hp")-number(p,"current_hp");
        double value=0.;
        if(effect=="passive_revive")throw NotPorted("Fairy cannot be manually used");
        if(effect=="block")value=std::min({threat,n,std::max(0,999-number(p,"block"))});
        else if(effect=="damage"||effect=="damage_all") {
            for(int j:living)if(effect=="damage_all"||j==target) {
                const auto &m=monsters[j];int hit=power(m,"intangible")>0?1:n;
                int loss=std::min(number(m,"current_hp"),std::max(0,hit-number(m,"block")));
                if(m.at("powers").contains("invincible"))loss=std::min(loss,std::max(0,power(m,"invincible")));
                value+=loss*.3;
                if(loss>=number(m,"current_hp"))value+=attack(m,p,power(m,"weak")>0,power(m,"strength"),power(p,"intangible")>0);
            }
        } else if(effect=="weak"||effect=="vulnerable") {
            if(target<0||target>=int(monsters.size()))throw NotPorted("potion public target");
            if(power(monsters[target],"artifact")==0 && power(monsters[target],effect)==0) {
                if(effect=="weak") {auto boosted=v;boosted["monsters"][target]["powers"]["weak"]=n;value=threat-incoming(boosted,power(p,"intangible")>0);}
                else for(const auto &x:hand)if(attackCard(x)&&cost(x)<=energy)value+=3.;
            }
        } else if(effect=="heal_percent")value=std::min(missing,number(p,"max_hp")*n/100);
        else if(effect=="max_hp")value=n+std::min(n,missing);
        else if(effect=="regen")value=std::min(missing,n*(n+1)/2)*.7;
        else if(effect=="strength"||effect=="temporary_strength"||effect=="dexterity"||effect=="temporary_dexterity") {
            const bool strength=effect.find("strength")!=std::string::npos;
            for(const auto &x:hand)if(cost(x)<=energy && (strength?attackCard(x):skill(x)))value+=n;
            if(effect=="strength")value+=n;
            if(!strength)value=std::min(double(threat),value);
        } else if(effect=="energy") {
            for(int i=0;i<int(hand.size());++i) {
                const auto &x=hand[i];if(cost(x)<=energy||cost(x)>energy+n)continue;
                for(int t:living) {auto boosted=v;boosted["player"]["energy"]=energy+n;
                    value=std::max(value,Rules(boosted,meta).score(J{{"kind","play_card"},{"card_index",i},{"target_index",t}}));}
            }
        } else if(effect=="plated_armor"||effect=="metallicize")value=n*1.5;
        else if(effect=="thorns") {int hits=0;for(int j:living)hits+=std::max(0,number(monsters[j],"move_hits"));value=n*hits*.75;}
        else if(effect=="ritual")value=n*4.;
        else if(effect=="artifact")value=power(p,"artifact")==0?3.5:0.;
        else if(effect=="draw"||effect=="draw_random_cost")value=power(p,"nodraw")?0.:std::min(room,n)*(energy>0?2.:.2);
        else if(effect=="redraw") {for(const auto &x:hand)if(disposable(x)||cost(x)>energy)value+=1.5;}
        else if(effect=="retrieve") {std::vector<double> values;for(auto x:v.at("discard")){if(number(x,"cost")>=0)x["cost"]=0;values.push_back(retrieval(x,energy));}
            std::sort(values.rbegin(),values.rend());for(int i=0;i<std::min({n,room,int(values.size())});++i)value+=values[i];}
        else if(effect=="exhaust") {for(const auto &x:hand)if(disposable(x))value+=consume(x)+nonCardBlock(std::max(0,power(p,"feelnopain")));}
        else if(effect=="upgrade_hand") {for(const auto &x:hand)value+=std::max(0.,upgradeValue(x,true,energy));}
        else if(effect.rfind("discovery_",0)==0)value=std::min(room,n)*5.;
        else if(effect=="autoplay")value=std::min(n,int(v.at("draw").size()+v.at("discard").size()))*3.;
        else if(effect=="duplication") {for(int i=0;i<int(hand.size());++i)if(hand[i].value("playable",true))
            for(int t:living)value=std::max(value,score(J{{"kind","play_card"},{"card_index",i},{"target_index",t}})*.5);}
        else if(effect=="refill") {int empty=1;for(const auto &x:v.at("potions"))if(potionReserve(x.get<std::string>())==0)++empty;value=empty*5.;}
        else if(effect=="escape")value=threat>=number(p,"current_hp")?100.:0.;
        if(relic("toyornithopter"))value+=std::min(5,missing);
        return value-potionReserve(id);
    }
    J scores(const J &actions) const {J out=J::array();for(const auto &a:actions)out.push_back(score(a));return out;}
    J choose(const J &actions,std::vector<double>* captured=nullptr) const {
        if(actions.empty()) throw std::invalid_argument("no non-potion legal action");
        std::size_t chosen=0;double best=score(actions[0]);
        if(captured){captured->clear();captured->reserve(actions.size());captured->push_back(best);}
        for(std::size_t i=1;i<actions.size();++i){double x=score(actions[i]);
            if(captured)captured->push_back(x);if(x>best){best=x;chosen=i;}}
        return actions[chosen];
    }
};

inline double combatValue(const J &v,int outcome) {
    spire_perf::Scope timing(spire_perf::Phase::Value);
    const auto &p=v.at("player"); double maxhp=std::max(1,number(p,"max_hp",80));
    double hp=std::min(1.,std::max(0.,number(p,"current_hp")/maxhp));
    if(outcome==2)return -3.; if(outcome==1)return 2.+hp;
    if(outcome==3)return .5+hp; // Escape is terminal survival, never a victory.
    int monsterHP=0,maxMonsterHP=0;
    for(const auto &m:v.at("monsters"))if(live(m)){monsterHP+=number(m,"current_hp");maxMonsterHP+=number(m,"max_hp");}
    double delayed=std::min(.1,bombPressure(v)/std::max(1,maxMonsterHP)*.1);
    return hp-double(monsterHP)/std::max(1,maxMonsterHP)-std::min(1.,incoming(v,power(p,"intangible")>0)/maxhp)*.5+delayed;
}
} // namespace spire_search
