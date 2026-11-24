#pragma once
#include "Engine.h"
#include "../draw_belief/SimulatorMechanisms.h"
#include <set>

// Opt-in certificate layer. It never adds visits, alters PUCT/backups, reads
// the live game's RNG, or executes an extra simulator action. A sampled win is
// NOT itself a certificate. Only the closed, public, straight-line fragment
// below may produce one. Unknown capabilities leave normal search untouched.
namespace spire_terminal_proof {
using J=nlohmann::json;
inline constexpr auto version="public-terminal-proof-v1";

inline const std::set<std::string>& straightPrimitives() {
    static const std::set<std::string> s={"DIRECT_DAMAGE","AREA_DAMAGE","CURRENT_BLOCK_DAMAGE",
        "DYNAMIC_DAMAGE","GROW_PLAYED_DAMAGE","GAIN_BLOCK","GAIN_ENERGY","GAIN_STRENGTH",
        "APPLY_LOSE_STRENGTH","GAIN_CURRENT_STRENGTH","LOSE_HP","APPLY_WEAK","APPLY_VULNERABLE",
        "ATTACK_SEQUENCE","RETAIN_BLOCK","CURRENT_BLOCK_GAIN","END_TURN_BLOCK","TEMP_RETALIATION",
        "SELF_VULNERABLE","APPLY_NO_DRAW","APPLY_NO_BLOCK","GAIN_ARTIFACT","GAIN_INTANGIBLE",
        "REGISTER_ATTACK_REPEAT","ENEMY_STRENGTH_LOSS","ON_HP_LOSS_COST_REDUCTION"};
    return s;
}
inline bool recipe(const J& r,bool straight) {
    if(!r.value("audited",false)||!r.contains("effects"))return false;
    // Permanent rewards/copy creation need a separate capability proof.
    if(r.contains("self_copy")&&!r.at("self_copy").is_null())return false;
    if(r.contains("master_deck")&&!r.at("master_deck").is_null()) {
        const auto& master=r.at("master_deck");
        // Ascender's Bane's removal prohibition creates no in-combat reward.
        if(master.value("removal","")!="EXCLUDED"||master.value("max_hp_loss_on_remove",-1)!=0)return false;
    }
    static const std::set<std::string> closed={"DRAW_CARDS","EXHAUST_RANDOM_HAND","ON_EXHAUST_ENERGY",
        "REGISTER_DRAW_TRIGGER","REGISTER_EXHAUST_TRIGGER","REGISTER_CORRUPTION","MODIFY_HAND_COST",
        "EXHAUST_CHOSEN_HAND","UPGRADE_CARDS","MOVE_SELECTED_CARD","COPY_SELECTED_CARD",
        "SELECT_BOTTOM_FREE","EXHAUST_UP_TO_HAND","RETRIEVE_RANDOM_ATTACKS","SHUFFLE_DISCARD_INTO_DRAW",
        "AUTOPLAY_TOP_CARD","REGISTER_DELAYED_DAMAGE"};
    for(auto group:{"effects","intrinsic_effects"})if(r.contains(group))for(const auto& e:r.at(group)) {
        const auto p=e.value("primitive","");
        if(!straightPrimitives().count(p)&&(straight||!closed.count(p)))return false;
        if(straight&&(e.value("target","")=="RANDOM_ENEMY"||e.value("upgraded_target",J())=="RANDOM_ENEMY"))return false;
    }
    return true;
}

// Reviewed capability contracts, NOT combat admission lists. New/unreviewed
// relics/powers simply make the certificate unknown. In particular no healing,
// revival, random generation, turn-dependent victory healing, or permanent
// rewards may hide behind the current-HP upper bound.
inline bool closedContext(const J& v) {
    static const std::set<std::string> relics={"burningblood","blackblood","anchor","ancientteaset",
        "bagofmarbles","bagofpreparation","bronzescales","vajra","oddlysmoothstone","orichalcum",
        "pen nib","pennib","nunchaku","kunai","shuriken","ornamentalfan","letteropener",
        "artofwar","happyflower","lantern","akabeko","theboot","boot","centennialpuzzle",
        "horncleat","captainswheel","mercuryhourglass","sundial","calipers","frozeneye",
        "redskull","threadandneedle","torii","tungstenrod","oddmushroom","paperphrog"};
    static const std::set<std::string> powers={"strength","dexterity","artifact","weak","frail","vulnerable",
        "entangled","constricted","confused","hex","losestrength","losedexterity","noblock","nodraw",
        "intangible","barricade","corruption","doubletap","duplication","freeattackpower","pennib",
        "darkembrace","evolve","firebreathing","feel nopain","feelnopain","demonform","brutality",
        "combust","juggernaut","metallicize","nextturnblock","platedarmor","rage","rupture","thorns",
        "flamebarrier","panache","thebomb","mayhem","drawcardnextturn","energized","doubledamage",
        "drawreduction","ritual","buffer","surrounded"};
    if(v.value("character","")!="IRONCLAD"||!v.value("powers_complete",false))return false;
    for(const auto& r:v.at("relics"))if(!r.is_string()||!relics.count(r.get<std::string>()))return false;
    for(auto it=v.at("player").at("powers").begin();it!=v.at("player").at("powers").end();++it)
        if(!powers.count(it.key()))return false;
    return true;
}
inline std::string boundGap(const J& root,const J& meta) {
    if(!closedContext(root))return "unreviewed_context_capability";
    if(root.value("input","")!="PLAYER_NORMAL"||!root.value("resolved",false))return "unsettled_root";
    // First version deliberately does not claim a bound with potion healing,
    // generation or replacement. Even a zero-value potion cannot be discarded.
    for(const auto& p:root.at("potions"))if(p!="emptypotionslot"&&p!="invalid")return "potion_opportunity";
    for(const auto* pile:{"hand","draw","discard","exhaust","held"})for(const auto& c:root.at(pile)) {
        const auto id=c.at("id").get<std::string>();
        if(!meta.contains(id)||!recipe(meta.at(id),false))return "card_resource_capability:"+id;
    }
    return "";
}

// Equality check only on OWNED SAMPLED worlds, never on SessionHistory::source_.
// Covers direct nextLong() calls which do not increment the public counter.
// No RNG word/counter is emitted or retained in a witness/model input.
struct EntropyStamp {
    std::array<std::uint64_t,18> data{};
    explicit EntropyStamp(const sts::BattleContext& b) {
        int i=0;for(const auto* r:{&b.aiRng,&b.cardRandomRng,&b.miscRng,&b.monsterHpRng,&b.potionRng,&b.shuffleRng}) {
            data[i++]=r->seed0;data[i++]=r->seed1;data[i++]=std::uint64_t(r->counter);
        }
    }
    bool unchanged(const sts::BattleContext& b)const{return data==EntropyStamp(b).data;}
};
struct Cache { J witness; void clear(){witness=J();} };
inline thread_local Cache* activeCache=nullptr;
struct CacheScope {
    Cache* previous;
    explicit CacheScope(Cache& c):previous(activeCache){activeCache=&c;}
    ~CacheScope(){activeCache=previous;}
};

class Collector {
    const J& root_;const J& meta_;J path_=J::array(),best_,carried_;
    std::string gap_;bool valid_=false;
    unsigned wins_=0,certificates_=0,randomRejects_=0,mechanismRejects_=0;
public:
    bool enabled=false;
    Collector(const J& budget,const J& root,const J& meta,const J& rich):root_(root),meta_(meta) {
        if(budget.contains("terminal_proof")) {
            if(budget.at("terminal_proof")!=version)throw std::invalid_argument("terminal_proof_version");
            enabled=true;gap_=boundGap(root,meta);
        }
        if(!activeCache)return;
        if(!enabled){activeCache->clear();return;}
        const auto& w=activeCache->witness;
        if(w.is_object()&&w.contains("steps"))for(std::size_t i=0;i<w.at("steps").size();++i) {
            const auto& s=w.at("steps")[i];
            if(s.at("public")==root&&s.at("rich")==rich) {
                carried_=w;carried_["steps"]=J::array();
                for(;i<w.at("steps").size();++i)carried_["steps"].push_back(w.at("steps")[i]);
                break;
            }
        }
        // A stale branch must not survive a public-state/counter mismatch.
        activeCache->clear();
    }
    void begin(){if(enabled){path_=J::array();valid_=closedContext(root_);}}
    bool tracking()const{return enabled&&valid_;}
    void before(const J& view,const J& rich,const J& action) {
        if(!tracking())return;
        if(action.value("kind","")!="play_card"||view.value("input","")!="PLAYER_NORMAL"||!view.value("resolved",false)) {valid_=false;return;}
        const auto i=action.at("card_index").get<std::size_t>();const auto id=view.at("hand").at(i).at("id").get<std::string>();
        if(!meta_.contains(id)||!recipe(meta_.at(id),true)){valid_=false;++mechanismRejects_;return;}
        path_.push_back({{"public",view},{"rich",rich},{"action",action}});
    }
    void after(bool noRandom,const std::vector<std::string>& observations) {
        if(!tracking())return;
        if(!noRandom){valid_=false;++randomRejects_;}
        // Ordinary visible discard-cost reset/Rampage growth is deterministic.
        // Any other deck event (including draws without a new RNG call) is out
        // of this proof fragment. The last record is the settled snapshot.
        bool simple=!observations.empty();
        for(std::size_t i=0;simple&&i+1<observations.size();++i) {
            const auto event=J::parse(observations[i]);
            simple=event.value("kind",-1)==int(spire_belief::Mechanism::Modify)&&
                (event.value("copy",-1)==int(spire_belief::trace::CopyRule::DiscardReset)||
                 event.value("copy",-1)==int(spire_belief::trace::CopyRule::AddDamage));
        }
        if(!simple){valid_=false;++mechanismRejects_;}
    }
    void complete(const J& view,int terminal) {
        if(!enabled||terminal!=1)return;++wins_;
        if(!valid_||path_.empty()||spire_search::number(view.at("player"),"current_hp")<=0)return;
        ++certificates_;J w={{"version",version},{"steps",path_},{"terminal",view},{"outcome",1}};
        if(best_.is_null()||optimal(w)>optimal(best_)||(optimal(w)==optimal(best_)&&
            w.at("steps").size()<best_.at("steps").size()))best_=std::move(w);
    }
    bool optimal(const J& w)const {
        if(!gap_.empty()||!w.is_object()||w.at("steps").empty())return false;
        const auto& v=w.at("terminal");
        return w.at("outcome")==1&&v.at("player").at("current_hp")==root_.at("player").at("current_hp")&&
            v.at("player").at("max_hp")==root_.at("player").at("max_hp")&&v.at("potions")==root_.at("potions");
    }
    J finish(const J& statistical,const J& legal,bool success) {
        if(!enabled)return J();
        J selected;
        if(success&&optimal(carried_))selected=carried_;
        if(success&&optimal(best_)&&(selected.is_null()||best_.at("steps").size()<selected.at("steps").size()))selected=best_;
        const bool used=!selected.is_null();
        const auto action=used?selected.at("steps")[0].at("action"):statistical;
        if(used&&std::find(legal.begin(),legal.end(),action)==legal.end())throw std::logic_error("terminal_proof_illegal");
        if(activeCache){if(used)activeCache->witness=selected;else activeCache->clear();}
        return {{"version",version},{"statistical_action",statistical},{"selected_action",action},
            {"used",used},{"changed",used&&action!=statistical},{"carried",used&&selected==carried_},
            {"sampled_wins",wins_},{"public_certificates",certificates_},{"random_rejections",randomRejects_},
            {"mechanism_rejections",mechanismRejects_},{"bound_gap",gap_},{"extra_simulator_steps",0},
            {"witness",selected},{"feasible_witness",used?J():best_}};
    }
};
} // namespace spire_terminal_proof
