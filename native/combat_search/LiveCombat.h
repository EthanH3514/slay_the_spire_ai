#pragma once
#include "SessionHistory.h"

namespace spire_session {
inline constexpr auto liveRootVersion="public-live-resolved-roots-v1";

// Unlike the simulator's state codec, this accepts visible intent signatures,
// not exact move IDs. Ambiguous signatures are refused, never assigned an
// arbitrary uniform prior. A later intent-belief service can replace this seam.
inline MonsterId liveMonsterId(const std::string& value) {
    static const std::map<std::string,std::string> aliases={
        {"lousenormal","redlouse"},{"lousedefensive","greenlouse"},
        {"slaverblue","blueslaver"},{"slaverred","redslaver"},
        {"fungusbeast","fungibeast"},{"gremlinfat","fatgremlin"},
        {"gremlinthief","sneakygremlin"},{"gremlinwarrior","madgremlin"},
        {"gremlintsundere","shieldgremlin"},{"champ","thechamp"},
        {"collector","thecollector"},{"maw","themaw"},{"guardian","theguardian"}};
    auto k=spire_search::key(value);auto a=aliases.find(k);if(a!=aliases.end())k=a->second;
    for(int i=1;i<=int(MonsterId::WRITHING_MASS);++i)
        if(spire_search::key(monsterIdStrings[i])==k)return static_cast<MonsterId>(i);
    throw Failure("unmapped_live_monster");
}
inline std::string powerKey(const std::string& name) {
    auto k=spire_search::key(name);
    static const std::map<std::string,std::string> aliases={
        {"weakened","weak"},{"confusion","confused"},{"intangibleplayer","intangible"},
        {"strengthdown","losestrength"},{"dexteritydown","losedexterity"},
        {"entangle","entangled"},{"drawcard","drawcardnextturn"},
        {"energizedblue","energized"},{"regeneration","regen"}};
    auto found=aliases.find(k);if(found!=aliases.end())return found->second;
    if(k.rfind("thebomb",0)==0&&k.substr(7).find_first_not_of("0123456789")==std::string::npos)return "thebomb";
    return k;
}
inline RelicId liveRelicId(const std::string& value) {
    const auto k=spire_search::key(value);
    for(int i=0;i<int(RelicId::INVALID);++i)
        if(spire_search::key(relicIds[i])==k)return static_cast<RelicId>(i);
    throw Failure("unmapped_live_relic");
}
inline Player livePlayer(const J& live) {
    const auto& v=live.at("player");Player p{};p.cc=CharacterClass::IRONCLAD;
    if(live.at("character")!="IRONCLAD"||spire_search::key(v.at("stance"))!="neutral"||v.at("orb_count")!=0)
        throw Failure("live_character_scope");
    p.curHp=checked<int>(v.at("hp"));p.maxHp=checked<int>(v.at("max_hp"));p.gold=checked<int16_t>(v.at("gold"));
    p.block=checked<int>(v.at("block"));p.energy=checked<int>(v.at("energy"));
    p.energyPerTurn=checked<int8_t>(v.at("energy_per_turn"));p.cardDrawPerTurn=checked<int8_t>(v.at("draw_per_turn"));
    p.timesDamagedThisCombat=checked<int16_t>(v.at("times_damaged"));
    p.cardsDiscardedThisTurn=checked<int16_t>(v.at("discarded_this_turn"));
    if(!v.at("played_types").is_array()||v.at("played_types").size()>32767)throw Failure("live_play_count");
    p.cardsPlayedThisTurn=int(v.at("played_types").size());
    for(const auto& t:v.at("played_types")) {
        if(t=="ATTACK")++p.attacksPlayedThisTurn;
        else if(t=="SKILL")++p.skillsPlayedThisTurn;
        else if(t!="POWER"&&t!="STATUS"&&t!="CURSE")throw Failure("live_play_type");
    }
    for(const auto& r:live.at("relics")) {
        const auto relic=liveRelicId(r.at("id"));int id=int(relic);
        if(id<64)p.relicBits0|=1ULL<<id;else if(id<128)p.relicBits1|=1ULL<<(id-64);
        if(relic==RelicId::RUNIC_DOME||relic==RelicId::PRISMATIC_SHARD)throw Failure("live_relic_scope");
        checked<int>(r.at("counter"));
        switch(relic) {
#define COUNTER(id,field) case RelicId::id:p.field=checked<decltype(p.field)>(r.at("counter"));break;
            COUNTER(HAPPY_FLOWER,happyFlowerCounter) COUNTER(INCENSE_BURNER,incenseBurnerCounter)
            COUNTER(INK_BOTTLE,inkBottleCounter) COUNTER(NUNCHAKU,nunchakuCounter)
            COUNTER(PEN_NIB,penNibCounter) COUNTER(SUNDIAL,sundialCounter)
#undef COUNTER
            case RelicId::NECRONOMICON:p.haveUsedNecronomiconThisTurn=!checked<bool>(r.at("available"));break;
            case RelicId::CENTENNIAL_PUZZLE:p.setHasRelic<RelicId::CENTENNIAL_PUZZLE>(checked<bool>(r.at("available")));break;
            case RelicId::LIZARD_TAIL:p.setHasRelic<RelicId::LIZARD_TAIL>(r.at("counter")!=-2);break;
            case RelicId::OMAMORI:p.setHasRelic<RelicId::OMAMORI>(checked<int>(r.at("counter"))>0);break;
            case RelicId::ORANGE_PELLETS:
                if(r.at("played_types").size()!=3)throw Failure("live_pellets_state");
                for(int j=0;j<3;++j)p.orangePelletsCardTypesPlayed[j]=checked<bool>(r["played_types"][j]);break;
            default:break;
        }
    }
    for(const auto& power:v.at("powers")) {
        const auto key=powerKey(power.at("id"));int index=0;
        for(int i=1;i<=int(PS::THE_BOMB);++i)if(spire_search::key(playerStatusEnumStrings[i])==key){index=i;break;}
        if(!index)throw Failure("unmapped_live_player_power");
        auto type=static_cast<PS>(index);const int n=checked<int>(power.at("amount"));
        // Off-class engine state must not be quietly initialized from defaults.
        if(type==PS::ECHO_FORM||type==PS::DEVA||type==PS::SURROUNDED||type==PS::FOCUS)
            throw Failure("live_power_auxiliary_state");
        if(type==PS::ARTIFACT){p.artifact=n;continue;}
        if(type==PS::STRENGTH){p.strength=n;continue;}
        if(type==PS::DEXTERITY){p.dexterity=n;continue;}
        if(type==PS::THE_BOMB){p.powerInstances.add(type,n,checked<int>(power.at("damage")));continue;}
        if(index<=int(PS::WEAK)&&type!=PS::INTANGIBLE) {
            if(checked<bool>(power.at("just_applied")))p.justAppliedBits|=1U<<index;
        }
        p.statusMap[type]=type==PS::PANACHE?checked<int>(power.at("damage")):n;
        if(type==PS::PANACHE)p.panacheCounter=checked<int8_t>(power.at("amount"));
        if(type==PS::COMBUST)p.combustHpLoss=checked<int>(power.at("hp_loss"));
        if(index<64)p.statusBits0|=1ULL<<index;else p.statusBits1|=1U<<(index-64);
        p.powerInstances.activate(type);
    }
    // Native uses -1 while the visible ninth-attack Pen Nib power is armed.
    if(p.hasRelic<RelicId::PEN_NIB>()&&p.penNibCounter==9)p.penNibCounter=-1;
    return p;
}
inline void liveMonsterPowers(Monster& m,const J& powers) {
    J v={{"id",int(m.id)},{"generation",m.publicGeneration},{"hp",m.curHp},{"max_hp",m.maxHp},
        {"block",m.block},{"gone",m.isEscapingB},{"half_dead",m.halfDead},{"intent",int(MMID::INVALID)},
        {"damage",0},{"hits",0},{"powers",J::array()},{"just_applied",{false,false,false}}};
    for(const auto& p:powers) {
        auto key=powerKey(p.at("id"));int id=-1;
        for(int i=0;i<int(MS::INVALID);++i)if(spire_search::key(monsterStatusEnumStrings[i])==key){id=i;break;}
        if(id<0)throw Failure("unmapped_live_monster_power");
        if(id==int(MS::STASIS))throw Failure("live_stasis_card_missing");
        int amount=isBooleanPower(static_cast<MS>(id))?1:checked<int>(p.at("amount"));
        v["powers"].push_back({id,amount});
        if(id==int(MS::VULNERABLE))v["just_applied"][0]=checked<bool>(p.at("just_applied"));
        if(id==int(MS::WEAK))v["just_applied"][1]=checked<bool>(p.at("just_applied"));
        if(id==int(MS::RITUAL))v["just_applied"][2]=checked<bool>(p.at("skip_first"));
    }
    const auto idx=m.idx;monster(m,v,idx);
}

class LiveCombat {
    Enemies enemies_;
    J last_,native_,relics_;
    int lastTarget_=1;
public:
    void cardUse(const CardInstance& card,int target) {
        if(card.requiresTarget()) {
            if(target<0||target>=5)throw Failure("live_observed_target_range");
            lastTarget_=target;
        }
    }
    template<class ProfileFn>
    void observe(const J& boundary,ProfileFn profile) {
        if(boundary==last_)return;
        const auto& live=boundary.at("live_combat");
        if(live.at("schema")!="public-live-combat-v1"||live.at("resolved")!=true||boundary.at("input")!="PLAYER_NORMAL")
            throw Failure("live_root_requires_resolved_boundary");
        BattleContext b;b.player=livePlayer(live);b.player.lastTargetedMonster=lastTarget_;
        b.ascension=checked<int>(live.at("ascension"));
        b.floorNum=checked<int>(live.at("floor"));b.turn=checked<int>(live.at("turn"));
        b.inputState=InputState::PLAYER_NORMAL;b.outcome=Outcome::UNDECIDED;
        bool found=false;
        for(int i=1;i<=int(ME::MYSTERIOUS_SPHERE_EVENT);++i)
            if(spire_search::key(monsterEncounterStrings[i])==spire_search::key(live.at("encounter")))
                {b.encounter=static_cast<ME>(i);found=true;break;}
        if(!found)throw Failure("unmapped_live_encounter");
        if(!enemies_.initialized&&b.turn!=1)throw Failure("missing_live_enemy_opening_history");
        const auto& monsters=live.at("monsters");if(monsters.empty()||monsters.size()>5)throw Failure("live_monster_capacity");
        b.monsters.monsterCount=int(monsters.size());b.monsters.monstersAlive=0;
        for(int i=0;i<int(monsters.size());++i) {
            const auto& v=monsters[i];auto& m=b.monsters.arr[i];m=Monster{};m.idx=i;
            m.id=liveMonsterId(v.at("id"));m.publicGeneration=checked<int>(v.at("generation"));
            m.curHp=checked<int>(v.at("hp"));m.maxHp=checked<int>(v.at("max_hp"));m.block=checked<int>(v.at("block"));
            m.halfDead=checked<bool>(v.at("half_dead"));m.isEscapingB=checked<bool>(v.at("gone"))&&m.curHp>0&&!m.halfDead;
            liveMonsterPowers(m,v.at("powers"));
            if(!m.isDeadOrEscaped())++b.monsters.monstersAlive;
            const auto* recipe=public_sampling::findMonsterBehaviorRecipe(m.id);
            if(!recipe)throw Failure("unmapped_live_enemy_recipe");
            const bool attacking=v.at("intent").get<std::string>().rfind("ATTACK",0)==0;
            std::vector<std::pair<MMID,int>> matches;
            for(auto move:recipe->moves) {
                if(move==MMID::INVALID)continue; // Old recipes contain padding.
                if(isMoveAttack(move)!=attacking)continue;
                auto test=m;test.moveHistory[0]=move;
                // Variable base damage cannot be read from getIntentBaseDmg:
                // rounding/intangible can hide it. Keep every compatible value.
                bool louse=move==MMID::RED_LOUSE_BITE||move==MMID::GREEN_LOUSE_BITE;
                int low=louse?(b.ascension>=2?6:5):0,high=louse?(b.ascension>=2?8:7):0;
                for(int aux=low;aux<=high;++aux) {
                    test.miscInfo=aux;auto damage=test.getMoveBaseDamage(b);
                    if(!attacking || (damage.attackCount==checked<int>(v.at("intent_hits"))&&
                       test.calculateDamageToPlayer(b,damage.damage)==checked<int>(v.at("intent_damage"))))
                        matches.push_back({move,aux});
                }
            }
            if(m.isDeadOrEscaped()&&enemies_.initialized&&i<enemies_.last.count&&enemies_.last.slots[i].id==m.id&&
               enemies_.last.slots[i].generation==m.publicGeneration) {
                m.moveHistory[0]=enemies_.last.slots[i].intent;
            } else {
                if(matches.size()!=1)throw std::runtime_error("ambiguous_live_intent:"+std::string(monsterIdStrings[int(m.id)])+":"+std::to_string(matches.size()));
                m.moveHistory[0]=matches[0].first;m.miscInfo=matches[0].second;
            }
        }
        int uid=0;
        auto make=[&](const J& c){return card(J(profile(c)),uid++);};
        if(boundary.at("hand").size()>10)throw Failure("live_hand_capacity");
        for(const auto& c:boundary.at("hand"))b.cards.moveToHand(make(c));
        for(const auto& c:boundary.at("draw_inventory"))b.cards.drawPile.push_back(make(c));
        for(const auto& c:boundary.at("discard"))b.cards.discardPile.push_back(make(c));
        for(const auto& c:boundary.at("exhaust"))b.cards.exhaustPile.push_back(make(c));
        b.potions.fill(Potion::EMPTY_POTION_SLOT);b.potionCapacity=int(live.at("potions").size());b.potionCount=0;
        if(b.potionCapacity>5)throw Failure("live_potion_capacity");
        for(int i=0;i<b.potionCapacity;++i) {
            auto name=spire_search::key(live["potions"][i]);bool mapped=false;
            if(name=="potionslot")name="emptypotionslot";
            for(int j=1;j<=int(Potion::WEAK_POTION);++j)if(spire_search::key(potionIds[j])==name||spire_search::key(potionEnumNames[j])==name)
                {b.potions[i]=static_cast<Potion>(j);mapped=true;break;}
            if(!mapped)throw Failure("unmapped_live_potion");
            if(b.potions[i]!=Potion::EMPTY_POTION_SLOT)++b.potionCount;
        }
        J relics=J::array();for(const auto& r:live.at("relics"))relics.push_back(spire_search::relicKey(liveRelicId(r.at("id"))));
        if(b.player.hasRelic<RelicId::FROZEN_EYE>())throw Failure("live_frozen_eye_root_contract_pending");
        // Commit only after the complete import validates; no partial enemy ledger.
        auto state=snapshot(b);auto checkedRoot=resolved(state);auto next=enemies_;next.observe(*checkedRoot);
        enemies_=std::move(next);native_=std::move(state);relics_=std::move(relics);last_=boundary;
    }
    const J& state() const {return native_;}
    const J& relics() const {return relics_;}
    const Enemies& enemies() const {return enemies_;}
};
} // namespace spire_session
