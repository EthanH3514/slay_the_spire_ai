#pragma once
#include "Engine.h"
#include "PublicNumericJson.h"
#include "../draw_belief/SimulatorTrace.h"

namespace spire_session {
using namespace sts;
using spire_search::J;
using Profile = spire_belief::trace::Card;
inline constexpr auto stateVersion = "public-session-state-v1";

// Explicit public/observed-history fields. Never serialize an entire native
// object: RNG, cached enemy phases, private card identities and queues are NOT
// reconstructible observations. The transport is adapter data, not ML features.
#define SPIRE_PLAYER_SCALARS(X) \
    X(gold) X(curHp) X(maxHp) X(energy) X(energyPerTurn) X(cardDrawPerTurn) \
    X(block) X(artifact) X(dexterity) X(focus) X(strength) X(lastTargetedMonster) \
    X(happyFlowerCounter) X(incenseBurnerCounter) X(inkBottleCounter) \
    X(inserterCounter) X(nunchakuCounter) X(penNibCounter) X(sundialCounter) \
    X(haveUsedNecronomiconThisTurn) X(combustHpLoss) X(devaFormEnergyPerTurn) \
    X(echoFormCardsDoubled) X(panacheCounter) X(cardsPlayedThisTurn) \
    X(attacksPlayedThisTurn) X(skillsPlayedThisTurn) X(cardsDiscardedThisTurn) \
    X(lastAttackUnblockedDamage) X(timesDamagedThisCombat)

template<class T> inline T checked(const J& input) {
    if constexpr (std::is_same_v<T,bool>) {
        if(!input.is_boolean())throw std::runtime_error("public_boolean_contract");
        return input.get<bool>();
    } else {
        if(!input.is_number_integer())throw std::runtime_error("public_integer_contract");
        auto n=input.get<std::int64_t>();
        if(n<std::numeric_limits<T>::min()||n>std::numeric_limits<T>::max())
            throw std::runtime_error("public_integer_range");
        return static_cast<T>(n);
    }
}
inline J card(const CardInstance& c) {return spire_belief::trace::project(c);}
inline CardInstance card(const J& value,int identity) {
    auto p=value.get<Profile>();
    (void)spire_belief::trace::mappedProfile(p,spire_belief::trace::AttributeRule::ResetTurnCosts);
    CardInstance c(static_cast<CardId>(p[0]));
    c.upgraded=p[1]!=0;c.cost=p[2];c.costForTurn=p[3];c.freeToPlayOnce=p[4]!=0;
    if(c.id==CardId::SEARING_BLOW)c.specialData=p[1];
    if(c.id==CardId::RAMPAGE)c.specialData=p[5]-8;
    if(c.id==CardId::RITUAL_DAGGER)c.specialData=p[5];
    c.uniqueId=identity;return c;
}
template<class Pile> inline J pile(const Pile& cards,bool unordered=false) {
    J out=J::array();for(const auto& c:cards)out.push_back(card(c));
    if(unordered)std::sort(out.begin(),out.end());return out;
}
inline J player(const Player& p) {
    if(!public_state::powersComplete(p))throw std::runtime_error("incomplete_public_powers");
    J out=J::object();
#define FIELD(name) out[#name]=p.name;
    SPIRE_PLAYER_SCALARS(FIELD)
#undef FIELD
    out["stance"]=int(p.stance);out["pellets"]=p.orangePelletsCardTypesPlayed.to_ulong();
    out["just_applied"]=p.justAppliedBits;
    out["active_relics"]=J::array();
    // Player stores only combat relics (the first 128 enum values).
    for(int i=0;i<128;++i)if(p.hasRelicRuntime(static_cast<RelicId>(i)))out["active_relics"].push_back(i);
    out["powers"]=J::array();
    for(const auto& power:public_state::playerPowers(p)) {
        J row=J::object();row["type"]=int(power.type);row["amount"]=power.amount;
        row["damage"]=power.damage;row["hp_loss"]=power.hpLoss;
        out["powers"].push_back(std::move(row));
    }
    return out;
}
inline std::string encodePlayer(const Player& p) {
    if(!public_state::powersComplete(p))throw std::runtime_error("incomplete_public_powers");
    // Sort only the immutable schema once, never retain state between calls.
#define FIELD(name) std::string_view(#name),
    static constexpr std::array names={SPIRE_PLAYER_SCALARS(FIELD)
        std::string_view("stance"),std::string_view("pellets"),std::string_view("just_applied"),
        std::string_view("active_relics"),std::string_view("powers")};
#undef FIELD
    static const auto order=[] {
        std::array<std::size_t,names.size()> indices{};
        for(std::size_t i=0;i<indices.size();++i)indices[i]=i;
        std::sort(indices.begin(),indices.end(),[](auto a,auto b){return names[a]<names[b];});
        return indices;
    }();
#define FIELD(name) PublicScalar(p.name),
    const std::array values={SPIRE_PLAYER_SCALARS(FIELD)
        PublicScalar(int(p.stance)),PublicScalar(p.orangePelletsCardTypesPlayed.to_ulong()),
        PublicScalar(p.justAppliedBits),PublicScalar(0),PublicScalar(0)};
#undef FIELD
    static_assert(names.size()==values.size(),"public player field alignment");
    PublicNumericJson out;out.raw("{");bool first=true;
    for(auto i:order){
        if(!first)out.raw(",");first=false;out.key(names[i]);
        if(i==names.size()-2){
            out.raw("[");bool relicFirst=true;
            for(int r=0;r<128;++r)if(p.hasRelicRuntime(static_cast<RelicId>(r))){
                if(!relicFirst)out.raw(",");relicFirst=false;out.value(r);
            }
            out.raw("]");
        }else if(i==names.size()-1){
            out.raw("[");bool powerFirst=true;
            for(const auto& power:public_state::playerPowers(p)){
                if(!powerFirst)out.raw(",");powerFirst=false;
                out.raw("{");out.field("amount",power.amount);out.raw(",");out.field("damage",power.damage);
                out.raw(",");out.field("hp_loss",power.hpLoss);out.raw(",");out.field("type",int(power.type));out.raw("}");
            }
            out.raw("]");
        }else out.scalar(values[i]);
    }
    out.raw("}");return std::move(out).finish();
}
inline void player(Player& p,const J& value) {
    p=Player{};p.cc=CharacterClass::IRONCLAD;
#define FIELD(name) p.name=checked<decltype(p.name)>(value.at(#name));
    SPIRE_PLAYER_SCALARS(FIELD)
#undef FIELD
    if(value.at("stance")!=int(Stance::NEUTRAL))throw std::runtime_error("off_class_public_stance");
    p.stance=Stance::NEUTRAL;
    p.orangePelletsCardTypesPlayed=checked<unsigned>(value.at("pellets"));
    p.justAppliedBits=checked<std::uint32_t>(value.at("just_applied"));
    for(const auto& item:value.at("active_relics")) {
        int i=checked<int>(item);if(i<0||i>=128)throw std::runtime_error("public_relic_range");
        if(i<64)p.relicBits0|=1ULL<<i;else p.relicBits1|=1ULL<<(i-64);
    }
    for(const auto& entry:value.at("powers")) {
        int i=checked<int>(entry.at("type"));if(i<=0||i>int(PS::THE_BOMB))throw std::runtime_error("public_power_range");
        auto type=static_cast<PS>(i);int amount=checked<int>(entry.at("amount"));
        if(public_state::directPower(type))continue; // authoritative scalar above
        if(type==PS::THE_BOMB){p.powerInstances.add(type,amount,checked<int>(entry.at("damage")));continue;}
        p.statusMap[type]=type==PS::PANACHE?checked<int>(entry.at("damage")):amount;
        if(i<64)p.statusBits0|=1ULL<<i;else p.statusBits1|=1U<<(i-64);
        p.powerInstances.activate(type);
    }
    if(player(p)!=value)throw std::runtime_error("public_player_roundtrip");
}

inline J monster(const Monster& m,const BattleContext& b) {
    J powers=J::array();
    for(int i=0;i<int(MS::INVALID);++i) {
        auto type=static_cast<MS>(i);auto amount=m.getStatusInternal(type);
        if(amount)powers.push_back({i,amount});
    }
    J out=J::object();out["id"]=int(m.id);out["generation"]=m.publicGeneration;
    out["hp"]=m.curHp;out["max_hp"]=m.maxHp;out["block"]=m.block;
    out["gone"]=m.isDeadOrEscaped();out["half_dead"]=m.halfDead;out["intent"]=int(m.moveHistory[0]);
    auto damage=m.getMoveBaseDamage(b);out["damage"]=damage.damage;out["hits"]=damage.attackCount;
    out["powers"]=std::move(powers);
    out["just_applied"]={m.wasJustApplied<MS::VULNERABLE>(),m.wasJustApplied<MS::WEAK>(),m.wasJustApplied<MS::RITUAL>()};
    return out;
}
inline void encodeMonster(PublicNumericJson& out,const Monster& m,const BattleContext& b) {
    // Exactly the sorted keys of monster(). No miscInfo, private IDs, RNG,
    // unannounced intent history or other native object fields are serialized.
    const auto damage=m.getMoveBaseDamage(b);
    out.raw("{");out.field("block",m.block);out.raw(",");out.field("damage",damage.damage);
    out.raw(",");out.field("generation",m.publicGeneration);out.raw(",");out.field("gone",m.isDeadOrEscaped());
    out.raw(",");out.field("half_dead",m.halfDead);out.raw(",");out.field("hits",damage.attackCount);
    out.raw(",");out.field("hp",m.curHp);out.raw(",");out.field("id",int(m.id));
    out.raw(",");out.field("intent",int(m.moveHistory[0]));out.raw(",\"just_applied\":[");
    out.value(m.wasJustApplied<MS::VULNERABLE>());out.raw(",");out.value(m.wasJustApplied<MS::WEAK>());
    out.raw(",");out.value(m.wasJustApplied<MS::RITUAL>());out.raw("],");out.field("max_hp",m.maxHp);
    out.raw(",\"powers\":[");bool first=true;
    for(int i=0;i<int(MS::INVALID);++i){
        auto amount=m.getStatusInternal(static_cast<MS>(i));if(!amount)continue;
        if(!first)out.raw(",");first=false;out.raw("[");out.value(i);out.raw(",");out.value(amount);out.raw("]");
    }
    out.raw("]}");
}
inline void monster(Monster& m,const J& v,int slot) {
    const int id=checked<int>(v.at("id"));
    if(id<0||id>int(MonsterId::WRITHING_MASS))throw std::runtime_error("public_monster_range");
    m=Monster{};m.idx=slot;m.id=static_cast<MonsterId>(id);
    m.publicGeneration=checked<int>(v.at("generation"));m.curHp=checked<int>(v.at("hp"));
    m.maxHp=checked<int>(v.at("max_hp"));m.block=checked<int>(v.at("block"));
    m.halfDead=checked<bool>(v.at("half_dead"));m.isEscapingB=v.at("gone").get<bool>()&&m.curHp>0&&!m.halfDead;
    const int intent=checked<int>(v.at("intent"));
    if(intent<0||intent>=int(std::size(monsterMoveStrings)))
        throw std::runtime_error("public_intent_range");
    m.moveHistory[0]=static_cast<MMID>(intent);
    // These fields are not imported from native internals. The currently
    // displayed base damage/hit count determines them for these intents.
    const int damage=checked<int>(v.at("damage"));
    switch(m.moveHistory[0]) {
        case MMID::GREEN_LOUSE_BITE:case MMID::RED_LOUSE_BITE:case MMID::HEXAGHOST_DIVIDER:
            m.miscInfo=damage;break;
        case MMID::BOOK_OF_STABBING_MULTI_STAB:m.miscInfo=checked<int>(v.at("hits"));break;
        default:break;
    }
    for(const auto& entry:v.at("powers")) {
        int type=checked<int>(entry.at(0)),amount=checked<int>(entry.at(1));
        if(type<0||type>=int(MS::INVALID))throw std::runtime_error("public_monster_power_range");
        m.statusBits|=1ULL<<type;
        switch(static_cast<MS>(type)) {
#define POWER(id,field) case MS::id:m.field=checked<decltype(m.field)>(entry.at(1));break;
            POWER(ARTIFACT,artifact) POWER(BLOCK_RETURN,blockReturn) POWER(CHOKED,choked)
            POWER(CORPSE_EXPLOSION,corpseExplosion) POWER(LOCK_ON,lockOn) POWER(MARK,mark)
            POWER(METALLICIZE,metallicize) POWER(PLATED_ARMOR,platedArmor) POWER(POISON,poison)
            POWER(REGEN,regen) POWER(SHACKLED,shackled) POWER(STRENGTH,strength)
            POWER(VULNERABLE,vulnerable) POWER(WEAK,weak)
#undef POWER
            default:if(type<=int(MS::TIME_WARP))m.uniquePower0=amount;
                else if(type<=int(MS::SHARP_HIDE))m.uniquePower1=checked<std::int16_t>(entry.at(1));break;
        }
    }
    const auto& applied=v.at("just_applied");
    m.setJustApplied<MS::VULNERABLE>(checked<bool>(applied.at(0)));
    m.setJustApplied<MS::WEAK>(checked<bool>(applied.at(1)));
    m.setJustApplied<MS::RITUAL>(checked<bool>(applied.at(2)));
}

// Semantic snapshot: neither private duplicate identities nor unknown order.
// Queue content is deliberately absent; a paused state MUST be replayed.
inline J snapshot(const BattleContext& b) {
    spire_perf::Scope timing(spire_perf::Phase::Snapshot);
    J out=J::object();out["schema"]=stateVersion;out["ascension"]=b.ascension;out["floor"]=b.floorNum;
    out["encounter"]=int(b.encounter);out["turn"]=b.turn;out["outcome"]=int(b.outcome);
    out["input"]=int(b.inputState);out["resolved"]=public_state::effectsResolved(b);
    out["player"]=player(b.player);out["potions"]=J::array();out["monsters"]=J::array();
    out["hand"]=J::array();out["draw"]=pile(b.cards.drawPile,true);
    out["discard"]=pile(b.cards.discardPile);out["exhaust"]=pile(b.cards.exhaustPile);
    out["held"]=J::array();out["selection"]=J::object();out["knowledge"]=J::object();
    for(int i=0;i<b.potionCapacity;++i)out["potions"].push_back(int(b.potions[i]));
    for(int i=0;i<b.monsters.monsterCount;++i)out["monsters"].push_back(monster(b.monsters.arr[i],b));
    for(int i=0;i<b.cards.cardsInHand;++i)out["hand"].push_back(card(b.cards.hand[i]));
    for(int i=0;i<2;++i) {
        bool visible=false;
        for(int slot=0;slot<b.monsters.monsterCount;++slot)
            if(std::min(1,slot)==i&&b.monsters.arr[slot].id==MonsterId::BRONZE_ORB&&
               !b.monsters.arr[slot].isDeadOrEscaped()&&b.monsters.arr[slot].hasStatus<MS::STASIS>())visible=true;
        out["held"].push_back(visible?card(b.cards.stasisCards[i]):J());
    }
    const auto& cm=b.cards;
    auto publicProfiles=[&](const std::vector<int>& ids,bool uncertain,bool sorted=false) {
        J cards=J::array();if(uncertain)return cards;
        for(int id:ids) {
            auto found=std::find_if(cm.drawPile.begin(),cm.drawPile.end(),[&](const auto& c){return c.uniqueId==id;});
            if(found==cm.drawPile.end())throw std::runtime_error("stale_public_order_reference");
            cards.push_back(card(*found));
        }
        if(sorted)std::sort(cards.begin(),cards.end());return cards;
    };
    auto& knowledge=out["knowledge"];knowledge["top"]=publicProfiles(cm.publicKnownTopIds,cm.publicTopOrderUncertain);
    knowledge["bottom"]=publicProfiles(cm.publicKnownBottomIds,cm.publicBottomOrderUncertain);
    knowledge["priority"]=publicProfiles(cm.publicOpeningPriorityIds,cm.publicOpeningOrderUncertain,true);
    knowledge["top_uncertain"]=cm.publicTopOrderUncertain;knowledge["bottom_uncertain"]=cm.publicBottomOrderUncertain;
    knowledge["opening_uncertain"]=cm.publicOpeningOrderUncertain;
    if(b.inputState==InputState::CARD_SELECT) {
        auto v=spire_search::project(b,J::array(),-1,"IRONCLAD",true);
        auto& s=out["selection"];s["task"]=v["selection"];
        for(const auto* field:{"selection_any","selection_max","selection_copies","selection_pick","selection_type"})s[field]=v[field];
        s["candidates"]=v["selection_candidates"];
        for(auto& c:s["candidates"])c.erase("uid");
        s["options"]=v["selection_options"];
    }
    if(b.player.hasRelic<RelicId::FROZEN_EYE>()) {
        out["revealed_order"]=J::array();for(auto i=b.cards.drawPile.rbegin();i!=b.cards.drawPile.rend();++i)out["revealed_order"].push_back(card(*i));
    }
    return out;
}

// Byte-identical normal-state snapshot without the temporary JSON object graph.
// Keep snapshot() as the reference and complex selection-window implementation.
inline std::string encodeSnapshot(const BattleContext& b) {
    if(b.inputState==InputState::CARD_SELECT)return snapshot(b).dump();
    spire_perf::Scope timing(spire_perf::Phase::Snapshot);
    const auto playerText=encodePlayer(b.player);
    auto profiles=[](const auto& cards){std::vector<Profile> out;out.reserve(cards.size());
        for(const auto& c:cards)out.push_back(spire_belief::trace::project(c));return out;};
    // Preserve validation order as well as public equivalence-class ordering.
    auto draw=profiles(b.cards.drawPile);std::sort(draw.begin(),draw.end());
    const auto discard=profiles(b.cards.discardPile),exhaust=profiles(b.cards.exhaustPile);
    PublicNumericJson monsters;monsters.raw("[");
    for(int i=0;i<b.monsters.monsterCount;++i){if(i)monsters.raw(",");encodeMonster(monsters,b.monsters.arr[i],b);}
    monsters.raw("]");const auto monsterText=std::move(monsters).finish();
    std::vector<Profile> hand;hand.reserve(b.cards.cardsInHand);
    for(int i=0;i<b.cards.cardsInHand;++i)hand.push_back(spire_belief::trace::project(b.cards.hand[i]));
    std::array<std::optional<Profile>,2> held;
    for(int i=0;i<2;++i)for(int slot=0;slot<b.monsters.monsterCount;++slot)
        if(std::min(1,slot)==i&&b.monsters.arr[slot].id==MonsterId::BRONZE_ORB&&
           !b.monsters.arr[slot].isDeadOrEscaped()&&b.monsters.arr[slot].hasStatus<MS::STASIS>()){
            held[i]=spire_belief::trace::project(b.cards.stasisCards[i]);break;}
    const auto& cm=b.cards;
    auto known=[&](const std::vector<int>& ids,bool uncertain){std::vector<Profile> out;
        if(uncertain)return out;out.reserve(ids.size());
        for(int id:ids){auto found=std::find_if(cm.drawPile.begin(),cm.drawPile.end(),[&](const auto& c){return c.uniqueId==id;});
            if(found==cm.drawPile.end())throw std::runtime_error("stale_public_order_reference");
            out.push_back(spire_belief::trace::project(*found));}return out;};
    const auto top=known(cm.publicKnownTopIds,cm.publicTopOrderUncertain);
    const auto bottom=known(cm.publicKnownBottomIds,cm.publicBottomOrderUncertain);
    auto priority=known(cm.publicOpeningPriorityIds,cm.publicOpeningOrderUncertain);std::sort(priority.begin(),priority.end());
    PublicNumericJson out;
    auto cardText=[&](const Profile& p){out.raw("[");for(unsigned i=0;i<p.size();++i){if(i)out.raw(",");out.value(p[i]);}out.raw("]");};
    auto pileText=[&](const auto& p){out.raw("[");bool first=true;for(const auto& c:p){if(!first)out.raw(",");first=false;cardText(c);}out.raw("]");};
    out.raw("{");out.field("ascension",b.ascension);out.raw(",\"discard\":");pileText(discard);
    out.raw(",\"draw\":");pileText(draw);out.raw(",");out.field("encounter",int(b.encounter));
    out.raw(",\"exhaust\":");pileText(exhaust);out.raw(",");out.field("floor",b.floorNum);
    out.raw(",\"hand\":");pileText(hand);out.raw(",\"held\":[");
    for(int i=0;i<2;++i){if(i)out.raw(",");if(held[i])cardText(*held[i]);else out.raw("null");}
    out.raw("],");out.field("input",int(b.inputState));out.raw(",\"knowledge\":{\"bottom\":");pileText(bottom);
    out.raw(",");out.field("bottom_uncertain",cm.publicBottomOrderUncertain);
    out.raw(",");out.field("opening_uncertain",cm.publicOpeningOrderUncertain);
    out.raw(",\"priority\":");pileText(priority);out.raw(",\"top\":");pileText(top);
    out.raw(",");out.field("top_uncertain",cm.publicTopOrderUncertain);out.raw("},\"monsters\":");out.raw(monsterText);
    out.raw(",");out.field("outcome",int(b.outcome));out.raw(",\"player\":");out.raw(playerText);
    out.raw(",\"potions\":[");for(int i=0;i<b.potionCapacity;++i){if(i)out.raw(",");out.value(int(b.potions[i]));}
    out.raw("],");out.field("resolved",public_state::effectsResolved(b));
    if(b.player.hasRelic<RelicId::FROZEN_EYE>()){
        out.raw(",\"revealed_order\":[");bool first=true;
        for(auto i=cm.drawPile.rbegin();i!=cm.drawPile.rend();++i){if(!first)out.raw(",");first=false;cardText(spire_belief::trace::project(*i));}out.raw("]");}
    out.raw(",\"schema\":\"");out.raw(stateVersion);out.raw("\",\"selection\":{},");out.field("turn",b.turn);out.raw("}");
    return std::move(out).finish();
}

inline void restoreKnowledge(BattleContext& b,const J& v) {
    auto& cm=b.cards;const auto& k=v.at("knowledge");
    cm.publicKnownTopIds.clear();cm.publicKnownBottomIds.clear();cm.publicOpeningPriorityIds.clear();
    cm.publicTopOrderUncertain=checked<bool>(k.at("top_uncertain"));
    cm.publicBottomOrderUncertain=checked<bool>(k.at("bottom_uncertain"));
    cm.publicOpeningOrderUncertain=checked<bool>(k.at("opening_uncertain"));
    if(k.at("top").size()>cm.drawPile.size()||k.at("bottom").size()>cm.drawPile.size())
        throw std::runtime_error("public_order_inventory");
    for(std::size_t i=0;i<k.at("top").size();++i) {
        const auto& c=cm.drawPile[cm.drawPile.size()-k["top"].size()+i];
        if(card(c)!=k["top"][i])throw std::runtime_error("sampled_public_top_mismatch");
        cm.publicKnownTopIds.push_back(c.uniqueId);
    }
    for(std::size_t i=0;i<k.at("bottom").size();++i) {
        const auto& c=cm.drawPile[i];if(card(c)!=k["bottom"][i])throw std::runtime_error("sampled_public_bottom_mismatch");
        cm.publicKnownBottomIds.push_back(c.uniqueId);
    }
    // An intact opening band has a known count above the ordinary unknown
    // region. Once interleaved, that legacy summary is NOT evidence of origins.
    const auto& priority=k.at("priority");std::size_t end=cm.drawPile.size()-k.at("top").size();
    if(priority.size()>end)throw std::runtime_error("public_opening_inventory");
    J seen=J::array();
    for(std::size_t i=end-priority.size();i<end;++i){seen.push_back(card(cm.drawPile[i]));cm.publicOpeningPriorityIds.push_back(cm.drawPile[i].uniqueId);}
    std::sort(seen.begin(),seen.end());std::sort(cm.publicOpeningPriorityIds.begin(),cm.publicOpeningPriorityIds.end());
    if(seen!=priority)throw std::runtime_error("sampled_public_opening_mismatch");
}

inline std::unique_ptr<BattleContext> resolved(const J& v) {
    if(v.at("schema")!=stateVersion||v.at("resolved")!=true||v.at("input")!=int(InputState::PLAYER_NORMAL))
        throw std::runtime_error("unresolved_public_anchor");
    auto out=std::make_unique<BattleContext>();auto& b=*out;
    b.ascension=checked<int>(v.at("ascension"));b.floorNum=checked<int>(v.at("floor"));
    const int encounter=checked<int>(v.at("encounter"));b.turn=checked<int>(v.at("turn"));
    if(b.ascension<0||b.ascension>20||b.turn<0||encounter<1||encounter>int(MonsterEncounter::MYSTERIOUS_SPHERE_EVENT))
        throw std::runtime_error("public_encounter_range");
    b.encounter=static_cast<MonsterEncounter>(encounter);
    if(v.at("outcome")!=int(Outcome::UNDECIDED))throw std::runtime_error("terminal_public_anchor");
    b.inputState=InputState::PLAYER_NORMAL;b.outcome=Outcome::UNDECIDED;
    player(b.player,v.at("player"));
    if(v.at("monsters").size()>5||v.at("potions").size()>5||v.at("hand").size()>10)throw std::runtime_error("public_capacity");
    b.potionCapacity=int(v.at("potions").size());b.potionCount=0;b.potions.fill(Potion::EMPTY_POTION_SLOT);
    for(int i=0;i<b.potionCapacity;++i){int id=checked<int>(v["potions"][i]);
        if(id<int(Potion::EMPTY_POTION_SLOT)||id>int(Potion::WEAK_POTION))throw std::runtime_error("public_potion_range");
        b.potions[i]=static_cast<Potion>(id);if(b.potions[i]!=Potion::EMPTY_POTION_SLOT)++b.potionCount;}
    b.monsters.monsterCount=int(v["monsters"].size());b.monsters.monstersAlive=0;
    for(int i=0;i<b.monsters.monsterCount;++i){monster(b.monsters.arr[i],v["monsters"][i],i);if(!v["monsters"][i]["gone"].get<bool>())++b.monsters.monstersAlive;}
    for(int i=0;i<b.monsters.monsterCount;++i)if(b.monsters.arr[i].moveHistory[0]==MMID::DARKLING_NIP)
        b.monsters.arr[i].miscInfo=checked<int>(v["monsters"][i]["damage"]);
    int identity=0;
    auto make=[&](const J& c){if(identity>=32000)throw std::runtime_error("public_card_limit");return card(c,identity++);};
    for(const auto& c:v.at("hand"))b.cards.moveToHand(make(c));
    for(const auto& c:v.at("draw")){auto instance=make(c);b.cards.drawPile.push_back(instance);b.cards.notifyAddToDrawPile(instance);}
    for(const auto& c:v.at("discard")){auto instance=make(c);b.cards.discardPile.push_back(instance);b.cards.notifyAddToDiscardPile(instance);}
    for(const auto& c:v.at("exhaust"))b.cards.exhaustPile.push_back(make(c));
    for(int i=0;i<2;++i)if(!v.at("held").at(i).is_null())b.cards.stasisCards[i]=make(v["held"][i]);
    b.cards.strikeCount=0;
    auto strike=[&](const CardInstance& c){if(c.isStrikeCard())++b.cards.strikeCount;};
    for(int i=0;i<b.cards.cardsInHand;++i)strike(b.cards.hand[i]);
    for(const auto* p:{&b.cards.drawPile,&b.cards.discardPile})for(const auto& c:*p)strike(c);
    // Stasis, like exhaust, is removed from the combat-wide Strike count.
    b.cards.nextUniqueCardId=identity;
    return out;
}

inline J policyView(const BattleContext& b,const J& relics,int knownTop=-1) {
    auto out=spire_search::project(b,relics,knownTop,"IRONCLAD",true,true);
    // This contract ranks publicly indistinguishable copies by view position,
    // not by a native instance ID. Keep the legacy projection unchanged.
    out["opening_priority_ids"]=J::array();
    return out;
}
#undef SPIRE_PLAYER_SCALARS
} // namespace spire_session
