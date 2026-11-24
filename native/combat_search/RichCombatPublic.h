#pragma once
// Explicit public augmentation of the old anonymous policy view. No RNG, private
// move identity, native phase/cache or card order is exported by this interface.
#include "Engine.h"
#include "RichIntentTable.h" // mechanically generated from rich_intents.py
namespace spire_rich {
using namespace sts;
using spire_search::J;
inline constexpr auto publicVersion="public-combat-entities-v2";
inline std::string intent(MMID move,int ascension,bool asleep=true) {
    if(move==MMID::DARKLING_HARDEN&&ascension>=17)return "DEFEND_BUFF";
    if(move==MMID::DECA_SQUARE_OF_PROTECTION&&ascension>=19)return "DEFEND_BUFF";
    if(move==MMID::LAGAVULIN_SLEEP&&!asleep)return "STUN";
    const auto found=visibleIntents.find(move);
    if(found==visibleIntents.end())throw std::runtime_error("unmapped_visible_intent");
    return found->second;
}
inline J project(const BattleContext& b,const J& relicOrder,const J& previous) {
    const auto& p=b.player;
    if(p.hasRelic<RelicId::RUNIC_DOME>()||p.hasRelic<RelicId::PRISMATIC_SHARD>())
        throw std::runtime_error("rich_public_relic_scope");
    J out={{"version",publicVersion},{"turn",b.turn},{"ascension",b.ascension},
        {"cards_played",p.cardsPlayedThisTurn},{"attacks_played",p.attacksPlayedThisTurn},
        {"skills_played",p.skillsPlayedThisTurn},{"discarded",p.cardsDiscardedThisTurn},
        {"monsters",J::array()},{"relics",J::array()}};
    for(int i=0;i<b.monsters.monsterCount;++i) {
        const auto& m=b.monsters.arr[i];
        out["monsters"].push_back({{"id",spire_search::key(monsterIdStrings[int(m.id)])},
            {"slot",i},{"intent",intent(m.moveHistory[0],b.ascension,m.hasStatus<MS::ASLEEP>())},
            {"previous_intent",i<int(previous.size())?previous[i]:J()}});
    }
    for(const auto& r:relicOrder) {
        const auto id=static_cast<RelicId>(r.at("id").get<int>());
        J row={{"id",spire_search::relicKey(id)},{"order",out["relics"].size()}};
        switch(id) {
#define COUNTER(id,field) case RelicId::id:row["counter"]=int(p.field);break;
            COUNTER(HAPPY_FLOWER,happyFlowerCounter) COUNTER(INCENSE_BURNER,incenseBurnerCounter)
            COUNTER(INK_BOTTLE,inkBottleCounter) COUNTER(NUNCHAKU,nunchakuCounter)
            COUNTER(SUNDIAL,sundialCounter)
#undef COUNTER
            case RelicId::PEN_NIB:row["counter"]=p.penNibCounter<0?9:int(p.penNibCounter);break;
            case RelicId::NECRONOMICON:row["available"]=!p.haveUsedNecronomiconThisTurn;break;
            case RelicId::CENTENNIAL_PUZZLE:case RelicId::LIZARD_TAIL:
                row["available"]=p.hasRelicRuntime(id);break;
            case RelicId::ORANGE_PELLETS:
                row["attack_played"]=p.orangePelletsCardTypesPlayed[0];
                row["skill_played"]=p.orangePelletsCardTypesPlayed[1];
                row["power_played"]=p.orangePelletsCardTypesPlayed[2];break;
            default:break; // Missing dynamic fields stay unknown, never fabricated zero.
        }
        out["relics"].push_back(std::move(row));
    }
    return out;
}
// Observed icons only. Same-turn repeated reads do not advance memory, and a
// same-species spawn is a new public generation. No moveHistory[1] is consulted.
struct Memory {
    int turn=-1;
    std::array<int,5> generation{{-1,-1,-1,-1,-1}},identity{{-1,-1,-1,-1,-1}};
    std::array<std::string,5> current;
    J previous=J::array({nullptr,nullptr,nullptr,nullptr,nullptr});
    void observe(const BattleContext& b) {
        for(int i=0;i<b.monsters.monsterCount;++i) {
            const auto& m=b.monsters.arr[i];
            auto icon=intent(m.moveHistory[0],b.ascension,m.hasStatus<MS::ASLEEP>());
            if(generation[i]!=m.publicGeneration||identity[i]!=int(m.id))previous[i]=nullptr;
            else if(turn!=b.turn||icon!=current[i])previous[i]=current[i];
            current[i]=icon;generation[i]=m.publicGeneration;identity[i]=int(m.id);
        }
        turn=b.turn;
    }
};
}
