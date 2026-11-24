#pragma once
#include "Exact.h"
#include "constants/CardPools.h"
#include "game/Game.h"
#include <array>
#include <set>

namespace spire_belief::trace {
// Opt-in event law for an already observed, immediate three-card offer. The
// ordered triple is evidence, not a new draw. The next card-RNG stream is
// independent of this event; this is NOT native-PRNG seed conditioning.
inline constexpr auto eventOfferVersion = "public-event-offer-v1";
inline constexpr std::size_t eventReplayAttempts = 64;
struct EventOffer {
    sts::CardType type;
    std::array<sts::CardId,3> cards;
    unsigned poolSize = 0;
    EventOffer(sts::CardType t,const std::array<sts::CardId,3>& observed,bool mixed=false):type(t),cards(observed) {
        using namespace sts;
        if(type!=CardType::ATTACK&&type!=CardType::SKILL&&type!=CardType::POWER&&type!=CardType::STATUS&&
           !(mixed&&type==CardType::INVALID))
            throw Failure("event_offer_pool_contract");
        poolSize=type==CardType::STATUS?CombatColorlessCardPool::getPoolSize():
            type==CardType::INVALID?CombatCardPool::getPoolSize(CharacterClass::IRONCLAD):
            CombatTypeCardPool::getPoolSize(CharacterClass::IRONCLAD,type);
        std::set<CardId> pool;
        for(unsigned i=0;i<poolSize;++i)pool.insert(type==CardType::STATUS?CombatColorlessCardPool::getCardAt(i):
            type==CardType::INVALID?CombatCardPool::getCardAt(CharacterClass::IRONCLAD,i):
            CombatTypeCardPool::getCardAt(CharacterClass::IRONCLAD,type,i));
        if(pool.size()!=poolSize||poolSize<3)throw Failure("event_offer_pool_contract");
        std::set<CardId> unique;
        for(auto c:cards)if(!pool.count(c)||!unique.insert(c).second)throw Failure("impossible_public_offer");
    }
    std::array<sts::CardId,3> consume(sts::CharacterClass cc,sts::CardType actual) const {
        if(cc!=sts::CharacterClass::IRONCLAD||actual!=type)throw Failure("event_offer_replay_contract");
        return cards;
    }
};
} // namespace spire_belief::trace
