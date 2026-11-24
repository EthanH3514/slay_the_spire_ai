#pragma once
#include "DrawPrefixCondition.h"
#include <map>
#include <set>

namespace spire_session {
// Exact native card-stream preconditioning for one observed three-card
// Distilled Chaos program. No source RNG, hidden queue or target is imported.
// All target tuples are retained when their FULL public replay matches. Sampling
// native seeds against that set preserves their native (not assumed uniform)
// conditional mass, including ambiguity of publicly indistinguishable targets.
struct AutoplayTargetPrefix {
    static constexpr auto version="public-autoplay-target-prefix-v1";
    using Tuple=std::array<int,3>;
    bool eligible=false,prepared=false,entropyVerified=false;
    std::string gap="not_requested",action;
    std::vector<std::string> expected;
    std::vector<int> alive;
    std::set<Tuple> acceptedTuples;
    std::map<std::uint64_t,std::uint64_t> acceptedSeeds;
    std::size_t partitions=0,representatives=0,seedAttempts=0,cacheHits=0;

    AutoplayTargetPrefix(const J& budget,const J& anchor,const std::vector<spire_belief::PublicStep>& tail) {
        if(!budget.contains("autoplay_target_prefix"))return;
        if(budget.at("autoplay_target_prefix")!=version)throw spire_belief::Failure("autoplay_target_prefix_version");
        gap="outside_reviewed_program";
        if(tail.size()!=1||anchor.at("draw").size()<3||tail[0].observations.size()!=4)return;
        const auto a=J::parse(tail[0].action);
        if(a.value("kind","")!="use_potion")return;
        const auto slot=a.at("potion_index").get<std::size_t>();
        if(slot>=anchor.at("potions").size()||anchor.at("potions").at(slot)!=int(sts::Potion::DISTILLED_CHAOS))return;
        for(const auto& relic:anchor.at("player").at("active_relics"))if(relic==int(sts::RelicId::SACRED_BARK))return;
        for(std::size_t i=0;i<3;++i) {
            const auto e=J::parse(tail[0].observations[i]);
            if(e.value("kind",-1)!=int(spire_belief::Mechanism::Draw)||e.at("cards").size()!=1||
                    e.at("confused").get<bool>()||e.at("corruption").get<bool>())return;
        }
        const auto final=J::parse(tail[0].observations.back());
        if(final.value("input",-1)!=int(sts::InputState::CARD_SELECT))return;
        action=tail[0].action;expected=tail[0].observations;eligible=true;gap="";
    }
    static bool sameRng(const sts::Random& a,const sts::Random& b) {
        return a.seed0==b.seed0&&a.seed1==b.seed1&&a.counter==b.counter;
    }
    struct Stamp {
        std::array<std::uint64_t,18> words{};
        explicit Stamp(const sts::BattleContext& b,const sts::Random* card=nullptr) {
            int i=0;for(const auto* r:{&b.aiRng,card?card:&b.cardRandomRng,&b.miscRng,&b.monsterHpRng,&b.potionRng,&b.shuffleRng}) {
                words[i++]=r->seed0;words[i++]=r->seed1;words[i++]=std::uint64_t(r->counter);
            }
        }
        bool same(const sts::BattleContext& b)const{return words==Stamp(b).words;}
    };
    Tuple targets(sts::Random& random)const {
        if(alive.empty()||alive.size()>5)throw spire_belief::Failure("autoplay_target_population");
        Tuple t{};for(auto& index:t)index=alive.at(random.random(int(alive.size())-1));return t;
    }
    Tuple targets(std::uint64_t seed)const {sts::Random random(seed);return targets(random);}

    template<class Factory,class Advance>
    void prepare(std::uint64_t seed,std::size_t conditionedDraws,spire_belief::Budget& work,Factory factory,Advance advance) {
        if(!eligible||prepared)return;prepared=true;
        if(conditionedDraws!=3){eligible=false;gap="unconditioned_draw_prefix";return;}
        auto first=factory(seed);
        for(int i=0;i<first.battle->monsters.monsterCount;++i)
            if(!first.battle->monsters.arr[i].isDeadOrEscaped())alive.push_back(i);
        if(alive.empty()||alive.size()>5||alive.size()!=std::size_t(first.battle->monsters.getAliveCount()))
            throw spire_belief::Failure("autoplay_target_population");
        partitions=alive.size()*alive.size()*alive.size();
        std::map<Tuple,std::uint64_t> representative;
        std::mt19937_64 proposals(seed^0x74e091a25d8f3bc1ULL);
        // Finite target partition, not a retained set of future worlds. The
        // 4096 representative proposals and global work/random caps are hard.
        for(std::size_t attempt=0;representative.size()<partitions&&attempt<4096;++attempt) {
            work.tick(4);if(work.randomWords==work.limits.randomWords)throw spire_belief::Failure("random_limit");
            ++work.randomWords;++representatives;auto candidate=proposals();
            representative.emplace(targets(candidate),candidate);
        }
        if(representative.size()!=partitions)throw spire_belief::Failure("autoplay_target_partition_limit");
        for(const auto& [tuple,candidate]:representative) {
            work.tick();auto world=factory(candidate);sts::Random expectedRng(candidate);
            if(!sameRng(expectedRng,world.battle->cardRandomRng))throw spire_belief::Failure("autoplay_target_seed_contract");
            if(targets(expectedRng)!=tuple)throw spire_belief::Failure("autoplay_target_partition_contract");
            Stamp expectedStamp(*world.battle,&expectedRng);
            auto observations=advance(world,action);
            std::size_t bytes=0;for(const auto& text:observations)bytes+=text.size();work.tick(bytes+1);
            // Any extra random consumer invalidates the factorization, even in
            // a rejected tuple. Original bounded replay then remains in force.
            if(!expectedStamp.same(*world.battle)) {
                eligible=false;gap="unreviewed_additional_randomness";acceptedTuples.clear();return;
            }
            if(observations==expected)acceptedTuples.insert(tuple);
        }
        if(acceptedTuples.empty())throw spire_belief::Failure("autoplay_target_condition_contradiction");
        entropyVerified=true;
    }
    std::uint64_t sample(std::uint64_t seed,spire_belief::Budget& work) {
        if(!eligible)return seed;
        if(!prepared||!entropyVerified)throw spire_belief::Failure("autoplay_target_unprepared");
        if(auto it=acceptedSeeds.find(seed);it!=acceptedSeeds.end()){++cacheHits;return it->second;}
        const auto original=seed;std::mt19937_64 proposals(seed);
        while(true) {
            work.tick(4);++seedAttempts;
            if(acceptedTuples.count(targets(seed))) {
                if(acceptedSeeds.size()>=65536)throw spire_belief::Failure("autoplay_target_cache_limit");
                acceptedSeeds.emplace(original,seed);return seed;
            }
            if(work.randomWords==work.limits.randomWords)throw spire_belief::Failure("random_limit");
            ++work.randomWords;seed=proposals();
        }
    }
    J diagnostic()const {
        return {{"version",version},{"eligible",eligible},{"prepared",prepared},{"gap",gap},
                {"target_partitions",partitions},{"accepted_target_partitions",acceptedTuples.size()},
                {"representative_seed_proposals",representatives},{"seed_attempts",seedAttempts},{"cache_hits",cacheHits},
                {"entropy_verified",entropyVerified},{"probability_model","native-card-rng-conditional-target-prefix-v1"}};
    }
};
}
