#pragma once
#include "draw_belief/ObservedDrawCosts.h"
#include "draw_belief/Continuation.h"
#include "draw_belief/EventShuffle.h"
#include <functional>
#include <numeric>

namespace spire_draw_shuffle {
using namespace spire_belief;
using J=nlohmann::json;
using Profile=trace::Card;
inline constexpr auto version="public-shuffle-draw-channel-v1";
struct Seen { Profile after; bool confused,corruption; };

// Exact matching of physical copies to observed draws. Counts, not permutations,
// are memoized. A branch has remaining_copy_count * channel_mass * suffix_mass.
// Each channel uses denominator 4, common to every assignment, including draws
// unaffected by Confusion. Thus deterministic and random preimages are weighted
// correctly, and duplicate copies are never collapsed to one equiprobable label.
class Law {
    Arena arena_;
    using Key=std::pmr::vector<unsigned>;
    std::pmr::map<Key,UInt> memo_;
    std::vector<std::vector<unsigned>> channel_;
    std::vector<unsigned> quota_;
    UInt count(std::vector<unsigned>& used,std::size_t depth,Budget& work) {
        work.tick();
        if(depth==seen.size())return UInt(1);
        Key key(used.begin(),used.end(),&arena_);
        if(auto it=memo_.find(key);it!=memo_.end())return it->second;
        if(memo_.size()>=work.limits.nodes)throw Failure("draw_shuffle_state_limit");
        UInt total;
        for(std::size_t g=0;g<profiles.size();++g) {
            work.tick();if(used[g]>=quota_[g]||!channel_[depth][g])continue;
            const auto copies=counts[g]-used[g];++used[g];
            const auto suffix=count(used,depth+1,work);--used[g];
            total=UInt::add(total,UInt::multiply(UInt(copies*channel_[depth][g]),suffix,work),work);
        }
        if(memo_.size()>=work.limits.nodes)throw Failure("draw_shuffle_state_limit");
        memo_.emplace(std::move(key),total);return total;
    }
public:
    std::vector<Profile> input,profiles;
    std::vector<unsigned> counts;
    std::vector<Seen> seen;
    UInt total;
    Law(std::vector<Profile> from,std::vector<Seen> observations,
        const std::optional<std::vector<Profile>>& remaining,Budget& work)
        :arena_(work.limits.bytes),memo_(&arena_),input(std::move(from)),seen(std::move(observations)) {
        if(input.empty()||input.size()>4096||seen.empty()||seen.size()>64||seen.size()>input.size())
            throw Failure("draw_shuffle_inventory_limit");
        std::map<Profile,unsigned> groups;
        for(const auto& p:input){work.tick();++groups[p];}
        for(const auto& [p,n]:groups){profiles.push_back(p);counts.push_back(n);}
        quota_=counts;
        if(remaining) {
            if(remaining->size()+seen.size()!=input.size())throw Failure("draw_shuffle_remaining_size");
            for(const auto& p:*remaining) {
                work.tick();auto it=std::lower_bound(profiles.begin(),profiles.end(),p);
                if(it==profiles.end()||*it!=p||!quota_[it-profiles.begin()])throw Failure("draw_shuffle_remaining_profile");
                --quota_[it-profiles.begin()];
            }
        }
        for(const auto& observation:seen) {
            std::vector<unsigned> row;
            for(const auto& before:profiles) {
                unsigned matches=0;const int n=observation.confused&&before[2]>=0?4:1;
                for(int roll=0;roll<n;++roll) {
                    work.tick();if(trace::drawProfile(before,roll,observation.confused,observation.corruption)==observation.after)++matches;
                }
                row.push_back(matches*(4/n));
            }
            channel_.push_back(std::move(row));
        }
        std::vector<unsigned> used(profiles.size());total=count(used,0,work);
        if(!total)throw Failure("impossible_draw_shuffle_observation");
    }
    std::vector<UInt> weights(std::vector<unsigned>& used,std::size_t depth,Budget& work) {
        std::vector<UInt> out(profiles.size());
        for(std::size_t g=0;g<profiles.size();++g) {
            work.tick();if(used[g]>=quota_[g]||!channel_[depth][g])continue;
            const auto copies=counts[g]-used[g];++used[g];
            out[g]=UInt::multiply(UInt(copies*channel_[depth][g]),count(used,depth+1,work),work);--used[g];
        }
        return out;
    }
    bool apply(sts::CardManager& cards,std::mt19937_64& rng,Budget& work,spire_draw_costs::Program& costs) {
        std::vector<std::vector<std::size_t>> copies(profiles.size());
        for(std::size_t i=0;i<input.size();++i) {
            work.tick();copies[std::lower_bound(profiles.begin(),profiles.end(),input[i])-profiles.begin()].push_back(i);
        }
        std::vector<unsigned> used(profiles.size());std::vector<std::size_t> top,rest;
        costs.events.clear();costs.cursor=0;costs.budget=&work;
        for(std::size_t depth=0;depth<seen.size();++depth) {
            const auto masses=weights(used,depth,work);UInt sum;
            for(const auto& mass:masses)sum=UInt::add(sum,mass,work);
            auto ticket=randomBelow(sum,rng,work);std::size_t g=0;
            for(;g<masses.size();++g){if(ticket<masses[g])break;ticket=UInt::subtract(ticket,masses[g],work);}
            if(g==masses.size())throw Failure("draw_shuffle_mass_contract");
            auto& pool=copies[g];const auto k=randomBelow(UInt(pool.size()),rng,work).limb[0];
            top.push_back(pool[k]);pool[k]=pool.back();pool.pop_back();++used[g];
            costs.events.push_back({profiles[g],seen[depth].after,seen[depth].confused,seen[depth].corruption});
        }
        for(const auto& pool:copies)for(auto index:pool){work.tick();rest.push_back(index);}
        for(std::size_t n=rest.size();n>1;--n){work.tick();std::swap(rest[n-1],rest[randomBelow(UInt(n),rng,work).limb[0]]);}
        rest.insert(rest.end(),top.rbegin(),top.rend());auto original=cards.discardPile;
        for(std::size_t i=0;i<rest.size();++i){work.tick();cards.discardPile[i]=original[rest[i]];}
        return true;
    }
    std::size_t states()const{return memo_.size();}
};

// Optional first-import channel. Only pure post-import draws are conditioned;
// other callbacks remain barriers. Complete replay still checks all side effects
// and the final snapshot. Ordinary unmodified draws keep their previous path.
struct Plan {
    bool eligible=false,charged=false;
    std::size_t setupWork=0;
    std::vector<Profile> input;
    std::vector<Seen> seen;
    std::optional<std::vector<Profile>> remaining;
    std::unique_ptr<Law> law;
    Plan(const J& anchor,const std::vector<PublicStep>& tail) {
        if(tail.empty()||J::parse(tail[0].action).value("kind","")=="opening_prelude")return;
        const auto& observations=tail[0].observations;std::size_t imported=observations.size();
        for(std::size_t i=0;i<observations.size();++i) {
            setupWork+=observations[i].size()+1;auto e=J::parse(observations[i]);int k=e.value("kind",-1);
            if(k==int(Mechanism::Import)){imported=i;break;}
            if(k!=int(Mechanism::Map)&&k!=int(Mechanism::Draw)&&k!=int(Mechanism::Modify))return;
        }
        if(imported==observations.size())return;
        auto event=J::parse(observations[imported]);auto side=J::parse(event.at("side").get<std::string>());
        if(!side.contains("discard_order"))return;
        input=side.at("discard_order").get<std::vector<Profile>>();auto sorted=input;std::sort(sorted.begin(),sorted.end());
        if(J(sorted)!=event.at("cards"))throw Failure("public_shuffle_input_inventory");
        std::size_t i=imported+1;bool mapped=false;
        for(;i<observations.size()&&seen.size()<64;++i) {
            setupWork+=observations[i].size()+1;auto e=J::parse(observations[i]);
            if(e.value("kind",-1)!=int(Mechanism::Draw))break;
            if(e.at("cards").size()!=1)throw Failure("public_shuffle_draw_contract");
            seen.push_back({e.at("cards").at(0).get<Profile>(),e.at("confused"),e.at("corruption")});
            mapped|=seen.back().confused||seen.back().corruption;
        }
        if(!mapped||seen.empty())return;
        // No intervening pile operation: the public final multiset tells exactly
        // which pre-draw profiles left the shuffle, even if their costs changed.
        if(i+1==observations.size()) {
            auto final=J::parse(observations[i]);
            if(final.contains("resolved")&&final.contains("draw"))remaining=final.at("draw").get<std::vector<Profile>>();
        }
        eligible=true;
    }
    void prepare(Budget& work) {
        if(!eligible||charged)return;
        work.tick(setupWork);law=std::make_unique<Law>(input,seen,remaining,work);charged=true;
    }
};
}
