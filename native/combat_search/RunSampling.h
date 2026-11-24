#pragma once
#include "combat_search/PublicSessionState.h"

namespace spire_run_sampling {
inline constexpr auto version="public-run-sampling-v1";
// Request-local, PUBLIC-only resolved blueprint. It contains no pending queue
// and no source RNG. Every proposal still reconstructs its enemy/RNG state,
// samples its order, and passes the full projection + continuation checks.
struct Blueprint {
    std::unique_ptr<sts::BattleContext> value;
    std::unique_ptr<sts::BattleContext> make(const nlohmann::json& state, spire_belief::Budget& work) {
        if(!value) {
            work.tick(state.dump().size()+1);
            value=spire_session::resolved(state);
            if(!sts::public_state::effectsResolved(*value))throw spire_belief::Failure("blueprint_unresolved");
        }
        work.tick(1+value->cards.drawPile.size()+value->cards.discardPile.size()+
            value->cards.exhaustPile.size()+value->cards.cardsInHand+value->monsters.monsterCount);
        return std::make_unique<sts::BattleContext>(*value);
    }
};
// A selector matching only one public label has likelihood ONE: its kth
// eligible occurrence is that label in every possible order. Sample its child
// once and deterministically delete that occurrence. Do NOT replace the result
// by a shuffled bag: removing the first duplicate can bias remaining positions.
inline spire_belief::Cards sampleOrder(spire_belief::Engine& e,spire_belief::Id root,std::mt19937_64& rng) {
    using namespace spire_belief;
    Vec<Id> selectors(&e.arena);Id cur=root;
    while(true) {
        e.budget.tick();const auto& n=e.node(cur);
        if((n.kind!=Kind::SelectErase&&n.kind!=Kind::SelectObserve)||n.order.size()!=1||n.order[0]!=n.label)break;
        selectors.push_back(cur);cur=n.children.at(0);
    }
    auto out=e.sample(cur,rng);
    for(auto i=selectors.rbegin();i!=selectors.rend();++i) {
        const auto& n=e.node(*i);unsigned rank=n.markers;bool found=false;
        for(auto pos=out.begin();pos!=out.end();++pos) {
            e.budget.tick();if(*pos!=n.label)continue;
            if(rank){--rank;continue;}
            if(n.kind==Kind::SelectErase)out.erase(pos);
            found=true;break;
        }
        if(!found)throw Failure("public_single_label_selector_contract");
    }
    return out;
}

// Cheap EXACT seed preconditioning for Warcry's first empty-deck shuffle.
// Only this reviewed straight-line action and an immediate import/draw prefix
// qualify. All other observations still undergo full joint replay. This does
// not force a future draw or use a real PRNG seed: independently proposed
// shuffle seeds retain precisely their original conditional native law.
struct ShufflePrefix {
    using J=nlohmann::json;using Profile=spire_belief::trace::Card;
    std::vector<Profile> discard,prefix;
    std::size_t attempts=0;
    ShufflePrefix(const J& anchor,const std::vector<spire_belief::PublicStep>& tail) {
        using namespace spire_belief;
        if(!anchor.at("draw").empty()||tail.empty())return;
        const auto a=J::parse(tail[0].action);
        if(a.value("kind","")!="play_card")return;
        auto index=a.at("card_index").get<std::size_t>();
        if(index>=anchor.at("hand").size()||anchor.at("hand")[index][0]!=int(sts::CardId::WARCRY))return;
        const auto& obs=tail[0].observations;if(obs.size()<2)return;
        auto first=J::parse(obs[0]);
        if(first.value("kind",-1)!=int(Mechanism::Import))return;
        for(const auto& c:anchor.at("discard"))discard.push_back(c.get<Profile>());
        auto sorted=discard;std::sort(sorted.begin(),sorted.end());
        if(first.at("cards")!=J(sorted)){discard.clear();return;}
        for(std::size_t i=1;i<obs.size();++i) {
            auto event=J::parse(obs[i]);
            if(event.value("kind",-1)!=int(Mechanism::Draw))break;
            if(event.at("confused")||event.at("corruption")||event.at("cards").size()!=1)break;
            prefix.push_back(event.at("cards")[0].get<Profile>());
        }
        if(prefix.size()>discard.size())throw Failure("shuffle_prefix_inventory");
    }
    std::uint64_t sample(std::uint64_t seed,spire_belief::Budget& work) {
        if(prefix.empty())return seed;
        using namespace spire_belief;
        std::mt19937_64 proposals(seed);
        // The parent seed is itself an independent uniform proposal. Keep it
        // when matching; following attempts are fresh namespace-local proposals.
        while(true) {
            work.tick(1+discard.size()+prefix.size());
            if(work.randomWords>=work.limits.randomWords)throw Failure("random_limit");
            ++work.randomWords;++attempts;
            auto order=discard;sts::Random rng(seed);
            java::Collections::shuffle(order.begin(),order.end(),java::Random(rng.randomLong()));
            bool match=true;
            for(std::size_t i=0;i<prefix.size();++i)match&=order[order.size()-1-i]==prefix[i];
            if(match)return seed;
            seed=proposals();
        }
    }
};
// Diagnostic only; reachable operator graph, never a sampled hidden order.
inline nlohmann::json graph(spire_belief::trace::MechanismReplay& order) {
    using namespace spire_belief;
    auto& b=order.belief; b.engine.budget.begin();
    auto root=b.publicRoot(); std::set<Id> seen; std::vector<Id> todo{root};
    nlohmann::json rows=nlohmann::json::array();
    while(!todo.empty()) {
        auto id=todo.back();todo.pop_back();if(!seen.insert(id).second)continue;
        const auto& n=b.engine.node(id);
        rows.push_back({{"id",id},{"kind",int(n.kind)},{"cards",n.cards},{"order",n.order},
            {"children",n.children},{"label",n.label},{"rank",n.markers}});
        for(auto c:n.children)todo.push_back(c);
        for(const auto& t:n.terms)todo.push_back(t.child);
    }
    return {{"root",root},{"nodes",rows},{"allocated",b.engine.allocatedNodes()},
        {"cache_entries",b.engine.cacheEntries()},{"arena_bytes",b.engine.arena.current}};
}
}
