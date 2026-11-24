#pragma once
#include "ShufflePrefixProbe.h"

namespace spire_run_sampling {
// Necessary-event conditioning, followed by the unchanged COMPLETE joint replay.
// Public pre-shuffle discard ORDER is observed at the import callback, not
// guessed from the anchor hand or copied from the hidden shuffled draw pile.
//
// The new path is the first shuffle in an end-turn transition, before any card
// is played in the next turn. Earlier shuffle/remove operations are barriers.
// Warped Tongs can consume this RNG without a draw-pile event: conservatively
// leave that case to the existing full replay until its public event is modeled.
// Existing reviewed Warcry requests retain their exact predicate/proposal order.
struct TurnShufflePrefix : ShufflePrefixProbe {
    static constexpr auto version = "public-turn-shuffle-prefix-v1";
    bool eventConditioned = false;
    std::size_t setupWork = 0;
    bool charged = false;
    TurnShufflePrefix(const J& anchor, const std::vector<spire_belief::PublicStep>& tail)
        : ShufflePrefixProbe(anchor, tail) {
        using namespace spire_belief;
        if (!prefix.empty() || tail.empty()) return;
        if (J::parse(tail.front().action).value("kind", "") != "end_turn") return;
        for (const auto& relic : anchor.at("player").at("active_relics"))
            if (relic.get<int>() == int(sts::RelicId::WARPED_TONGS)) return;
        const auto& observations = tail.front().observations;
        std::size_t imported = observations.size();
        for (std::size_t i = 0; i < observations.size(); ++i) {
            setupWork += observations[i].size() + 1;
            const auto event = J::parse(observations[i]);
            const int kind = event.value("kind", -1);
            if (kind == int(Mechanism::Import)) { imported = i; break; }
            // Maps, already-observed draws and per-copy cost settlements do not
            // advance shuffleRng. Anything else fails this optional prefilter.
            if (kind != int(Mechanism::Map) && kind != int(Mechanism::Draw) &&
                kind != int(Mechanism::Modify)) return;
        }
        if (imported == observations.size()) return;
        const auto event = J::parse(observations[imported]);
        const auto side = J::parse(event.at("side").get<std::string>());
        if (!side.contains("discard_order")) return; // no fabricated old evidence
        if (side.at("turn").get<int>() != anchor.at("turn").get<int>() + 1 ||
            side.at("player").at("cardsPlayedThisTurn").get<int>() != 0) return;
        auto input = side.at("discard_order").get<std::vector<Profile>>();
        auto sorted = input; std::sort(sorted.begin(), sorted.end());
        if (J(sorted) != event.at("cards"))
            throw Failure("public_shuffle_input_inventory");
        std::vector<Profile> observed;
        for (std::size_t i = imported + 1; i < observations.size() && observed.size() < 64; ++i) {
            setupWork += observations[i].size() + 1;
            const auto draw = J::parse(observations[i]);
            if (draw.value("kind", -1) != int(Mechanism::Draw)) break;
            if (draw.at("confused") || draw.at("corruption")) break;
            if (draw.at("cards").size() != 1) throw Failure("public_shuffle_draw_contract");
            observed.push_back(draw.at("cards").at(0).get<Profile>());
        }
        if (observed.empty()) return;
        auto remaining = sorted;
        for (const auto& card : observed) {
            auto found = std::find(remaining.begin(), remaining.end(), card);
            if (found == remaining.end()) throw Failure("public_shuffle_prefix_inventory");
            remaining.erase(found);
        }
        discard = std::move(input); prefix = std::move(observed);
        eventConditioned = true;
    }
    std::uint64_t sample(std::uint64_t seed, spire_belief::Budget& work) {
        if (!charged) { work.tick(setupWork); charged = true; }
        return ShufflePrefixProbe::sample(seed, work);
    }
};
}
