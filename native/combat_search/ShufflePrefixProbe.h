#pragma once
#include "RunSampling.h"

namespace spire_run_sampling {
// Exact same native seed predicate/proposal order as ShufflePrefix. A descending
// Fisher-Yates shuffle fixes the next draw slot on each iteration. Reject as
// soon as that fixed slot disagrees; never shuffle/copy the irrelevant suffix.
// This is only a predicate: the accepted seed still undergoes FULL native replay.
struct ShufflePrefixProbe : ShufflePrefix {
    static constexpr auto version = "public-shuffle-prefix-probe-v1";
    using ShufflePrefix::ShufflePrefix;

    bool matches(std::uint64_t seed, spire_belief::Budget& work) const {
        using namespace spire_belief;
        if (prefix.size() > discard.size()) throw Failure("shuffle_prefix_inventory");
        if (prefix.empty()) return true;
        sts::Random source(seed); java::Random rng(source.randomLong());
        // Sparse surviving-slot overrides. Removed top slots are never read
        // again. Original indices preserve duplicate cards and all attributes.
        std::vector<std::pair<std::size_t,std::size_t>> changed;
        changed.reserve(prefix.size());
        auto at = [&](std::size_t index) {
            work.tick();
            for (auto i=changed.rbegin(); i!=changed.rend(); ++i) {
                work.tick(); if (i->first==index) return i->second;
            }
            return index;
        };
        for (std::size_t i=0; i<prefix.size(); ++i) {
            const auto count=discard.size()-i, top=count-1;
            work.tick();
            const auto selected=count>1 ? std::size_t(rng.nextInt(int(count))) : 0;
            const auto selectedOriginal=at(selected);
            work.tick();
            if (discard[selectedOriginal]!=prefix[i]) return false;
            if (i+1<prefix.size() && selected!=top) {
                const auto originalTop=at(top);
                work.tick(); changed.emplace_back(selected,originalTop);
            }
        }
        return true;
    }

    std::uint64_t sample(std::uint64_t seed, spire_belief::Budget& work) {
        if (prefix.empty()) return seed;
        std::mt19937_64 proposals(seed);
        while (true) {
            work.tick();
            if (work.randomWords>=work.limits.randomWords) throw spire_belief::Failure("random_limit");
            ++work.randomWords; ++attempts;
            if (matches(seed,work)) return seed;
            seed=proposals();
        }
    }
};
}
