#pragma once
#include "Engine.h"
#include "../draw_belief/Continuation.h"
#include <functional>

namespace spire_search {

// Separate, opt-in service contract. It does not silently change the existing
// public-flat-search-v6 binding or enable the live bridge under its old ABI.
inline constexpr auto conditionalRootsVersion = "public-history-flat-roots-v1";
using RootSeeds = std::array<std::uint64_t,7>;

struct RootEvidence {
    std::string posteriorCertificate;
    std::string factoryVersion, projectorVersion, actionVersion;
    std::vector<std::string> anchorObservations;
    std::vector<spire_belief::PublicStep> history;
    J policy, candidates, relics;
    std::string character = "IRONCLAD";

    // Full structured bytes, not a digest. Caller cache hits must compare them.
    // Includes all actions and intermediate evidence, not just the final view.
    std::string certificate(const J& budget, const J& metadata,
                            const std::vector<RootSeeds>& seeds) const {
        if (posteriorCertificate.empty() || factoryVersion.empty() ||
            projectorVersion.empty() || actionVersion.empty() || anchorObservations.empty())
            throw spire_belief::Failure("incomplete_root_contract");
        J steps = J::array();
        for (const auto& step : history) steps.push_back({{"action",step.action},{"observations",step.observations}});
        // The binary graph certificate need not be UTF-8; encode bytes exactly.
        std::vector<unsigned> graph;
        for (unsigned char c : posteriorCertificate) graph.push_back(c);
        return J{{"version",conditionalRootsVersion},{"posterior",graph},
            {"factory",factoryVersion},{"projector",projectorVersion},{"actions",actionVersion},
            {"anchor",anchorObservations},{"history",steps},{"policy",policy},{"candidates",candidates},
            {"relics",relics},{"character",character},{"budget",budget},{"metadata",metadata},{"seeds",seeds}}.dump();
    }
};

class ConditionalRoots {
public:
    // Factory builds a fresh owned world from the declared public anchor law,
    // independent proposal seeds and a request budget, never a real game/RNG.
    // Observe/Advance must include all ordered public callback + settled facts.
    using Factory = std::function<SearchRoot(const RootSeeds&,spire_belief::Budget&)>;
    using Observe = std::function<std::vector<std::string>(const SearchRoot&)>;
    using Advance = std::function<std::vector<std::string>(SearchRoot&,const std::string&)>;
    using Projection = std::function<J(const BattleContext&,const J&,int,const std::string&)>;
private:
    RootEvidence evidence_;
    Factory factory_;
    Observe observe_;
    Advance advance_;
    Projection projection_;
    bool factoryValidatesAnchor_ = false;
    spire_belief::Budget work_;
    spire_belief::ReplayLimits limits_;
    std::unique_ptr<SearchRoot> first_;
    RootSeeds accepted_{};
    int prepared_ = -1;
    bool valid_ = true;
    std::string failureCode_;
    std::size_t rebuilds_ = 0, preparedRounds_ = 0;
    double seconds_ = 0;
    spire_belief::ReplayStats stats_;

    std::unique_ptr<SearchRoot> construct(const RootSeeds& seeds) {
        spire_perf::Scope timing(spire_perf::Phase::RootConstruct);
        work_.tick();
        auto root = std::make_unique<SearchRoot>(factory_(seeds,work_));
        if (!root->battle || !sts::public_state::effectsResolved(*root->battle) ||
            root->battle->inputState != InputState::PLAYER_NORMAL)
            throw spire_belief::Failure("invalid_resolved_anchor");
        // SessionHistory::build already checks its complete structured
        // snapshot against the anchor after all resampling. Avoid a
        // second serialization/comparison of the same state only on
        // the explicit reviewed factory contract. Legacy stays strict.
        if (factoryValidatesAnchor_) return root;
        const auto observed = observe_(*root);
        std::size_t bytes = 0;
        for (const auto& item : observed) {
            if (item.size() > limits_.observationBytes - bytes) throw spire_belief::Failure("continuation_limit");
            bytes += item.size();
        }
        work_.tick(bytes+1);
        if (observed != evidence_.anchorObservations) throw spire_belief::Failure("anchor_projection_mismatch");
        return root;
    }
public:
    ConditionalRoots(RootEvidence evidence, Factory factory, Observe observe, Advance advance,
                     spire_belief::Limits workLimits = {}, spire_belief::ReplayLimits replayLimits = {},
                     Projection projection = {}, bool factoryValidatesAnchor = false)
        : evidence_(std::move(evidence)), factory_(std::move(factory)), observe_(std::move(observe)),
          advance_(std::move(advance)), projection_(std::move(projection)), work_(workLimits), limits_(replayLimits) {
        factoryValidatesAnchor_ = factoryValidatesAnchor;
        if (!factory_ || !observe_ || !advance_ || !limits_.attempts || !limits_.steps || !limits_.observationBytes)
            throw spire_belief::Failure("incomplete_root_contract");
    }
    bool atomic() const { return true; }
    const std::string& failureCode() const { return failureCode_; }
    J projectView(const BattleContext& b,const J& relics,int top,const std::string& character) const {
        return projection_?projection_(b,relics,top,character):project(b,relics,top,character);
    }
    bool sameActions(J actual,J expected) const {
        std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
        return actual==expected&&std::adjacent_find(expected.begin(),expected.end())==expected.end();
    }
    const RootEvidence& evidence() const { return evidence_; }
    // Diagnostic batches only: explicitly new construction budget, reported
    // separately. Never used to reset a production search's spent work/deadline.
    std::unique_ptr<ConditionalRoots> diagnosticBatch() const {
        return std::make_unique<ConditionalRoots>(evidence_,factory_,observe_,advance_,work_.limits,limits_,projection_,factoryValidatesAnchor_);
    }
    // Explicit observed simulation capability for the opt-in history tree.
    // Legacy flatSearch continues to use its unchanged advance/project path.
    std::vector<std::string> advancePublic(SearchRoot& root,const J& action) {
        if(!valid_||!root.battle)throw spire_belief::Failure("invalid_tree_world");
        return advance_(root,action.dump());
    }
    std::string certificate(const J& budget,const J& metadata,const std::vector<RootSeeds>& seeds) const {
        spire_perf::Scope timing(spire_perf::Phase::RootCertificate);
        const auto& l=work_.limits;
        J limits={{"search",budget},{"construction",{{"bytes",l.bytes},{"nodes",l.nodes},{"cards",l.cards},
            {"work",l.work},{"bits",l.bits},{"random_words",l.randomWords},{"rejection_attempts",l.rejectionAttempts}}},
            {"replay",{{"attempts",limits_.attempts},{"steps",limits_.steps},{"observation_bytes",limits_.observationBytes}}}};
        return evidence_.certificate(limits,metadata,seeds);
    }
    void validate(const J& original, const J& candidates, const J& meta) const {
        if (!valid_ || original != evidence_.policy || candidates != evidence_.candidates)
            throw spire_belief::Failure("stale_root_contract");
        guard(original,meta); // Not the legacy independent-shuffle root guard.
    }
    void prepare(int round, const RootSeeds& seed, Clock::time_point until) {
        spire_perf::Scope timing(spire_perf::Phase::RootPrepare);
        if (!valid_ || round != prepared_+1) throw spire_belief::Failure("root_round_contract");
        auto started = Clock::now();
        try {
            first_.reset();
            // All rounds/rebuilds share a request cap; a new round is not a
            // license to reset already-spent construction/replay work.
            if(round==0)work_.begin(until);else work_.deadline=std::min(work_.deadline,until);
            work_.tick();
            // Derive proposals only from the common round namespace. Neither
            // candidate index/order nor a hidden source fingerprint is used.
            std::array<std::uint32_t,14> words{};
            for (std::size_t i=0;i<seed.size();++i) {words[2*i]=std::uint32_t(seed[i]);words[2*i+1]=std::uint32_t(seed[i]>>32);}
            std::seed_seq sequence(words.begin(),words.end()); std::mt19937_64 random(sequence);
            first_ = spire_belief::replayContinuation<SearchRoot>(evidence_.history,[&] {
                for (auto& word : accepted_) {
                    work_.tick();
                    if(work_.randomWords==work_.limits.randomWords)throw spire_belief::Failure("random_limit");
                    ++work_.randomWords;word=random();
                }
                return construct(accepted_);
            },[&](SearchRoot& world,const std::string& action) {return advance_(world,action);},work_,stats_,limits_);
            prepared_ = round; ++preparedRounds_;
            seconds_ += std::chrono::duration<double>(Clock::now()-started).count();
        } catch (const std::exception& e) {failureCode_=e.what();valid_=false;seconds_+=std::chrono::duration<double>(Clock::now()-started).count();throw;}
    }
    SearchRoot branch(int round) {
        spire_perf::Scope timing(spire_perf::Phase::RootBranch);
        if (!valid_ || round != prepared_) throw spire_belief::Failure("root_round_contract");
        if (first_) {auto root=std::move(*first_);first_.reset();return root;}
        const auto started=Clock::now();
        try {
            // Replay precisely the accepted proposal, not a new conditional
            // draw, a clone of a paused queue, or another rejection attempt.
            ++rebuilds_;
            auto root=construct(accepted_);
            for (const auto& step:evidence_.history) {
                work_.tick();++stats_.actions;
                auto actual=advance_(*root,step.action);
                std::size_t bytes=0;
                for(const auto& item:actual) {
                    if(item.size()>limits_.observationBytes-bytes)throw spire_belief::Failure("continuation_limit");
                    bytes+=item.size();
                }
                work_.tick(bytes+1);
                if(actual!=step.observations)throw spire_belief::Failure("nondeterministic_root_rebuild");
            }
            seconds_+=std::chrono::duration<double>(Clock::now()-started).count();
            return std::move(*root);
        } catch (const std::exception& e) {failureCode_=e.what();valid_=false;seconds_+=std::chrono::duration<double>(Clock::now()-started).count();throw;}
    }
    J diagnostics() const {
        return {{"contract",conditionalRootsVersion},{"prepared_rounds",preparedRounds_},
            {"proposals",stats_.proposals},{"rejected",stats_.rejected},{"replay_actions",stats_.actions},
            {"candidate_rebuilds",rebuilds_},{"construction_work",work_.work},{"random_words",work_.randomWords},
            {"construction_work_limit",work_.limits.work},{"random_word_limit",work_.limits.randomWords},
            {"root_seconds",seconds_},{"valid",valid_}};
    }
};

inline J searchConditioned(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                          const J& budget,const J& metadata,bool trace=false) {
    const auto started=Clock::now();
    const auto& input=roots.evidence();
    // This entry explicitly requires a versioned full history contract. It is
    // not selected just because a legacy guard returned unsupported.
    const auto certificate=roots.certificate(budget,metadata,seeds);
    if(certificate.size()>16*1024*1024)throw spire_belief::Failure("root_certificate_limit");
    auto result=flatSearch(roots,input.policy,input.relics,input.character,seeds,
        input.candidates,budget,metadata,trace,started,input.actionVersion=="native-public-occurrences-v1");
    result["root_sampling"]=roots.diagnostics();
    // Full certificates / proposal RNG are adapter-private, never policy data.
    return result;
}
} // namespace spire_search
