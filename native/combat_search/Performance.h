#pragma once
// Diagnostic-only, thread-local timing. No state/seed/action is collected, and
// no policy reads these counters. Disabled unless an explicit Capture owns it.
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace spire_perf {
using Clock=std::chrono::steady_clock;
enum class Phase : unsigned { Search, RootPrepare, RootBranch, RootConstruct,
    RootCertificate, Projection, Snapshot, Observations, Side, Simulation,
    LegalActions, Rule, PotionRequest, Guard, Value, Tree, Count };
inline constexpr const char* names[]={"search","root_prepare","root_branch","root_construct",
    "root_certificate","policy_projection","public_snapshot","observation_encoding","event_side",
    "simulation_with_callbacks","legal_actions","rule_policy","potion_request_copy","scope_guard",
    "value","tree_operations"};
struct Cell {std::uint64_t calls=0,inclusiveNs=0,exclusiveNs=0;};
struct Profile {std::array<Cell,unsigned(Phase::Count)> cells{};};
class Scope;
inline thread_local Profile* current=nullptr;
inline thread_local Scope* parent=nullptr;
class Scope {
    Profile* profile_=current;
    Scope* parent_=nullptr;
    Phase phase_;
    Clock::time_point start_{};
    std::uint64_t children_=0;
public:
    explicit Scope(Phase phase):phase_(phase) {
        if(profile_){parent_=parent;parent=this;start_=Clock::now();}
    }
    ~Scope() {
        if(!profile_)return;
        const auto elapsed=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_).count());
        auto& c=profile_->cells[unsigned(phase_)];++c.calls;c.inclusiveNs+=elapsed;c.exclusiveNs+=elapsed-children_;
        if(parent_)parent_->children_+=elapsed;
        parent=parent_;
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
class Capture {
public:
    explicit Capture(Profile& profile) {
        if(current||parent)throw std::logic_error("nested_search_profile_capture");
        current=&profile;
    }
    ~Capture(){current=nullptr;}
    Capture(const Capture&)=delete;
    Capture& operator=(const Capture&)=delete;
};
} // namespace spire_perf
