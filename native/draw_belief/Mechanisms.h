#pragma once
#include "Order.h"
#include <functional>
#include <memory>

namespace spire_belief {

// Public handles name observed actions/visible-card occurrences, NOT game UIDs.
// Roles are latent causal identities introduced only by an explicit Track event.
using Handle = std::uint64_t;
enum class Mechanism { Expose, Alias, Draw, Top, Bottom, Insert, Import, Shuffle, Reveal,
                       RandomRemove, RankedRemove, Map, Track, Modify, Forget, Release };
struct Evidence {
    Mechanism kind;
    Cards cards, eligible;
    Vec<Handle> sources;
    Handle handle = 0;
    unsigned rank = 0;
    bool fromBottom = false;
    explicit Evidence(Mechanism k, std::pmr::memory_resource* r) : kind(k), cards(r), eligible(r), sources(r) {}
};
using Channel = std::function<Rational(Label, Label)>;
using Transform = std::function<Label(Label)>;

class MechanismBelief {
public:
    Engine engine;
private:
    struct State {
        Id root;
        Rational mass;
        Label viewPivot = 0; // Latent symbol for the pinned UI slot, not a UID.
        std::pmr::map<Handle, Handle> links;
        std::pmr::set<Handle> active;
        State(Id r, Rational p, std::pmr::memory_resource* a) : root(r), mass(p), links(a), active(a) {}
    };
    using Symbol = std::pair<Label, Handle>;
    std::pmr::map<Symbol, Label> symbols_;
    Vec<Symbol> decoded_;
    Vec<State> states_;
    Cards view_, viewEligible_;
    Handle viewId_ = 0, lastViewId_ = 0;
    std::size_t stateLimit_;
    bool valid_ = true;
    State copy(const State& s) {
        State out(s.root, s.mass, &engine.arena); out.links = s.links; out.active = s.active; out.viewPivot = s.viewPivot; return out;
    }
    void push(Vec<State>& out, State s) {
        if (!s.mass) return;
        engine.budget.tick(out.size() + s.links.size() + s.active.size() + 1);
        for (auto& old : out) if (old.root == s.root && old.links == s.links && old.active == s.active && old.viewPivot == s.viewPivot) {
            old.mass = Rational::add(old.mass, s.mass, engine.budget); return;
        }
        if (out.size() >= stateLimit_) throw Failure("belief_state_limit");
        out.push_back(std::move(s));
    }
    void bind(State& s, Handle handle, Handle role) {
        if (!handle || s.links.count(handle) || s.links.size() >= engine.budget.limits.cards) throw Failure("reference_contract");
        s.links.emplace(handle, role);
    }
    Handle role(const State& s, Handle handle) const {
        if (!handle) return 0;
        auto it = s.links.find(handle);
        if (it == s.links.end()) throw Failure("reference_contract");
        return it->second;
    }
    Handle retainedRole(const State& s, Handle handle) const {
        auto r = role(s, handle);
        for (auto active : s.active) if (role(s, active) == r) return r;
        return 0;
    }
    void uniqueRoles(const Cards& cards) {
        std::pmr::set<Handle> seen(&engine.arena);
        for (auto c : cards) {
            engine.budget.tick(); auto role = decoded_.at(c).second;
            if (role && !seen.insert(role).second) throw Failure("duplicate_reference_in_pile");
        }
    }
    Id transform(Id root, const std::function<Symbol(Symbol)>& f) {
        Vec<std::pair<Label, Label>> mapping(&engine.arena);
        for (auto label : engine.node(root).cards) {
            if (!mapping.empty() && mapping.back().first == label) continue;
            auto next = f(decoded_.at(label)); mapping.emplace_back(label, symbol(next.first, next.second));
        }
        return engine.mapped(root, mapping);
    }
    Rational publish(Vec<State> next) {
        Rational total;
        for (const auto& s : next) total = Rational::add(total, s.mass, engine.budget);
        if (!total) throw Failure("impossible_evidence");
        for (auto& s : next) s.mass = Rational::divide(s.mass, total, engine.budget);
        states_ = std::move(next); return total;
    }
public:
    explicit MechanismBelief(Limits limits = {}, std::size_t stateLimit = 1024)
        : engine(limits), symbols_(&engine.arena), decoded_(&engine.arena), states_(&engine.arena),
          view_(&engine.arena), viewEligible_(&engine.arena), stateLimit_(stateLimit) {}
    MechanismBelief(const MechanismBelief&) = delete;
    MechanismBelief& operator=(const MechanismBelief&) = delete;
    Label symbol(Label profile, Handle role = 0) {
        std::uint64_t levels = 1; for (auto n = decoded_.size(); n; n >>= 1) ++levels;
        engine.budget.tick(levels);
        auto key = Symbol(profile, role); auto found = symbols_.find(key);
        if (found != symbols_.end()) return found->second;
        if (decoded_.size() >= engine.budget.limits.cards) throw Failure("public_label_limit");
        auto id = Label(decoded_.size()); decoded_.push_back(key); symbols_.emplace(key, id); return id;
    }
    Label profile(Label symbol) const { return decoded_.at(symbol).first; }
    void anchor(const Cards& priority, const Cards& other) {
        if (!states_.empty()) throw Failure("duplicate_anchor");
        Cards top(&engine.arena), rest(&engine.arena);
        for (auto c : priority) top.push_back(symbol(c));
        for (auto c : other) rest.push_back(symbol(c));
        states_.emplace_back(engine.concat(engine.bag(top), engine.bag(rest)), Rational(1), &engine.arena);
    }
    std::size_t states() const { return states_.size(); }
    bool valid() const { return valid_; }
    std::size_t referenceCount() const { return states_.empty() ? 0 : states_.front().links.size(); }
    bool pendingRandomizedView() const { return viewId_ != 0; }
    bool hasActiveReferences() const {
        for (const auto& s : states_) if (!s.active.empty()) return true;
        return false;
    }
    // Request-local synthetic observations must not collide with the public
    // journal's existing visible handles or latent roles, even when none are
    // active. Returns a checked block; caller binds it before requesting more.
    Handle freshHandles(unsigned count) {
        if (!count) throw Failure("reference_contract");
        Handle last = 0;
        for (const auto& s : states_) for (const auto& [handle, role] : s.links) {
            engine.budget.tick(); last = std::max({last, handle, role});
        }
        for (const auto& symbol : decoded_) {
            engine.budget.tick(); last = std::max(last, symbol.second);
        }
        if (count > std::numeric_limits<Handle>::max() - last) throw Failure("reference_contract");
        return last + 1;
    }

    // An independent, compact checkpoint. Keep every positive-mass state and
    // every live causal link; no beam, epsilon pruning, or particle reservoir.
    std::unique_ptr<MechanismBelief> compact() {
        if (!valid_) throw Failure("stale_belief");
        auto out = std::make_unique<MechanismBelief>(engine.budget.limits, stateLimit_);
        out->engine.budget.deadline = engine.budget.deadline;
        Vec<Id> roots(&engine.arena); for (const auto& s : states_) roots.push_back(s.root);
        auto copied = out->engine.importRoots(engine, roots, [&](Label c) {
            const auto& p = decoded_.at(c); return out->symbol(p.first, p.second);
        });
        for (std::size_t i = 0; i < states_.size(); ++i) {
            const auto& s = states_[i]; State n(copied[i], s.mass, &out->engine.arena);
            n.links = s.links; n.active = s.active;
            n.viewPivot = viewId_ ? out->symbol(decoded_.at(s.viewPivot).first, decoded_.at(s.viewPivot).second) : 0;
            out->states_.push_back(std::move(n));
        }
        out->view_ = view_; out->viewEligible_ = viewEligible_;
        out->viewId_ = viewId_; out->lastViewId_ = lastViewId_;
        // Charge both construction and copying against this transition's
        // budget. On failure the original published object remains usable.
        engine.budget.tick(out->engine.budget.work); return out;
    }
    std::pmr::string certificate() {
        if (!valid_) throw Failure("stale_belief");
        auto integer = [](std::pmr::string& out, std::uint64_t value) {
            for (unsigned i = 0; i < 8; ++i) out.push_back(char(value >> (i * 8)));
        };
        Vec<Id> roots(&engine.arena); for (const auto& s : states_) roots.push_back(s.root);
        auto out = engine.certificate(roots, [&](std::pmr::string& key, Label c) {
            integer(key, decoded_.at(c).first); integer(key, decoded_.at(c).second);
        });
        integer(out, states_.size());
        for (const auto& s : states_) {
            auto mass = s.mass.text(engine.budget); integer(out, mass.size()); out.append(mass);
            integer(out, s.links.size()); for (const auto& p : s.links) { integer(out, p.first); integer(out, p.second); }
            integer(out, s.active.size()); for (auto h : s.active) integer(out, h);
            if (viewId_) { integer(out, decoded_.at(s.viewPivot).first); integer(out, decoded_.at(s.viewPivot).second); }
        }
        integer(out, viewId_); integer(out, lastViewId_);
        integer(out, view_.size()); for (auto c : view_) integer(out, c);
        integer(out, viewEligible_.size()); for (auto c : viewEligible_) integer(out, c);
        return out;
    }

    // Observe the public profile at a rank among a public eligibility group.
    // Used for stable-sort ties, never a private draw-pile/UID index.
    Rational observeRank(const Cards& eligible, unsigned rank, Label observed, bool fromBottom) {
        if (!valid_) throw Failure("stale_belief");
        try {
            if (states_.empty()) throw Failure("missing_anchor");
            if (viewId_) throw Failure("unresolved_selector");
            if (!std::is_sorted(eligible.begin(), eligible.end()) ||
                std::adjacent_find(eligible.begin(), eligible.end()) != eligible.end()) throw Failure("selector_contract");
            Vec<State> next(&engine.arena);
            for (const auto& s : states_) {
                Cards group(&engine.arena); unsigned count = 0;
                for (auto c : engine.node(s.root).cards) if (std::binary_search(eligible.begin(), eligible.end(), profile(c))) {
                    ++count; if (group.empty() || group.back() != c) group.push_back(c);
                }
                if (rank >= count) continue;
                for (auto c : group) if (profile(c) == observed) {
                    auto result = engine.observeSelected(s.root, group, c, fromBottom ? count - 1 - rank : rank);
                    if (!result.likelihood) continue;
                    auto branch = copy(s); branch.root = result.root;
                    branch.mass = Rational::multiply(branch.mass, result.likelihood, engine.budget);
                    push(next, std::move(branch));
                }
            }
            return publish(std::move(next));
        } catch (...) { valid_ = false; throw; }
    }
    Cards profiles() {
        Cards out(&engine.arena);
        for (const auto& s : states_) for (auto c : engine.node(s.root).cards) out.push_back(profile(c));
        std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end()); return out;
    }
    void invalidate() { valid_ = false; }
    // Original Attack/SkillFromDeckToHandAction scans the pile bottom-first and
    // inserts each candidate uniformly BELOW the existing top. Consequently the
    // last UI slot is the bottommost eligible card; earlier slots are a uniform
    // permutation of the other eligible copies. This is NOT a uniform full
    // shuffle, native bottom-rank selector, or a private UID observation.
    Rational openRandomizedView(Handle viewId, const Cards& displayed, const Cards& eligible) {
        if (!valid_) throw Failure("stale_belief");
        try {
            if (states_.empty()) throw Failure("missing_anchor");
            if (!viewId || viewId <= lastViewId_ || viewId_ || displayed.empty() ||
                displayed.size() > engine.budget.limits.cards || eligible.size() > engine.budget.limits.cards ||
                !std::is_sorted(eligible.begin(), eligible.end()) ||
                std::adjacent_find(eligible.begin(), eligible.end()) != eligible.end()) throw Failure("selector_contract");
            Cards expected(displayed, &engine.arena); std::sort(expected.begin(), expected.end());
            for (auto c : expected) if (!std::binary_search(eligible.begin(), eligible.end(), c)) throw Failure("selector_contract");
            // Exact probability of the displayed prefix after fixing its last
            // card. Count duplicate profiles, not permutations of private IDs.
            Rational presentation(1);
            Cards remaining(displayed.begin(), displayed.end() - 1, &engine.arena);
            for (std::size_t i = 0; i + 1 < displayed.size(); ++i) {
                engine.budget.tick(remaining.size() + 1);
                auto count = std::count(remaining.begin(), remaining.end(), displayed[i]);
                presentation = Rational::multiply(presentation, Rational::ratio(count, remaining.size(), engine.budget), engine.budget);
                remaining.erase(std::find(remaining.begin(), remaining.end(), displayed[i]));
            }
            Vec<State> next(&engine.arena);
            for (const auto& original : states_) {
                Cards actual(&engine.arena), matches(&engine.arena);
                for (auto c : engine.node(original.root).cards) {
                    engine.budget.tick(eligible.size() + 1);
                    if (std::binary_search(eligible.begin(), eligible.end(), profile(c))) {
                        actual.push_back(profile(c));
                        if (matches.empty() || matches.back() != c) matches.push_back(c);
                    }
                }
                std::sort(actual.begin(), actual.end()); if (actual != expected) continue;
                for (auto c : matches) if (profile(c) == displayed.back()) {
                    auto observed = engine.observeSelected(original.root, matches, c, unsigned(displayed.size() - 1));
                    if (!observed.likelihood) continue;
                    auto branch = copy(original); branch.root = observed.root;
                    branch.viewPivot = c;
                    branch.mass = Rational::multiply(branch.mass, observed.likelihood, engine.budget);
                    push(next, std::move(branch));
                }
            }
            auto z = Rational::multiply(publish(std::move(next)), presentation, engine.budget);
            view_ = displayed; viewEligible_ = eligible; viewId_ = lastViewId_ = viewId;
            return z;
        } catch (...) { valid_ = false; throw; }
    }
    Rational chooseRandomizedView(Handle viewId, unsigned publicIndex, Handle handle) {
        if (!valid_) throw Failure("stale_belief");
        try {
            if (!viewId_ || viewId != viewId_ || publicIndex >= view_.size()) throw Failure("selector_contract");
            auto observed = view_[publicIndex]; bool pivot = publicIndex + 1 == view_.size();
            auto copies = std::count(view_.begin(), view_.end(), observed) - unsigned(view_.back() == observed);
            if (!pivot && !copies) throw Failure("selector_contract");
            Vec<State> next(&engine.arena);
            for (const auto& original : states_) {
                Cards matches(&engine.arena);
                for (auto c : engine.node(original.root).cards) if (std::binary_search(viewEligible_.begin(), viewEligible_.end(), profile(c))) {
                    engine.budget.tick(viewEligible_.size() + 1);
                    if (matches.empty() || matches.back() != c) matches.push_back(c);
                }
                for (auto c : matches) if (profile(c) == observed) {
                    Rational weight; Id remainder = original.root;
                    if (pivot) {
                        if (c != original.viewPivot) continue;
                        auto removed = engine.eraseSelected(original.root, matches, c, unsigned(view_.size() - 1));
                        weight = removed.likelihood; remainder = removed.root;
                    } else {
                        const auto& inventory = engine.node(original.root).cards;
                        engine.budget.tick(inventory.size() + 1);
                        auto count = std::count(inventory.begin(), inventory.end(), c) - unsigned(c == original.viewPivot);
                        if (!count) continue;
                        weight = Rational::ratio(count, copies, engine.budget);
                        remainder = c == original.viewPivot ? engine.eraseEarlier(original.root, c) : engine.erase(original.root, c);
                    }
                    if (!weight) continue;
                    auto branch = copy(original); branch.root = remainder; branch.viewPivot = 0;
                    branch.mass = Rational::multiply(branch.mass, weight, engine.budget);
                    auto latent = decoded_.at(c).second;
                    bind(branch, handle, latent ? latent : handle); push(next, std::move(branch));
                }
            }
            auto z = publish(std::move(next));
            // A chosen visible slot is a deliberate action, not new chance
            // evidence; all source-copy alternatives must sum to one.
            if (!(z == Rational(1))) throw Failure("selector_contract");
            viewId_ = 0; view_.clear(); viewEligible_.clear(); return z;
        } catch (...) { valid_ = false; throw; }
    }
    // Publication is transactional to callers: any failed event makes this
    // instance stale. Do not expose a prefix posterior after a failed drain.
    Rational apply(const Evidence& event, Channel channel = {}, Transform change = {}) {
        if (!valid_) throw Failure("stale_belief");
        try {
            if (states_.empty()) throw Failure("missing_anchor");
            if (viewId_ && event.kind != Mechanism::Reveal) throw Failure("unresolved_selector");
            Vec<State> next(&engine.arena);
            if (event.kind == Mechanism::Reveal) {
                struct Partial {
                    State state; Cards prefix;
                    Partial(State s, Cards p) : state(std::move(s)), prefix(std::move(p)) {}
                };
                Vec<Partial> current(&engine.arena);
                for (const auto& s : states_) if (engine.node(s.root).cards.size() == event.cards.size())
                    current.emplace_back(copy(s), Cards(&engine.arena));
                for (auto seen : event.cards) {
                    Vec<Partial> expanded(&engine.arena);
                    for (const auto& part : current) {
                        Cards options(engine.node(part.state.root).cards, &engine.arena);
                        options.erase(std::unique(options.begin(), options.end()), options.end());
                        for (auto label : options) if (profile(label) == seen) {
                            auto draw = engine.pop(part.state.root, label); if (!draw.likelihood) continue;
                            auto s = copy(part.state); s.root = draw.root;
                            s.mass = Rational::multiply(s.mass, draw.likelihood, engine.budget);
                            Cards prefix(part.prefix, &engine.arena); prefix.push_back(label);
                            if (expanded.size() >= stateLimit_) throw Failure("belief_state_limit");
                            expanded.emplace_back(std::move(s), std::move(prefix));
                        }
                    }
                    current = std::move(expanded);
                }
                for (auto& part : current) { part.state.root = engine.word(part.prefix); push(next, std::move(part.state)); }
                return publish(std::move(next));
            }
            for (const auto& original : states_) {
                auto s = copy(original);
                auto single = [&] { if (event.cards.size() != 1) throw Failure("event_contract"); return event.cards.front(); };
                if (event.kind == Mechanism::Draw || event.kind == Mechanism::RandomRemove || event.kind == Mechanism::RankedRemove) {
                    auto observed = single();
                    Cards options(engine.node(s.root).cards, &engine.arena), matches(&engine.arena);
                    std::size_t eligibleCount = 0;
                    for (auto label : options) if (std::find(event.eligible.begin(), event.eligible.end(), profile(label)) != event.eligible.end()) {
                        ++eligibleCount;
                        if (matches.empty() || matches.back() != label) matches.push_back(label);
                    }
                    options.erase(std::unique(options.begin(), options.end()), options.end());
                    for (auto label : options) {
                        Rational likelihood; Id remainder = s.root;
                        if (event.kind == Mechanism::Draw) {
                            auto emission = channel ? channel(profile(label), observed) : Rational(profile(label) == observed ? 1 : 0);
                            if (!emission) continue;
                            auto draw = engine.pop(s.root, label); remainder = draw.root;
                            likelihood = Rational::multiply(draw.likelihood, emission, engine.budget);
                        } else if (profile(label) == observed && std::binary_search(matches.begin(), matches.end(), label)) {
                            if (event.kind == Mechanism::RandomRemove) {
                                const auto& cards = engine.node(s.root).cards;
                                likelihood = Rational::ratio(std::count(cards.begin(), cards.end(), label), eligibleCount, engine.budget);
                                remainder = engine.erase(s.root, label);
                            } else {
                                if (event.rank >= eligibleCount) continue;
                                auto rank = event.fromBottom ? unsigned(eligibleCount - 1 - event.rank) : event.rank;
                                auto removal = engine.eraseSelected(s.root, matches, label, rank);
                                likelihood = removal.likelihood; remainder = removal.root;
                            }
                        }
                        if (!likelihood) continue;
                        auto branch = copy(s); branch.root = remainder;
                        branch.mass = Rational::multiply(s.mass, likelihood, engine.budget);
                        auto latentRole = decoded_.at(label).second;
                        bind(branch, event.handle, latentRole ? latentRole : event.handle); push(next, std::move(branch));
                    }
                    continue;
                }
                switch (event.kind) {
                    case Mechanism::Expose: single(); bind(s, event.handle, event.handle); break;
                    case Mechanism::Alias:
                        if (event.sources.size() != 1) throw Failure("reference_contract");
                        bind(s, event.handle, role(s, event.sources[0])); break;
                    case Mechanism::Top: case Mechanism::Bottom: case Mechanism::Insert: {
                        auto label = symbol(single(), retainedRole(s, event.handle));
                        if (decoded_[label].second) for (auto c : engine.node(s.root).cards)
                            if (decoded_[c].second == decoded_[label].second) throw Failure("duplicate_reference_in_pile");
                        auto word = engine.word(Cards({label}, &engine.arena));
                        s.root = event.kind == Mechanism::Top ? engine.concat(word, s.root) :
                            event.kind == Mechanism::Bottom ? engine.concat(s.root, word) : engine.belowTop(s.root, label);
                        break;
                    }
                    case Mechanism::Import: {
                        if (event.sources.size() != event.cards.size()) throw Failure("event_contract");
                        Cards cards(&engine.arena);
                        for (std::size_t i = 0; i < event.cards.size(); ++i) cards.push_back(symbol(event.cards[i], retainedRole(s, event.sources[i])));
                        s.root = engine.concat(engine.bag(cards), s.root); uniqueRoles(engine.node(s.root).cards); break;
                    }
                    case Mechanism::Shuffle: s.root = engine.bag(engine.node(s.root).cards); break;
                    case Mechanism::Map:
                        if (!change) throw Failure("event_contract");
                        s.root = transform(s.root, [&](Symbol p) { return Symbol(change(p.first), p.second); }); break;
                    case Mechanism::Track: {
                        if (!role(s, event.handle) || !s.active.insert(event.handle).second) throw Failure("reference_contract");
                        break;
                    }
                    case Mechanism::Modify: {
                        auto target = role(s, event.handle);
                        if (!change || !target || !s.active.count(event.handle)) throw Failure("reference_contract");
                        s.root = transform(s.root, [&](Symbol p) { return Symbol(p.second == target ? change(p.first) : p.first, p.second); }); break;
                    }
                    case Mechanism::Forget: {
                        if (!s.active.erase(event.handle)) throw Failure("reference_contract");
                        std::pmr::set<Handle> retained(&engine.arena);
                        for (auto h : s.active) retained.insert(role(s, h));
                        s.root = transform(s.root, [&](Symbol p) { return Symbol(p.first, retained.count(p.second) ? p.second : 0); });
                        break;
                    }
                    case Mechanism::Release:
                        // Explicit public liveness end. A delayed modifier must
                        // Forget its reference before it can be released.
                        if (s.active.count(event.handle) || !s.links.erase(event.handle)) throw Failure("reference_contract");
                        break;
                    default: throw Failure("event_contract");
                }
                push(next, std::move(s));
            }
            return publish(std::move(next));
        } catch (...) { valid_ = false; throw; }
    }
    Rational reconcile(const Cards& publicInventory) {
        if (!valid_) throw Failure("stale_belief");
        try {
            Cards expected(publicInventory, &engine.arena); std::sort(expected.begin(), expected.end());
            Vec<State> next(&engine.arena);
            for (const auto& s : states_) {
                Cards actual(&engine.arena); for (auto c : engine.node(s.root).cards) actual.push_back(profile(c));
                std::sort(actual.begin(), actual.end()); if (actual == expected) push(next, copy(s));
            }
            return publish(std::move(next));
        } catch (...) { valid_ = false; throw; }
    }
    Rational reconcileEligible(const Cards& eligible, const Cards& displayed) {
        if (!valid_) throw Failure("stale_belief");
        try {
            if (!std::is_sorted(eligible.begin(), eligible.end()) ||
                std::adjacent_find(eligible.begin(), eligible.end()) != eligible.end()) throw Failure("selector_contract");
            Cards expected(displayed, &engine.arena); std::sort(expected.begin(), expected.end());
            Vec<State> next(&engine.arena);
            for (const auto& s : states_) {
                Cards actual(&engine.arena);
                for (auto c : engine.node(s.root).cards) if (std::binary_search(eligible.begin(), eligible.end(), profile(c)))
                    actual.push_back(profile(c));
                std::sort(actual.begin(), actual.end()); if (actual == expected) push(next, copy(s));
            }
            return publish(std::move(next));
        } catch (...) { valid_ = false; throw; }
    }
    Vec<Term> publicComponents() {
        if (!valid_) throw Failure("stale_belief");
        try {
            Vec<Term> terms(&engine.arena);
            for (const auto& s : states_) terms.push_back({transform(s.root, [](Symbol p) { return Symbol(p.first, 0); }), s.mass});
            return terms; // May have DIFFERENT inventories after coarsened draws.
        } catch (...) { valid_ = false; throw; }
    }
    Id publicRoot() {
        try { return engine.mixture(publicComponents()).root; } // Requires one reconciled public inventory.
        catch (...) { valid_ = false; throw; }
    }
};

} // namespace spire_belief
