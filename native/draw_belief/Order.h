#pragma once

#include "Exact.h"
#include "RankedBagSample.h"
#include <deque>
#include <map>
#include <set>
#include <vector>
#include <functional>

namespace spire_belief {

using Label = std::uint32_t; // Public equivalence class, NEVER a simulator UID.
using Id = std::uint32_t;
template<class T> using Vec = std::pmr::vector<T>;
using Cards = Vec<Label>;
enum class Kind { Word, Bag, Concat, Shuffle, Protect, Delete, Mix, Map, Family, SelectErase, SelectObserve, DeleteEarlier };
struct Term { Id child; Rational mass; };
struct CountTerm { unsigned unopened, ghosts; Rational mass; };
struct Result { Rational likelihood; Id root; };

struct Node {
    Kind kind;
    Cards cards, order;
    Vec<Id> children;
    Vec<Term> terms;
    Vec<std::pair<Label, Label>> mapping;
    Vec<CountTerm> counts;
    Label label = 0;
    unsigned markers = 0;
    explicit Node(Kind k, std::pmr::memory_resource* resource)
        : kind(k), cards(resource), order(resource), children(resource), terms(resource), mapping(resource), counts(resource) {}
};

// Immutable handles in a bounded, interned arena. A handle belongs to this
// engine only. Worklists (including derivatives, sampling and deletion descent)
// are iterative. No sampled assignment is retained as the distribution itself.
class Engine {
public:
    Budget budget;
    Arena arena;
private:
    std::pmr::deque<Node> nodes_;
    std::pmr::map<std::pmr::string, Id> intern_;
    using Query = std::pair<Id, Label>;
    std::pmr::map<Query, Result> cache_;
    std::pmr::map<Id, Rational> selectionMass_;
    Id empty_ = 0;

    void sortCharge(std::size_t n) {
        std::uint64_t levels = 1;
        for (auto v = n; v > 1; v >>= 1) ++levels;
        budget.tick(n * levels + 1);
    }
    void checkCards(std::size_t n) {
        budget.tick(n + 1);
        if (n > budget.limits.cards) throw Failure("card_limit");
    }
    static void keyInt(std::pmr::string& key, std::uint32_t n) {
        for (unsigned i = 0; i < 4; ++i) key.push_back(char((n >> (8 * i)) & 255));
    }
    void keyUInt(std::pmr::string& key, const UInt& n) {
        budget.tick(n.used + 1); keyInt(key, n.used);
        for (unsigned i = 0; i < n.used; ++i) keyInt(key, n.limb[i]);
    }
    void keyMass(std::pmr::string& key, const Rational& mass) {
        keyUInt(key, mass.numerator); keyUInt(key, mass.denominator);
    }
    Id make(Node n) {
        checkCards(n.cards.size()); sortCharge(n.cards.size());
        std::sort(n.cards.begin(), n.cards.end());
        std::pmr::string key(&arena);
        keyInt(key, unsigned(n.kind)); keyInt(key, n.label); keyInt(key, n.markers);
        keyInt(key, unsigned(n.order.size())); for (auto c : n.order) keyInt(key, c);
        keyInt(key, unsigned(n.children.size())); for (auto c : n.children) keyInt(key, c);
        keyInt(key, unsigned(n.terms.size()));
        for (const auto& t : n.terms) { keyInt(key, t.child); keyMass(key, t.mass); }
        keyInt(key, unsigned(n.mapping.size()));
        for (const auto& p : n.mapping) { keyInt(key, p.first); keyInt(key, p.second); }
        keyInt(key, unsigned(n.counts.size()));
        for (const auto& t : n.counts) { keyInt(key, t.unopened); keyInt(key, t.ghosts); keyMass(key, t.mass); }
        // Family order is the terminal pool, not a realized assignment.
        budget.tick((key.size() + 1) * (1 + logarithm(intern_.size() + 1)));
        auto found = intern_.find(key);
        if (found != intern_.end()) return found->second;
        if (nodes_.size() >= budget.limits.nodes || nodes_.size() >= std::numeric_limits<Id>::max())
            throw Failure("node_limit");
        auto id = Id(nodes_.size());
        // Failure can leave unreachable immutable nodes, but never an invalid
        // published root or intern entry. The arena cap still counts that work.
        nodes_.push_back(std::move(n));
        intern_.emplace(std::move(key), id);
        return id;
    }
    static unsigned logarithm(std::size_t value) { unsigned n = 0; while (value >>= 1) ++n; return n; }
    std::size_t multiplicity(Id root, Label c) {
        const auto& cards = node(root).cards;
        budget.tick(cards.size() + 1);
        return std::count(cards.begin(), cards.end(), c);
    }
    Cards without(const Cards& cards, Label label) {
        Cards out(cards, &arena);
        budget.tick(cards.size() + 1);
        auto found = std::find(out.begin(), out.end(), label);
        if (found == out.end()) throw std::invalid_argument("missing public card label");
        out.erase(found); return out;
    }
    template<class Map, class Key> void addMass(Map& into, const Key& key, const Rational& mass) {
        budget.tick(1 + logarithm(into.size() + 1));
        if (!mass) return;
        auto it = into.find(key);
        if (it == into.end()) into.emplace(key, mass);
        else it->second = Rational::add(it->second, mass, budget);
    }
    Result familyPop(Id root, Label observed) {
        const auto& n = node(root);
        unsigned count = observed == n.label ? 0 : unsigned(std::count(n.order.begin(), n.order.end(), observed));
        if (observed == n.label ? !n.markers : !count) return {Rational(), empty_};
        // Keys descend by ghost count: all incoming epsilon mass arrives before
        // the state is visited. No hidden identity subset is enumerated.
        using State = std::pair<unsigned, unsigned>;
        std::pmr::map<State, Rational, std::greater<State>> pending(&arena);
        std::pmr::map<State, Rational> out(&arena);
        for (const auto& t : n.counts) addMass(pending, State(t.ghosts, t.unopened), t.mass);
        while (!pending.empty()) {
            budget.tick();
            auto it = pending.begin();
            unsigned g = it->first.first, u = it->first.second;
            Rational mass = it->second; pending.erase(it);
            ++epsilonVisits;
            std::uint64_t k = n.order.size() + g, length = n.markers + k;
            if (observed == n.label) {
                if (u) addMass(out, State(u - 1, g), Rational::multiply(mass, Rational::ratio(2ull * u, length, budget), budget));
                if (n.markers > u) addMass(out, State(u, g), Rational::multiply(mass, Rational::ratio(n.markers - u, length, budget), budget));
            } else if (k > u) {
                addMass(out, State(u, g), Rational::multiply(mass, Rational::ratio(count * (k - u), k * length, budget), budget));
            }
            if (g && k > u) addMass(pending, State(g - 1, u),
                Rational::multiply(mass, Rational::ratio(g * (k - u), k * length, budget), budget));
        }
        Cards pool(n.order, &arena);
        if (observed != n.label) pool = without(pool, observed);
        Vec<CountTerm> terms(&arena);
        for (const auto& p : out) terms.push_back({p.first.first, p.first.second, p.second});
        return family(n.label, pool, n.markers - unsigned(observed == n.label), terms);
    }
    // A structural known-head query, never an inspection of a sampled pile.
    bool knownHead(Id root, Label& head, Id& tail) {
        Vec<Id> suffixes(&arena);
        auto cur = root;
        while (node(cur).kind == Kind::Concat) {
            budget.tick(); suffixes.push_back(node(cur).children[1]); cur = node(cur).children[0];
        }
        const auto& n = node(cur);
        if (n.kind == Kind::Word && !n.order.empty()) {
            head = n.order.front(); Cards rest(n.order.begin() + 1, n.order.end(), &arena); tail = word(rest);
        } else if (n.kind == Kind::Family && n.markers && (n.order.empty() ||
            std::all_of(n.counts.begin(), n.counts.end(), [&](const CountTerm& t) { return t.unopened == n.order.size() + t.ghosts; }))) {
            head = n.label; auto popped = familyPop(cur, head);
            if (!(popped.likelihood == Rational(1))) throw std::logic_error("non-deterministic known head");
            tail = popped.root;
        } else return false;
        for (auto it = suffixes.rbegin(); it != suffixes.rend(); ++it) tail = concat(tail, *it);
        return true;
    }
    Result cached(Id root, Label label) {
        budget.tick(1 + logarithm(cache_.size() + 1));
        auto it = cache_.find({root, label});
        if (it == cache_.end()) throw std::logic_error("missing derivative dependency");
        return it->second;
    }
    void save(Query key, Result value) {
        budget.tick(1 + logarithm(cache_.size() + 1));
        cache_.emplace(key, std::move(value));
    }
    unsigned categorical(const Vec<Rational>& masses, std::mt19937_64& rng) {
        UInt denominator(1);
        for (const auto& p : masses) {
            auto common = UInt::gcd(denominator, p.denominator, budget);
            denominator = UInt::multiply(UInt::divide(denominator, common, budget).first, p.denominator, budget);
        }
        UInt draw = randomBelow(denominator, rng, budget);
        for (unsigned i = 0; i < masses.size(); ++i) {
            auto mass = UInt::multiply(masses[i].numerator, UInt::divide(denominator, masses[i].denominator, budget).first, budget);
            if (draw < mass) return i;
            draw = UInt::subtract(draw, mass, budget);
        }
        throw std::logic_error("unnormalized categorical law");
    }
    template<class T> void permute(Vec<T>& values, std::mt19937_64& rng) {
        for (std::size_t n = values.size(); n > 1; --n) {
            auto i = randomBelow(UInt(n), rng, budget).limb[0];
            std::swap(values[n - 1], values[i]); budget.tick();
        }
    }
    Cards sampleFamily(const Node& n, std::mt19937_64& rng) {
        Vec<Rational> masses(&arena);
        for (const auto& t : n.counts) masses.push_back(t.mass);
        auto term = n.counts[categorical(masses, rng)];
        // Ghost markers need a separate flag: every uint32 value is a valid
        // public label, so a magic card ID cannot stand for a removed card.
        Vec<std::pair<bool, Label>> terminals(&arena);
        for (auto c : n.order) terminals.push_back({true, c});
        for (unsigned i = 0; i < term.ghosts; ++i) terminals.push_back({false, 0});
        permute(terminals, rng);
        Vec<unsigned> schedule(&arena), offsets(&arena);
        for (unsigned i = 0; i < terminals.size(); ++i) {
            schedule.push_back(i); if (i < term.unopened) schedule.push_back(i);
        }
        for (unsigned i = 0; i < n.markers - term.unopened; ++i) schedule.push_back(unsigned(terminals.size()) + i);
        offsets.resize(terminals.size() + n.markers - term.unopened);
        permute(schedule, rng);
        Cards out(&arena); out.reserve(n.cards.size());
        for (auto i : schedule) {
            budget.tick();
            if (i >= terminals.size() || (i < term.unopened && offsets[i]++ == 0)) out.push_back(n.label);
            else if (terminals[i].first) out.push_back(terminals[i].second);
        }
        return out;
    }
public:
    std::size_t epsilonVisits = 0, maximumCountTerms = 0;
    explicit Engine(Limits limits = {}) : budget(limits), arena(limits.bytes), nodes_(&arena), intern_(&arena), cache_(&arena), selectionMass_(&arena) {
        empty_ = word(Cards(&arena));
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    const Node& node(Id id) const {
        if (id >= nodes_.size()) throw std::invalid_argument("foreign draw-belief handle");
        return nodes_[id];
    }
    Id empty() const { return empty_; }
    std::size_t allocatedNodes() const { return nodes_.size(); }
    std::size_t cacheEntries() const { return cache_.size(); }

    // Copy only the reachable immutable graph, never derivative caches or a
    // sampled order. Both arenas remain independently owned. Traversal is
    // iterative, including a many-thousand-operator history.
    Vec<Id> importRoots(const Engine& source, const Vec<Id>& roots,
                        const std::function<Label(Label)>& relabel) {
        if (&source == this) throw Failure("compaction_contract");
        std::pmr::map<Id, Id> copied(&arena);
        Vec<std::pair<Id, bool>> pending(&arena);
        for (auto root : roots) {
            pending.emplace_back(root, false);
            while (!pending.empty()) {
                budget.tick(); auto item = pending.back(); pending.pop_back();
                if (copied.count(item.first)) continue;
                const auto& old = source.node(item.first);
                if (!item.second) {
                    pending.emplace_back(item.first, true);
                    for (auto it = old.children.rbegin(); it != old.children.rend(); ++it)
                        if (!copied.count(*it)) pending.emplace_back(*it, false);
                    for (auto it = old.terms.rbegin(); it != old.terms.rend(); ++it)
                        if (!copied.count(it->child)) pending.emplace_back(it->child, false);
                    continue;
                }
                Node n(old.kind, &arena);
                budget.tick(old.cards.size() + old.order.size() + old.mapping.size() + old.counts.size() + 1);
                for (auto c : old.cards) n.cards.push_back(relabel(c));
                for (auto c : old.order) n.order.push_back(relabel(c));
                if (old.kind == Kind::Bag || old.kind == Kind::Family || old.kind == Kind::SelectErase || old.kind == Kind::SelectObserve)
                    std::sort(n.order.begin(), n.order.end());
                for (auto c : old.children) n.children.push_back(copied.at(c));
                for (const auto& t : old.terms) n.terms.push_back({copied.at(t.child), t.mass});
                for (const auto& p : old.mapping) n.mapping.emplace_back(relabel(p.first), relabel(p.second));
                std::sort(n.mapping.begin(), n.mapping.end());
                n.counts = old.counts; n.markers = old.markers;
                if (old.kind == Kind::Protect || old.kind == Kind::Delete || old.kind == Kind::DeleteEarlier ||
                    old.kind == Kind::Family || old.kind == Kind::SelectErase || old.kind == Kind::SelectObserve)
                    n.label = relabel(old.label);
                copied.emplace(item.first, make(std::move(n)));
            }
        }
        Vec<Id> result(&arena); for (auto r : roots) result.push_back(copied.at(r)); return result;
    }

    // Collision-free structural certificate (not a digest). Node numbers are
    // assigned by deterministic reachable traversal, not allocation history.
    // Public label certificates are supplied by the owning mechanism layer.
    std::pmr::string certificate(const Vec<Id>& roots,
                                 const std::function<void(std::pmr::string&, Label)>& labelKey) {
        std::pmr::string out(&arena);
        std::pmr::map<Id, Id> ids(&arena); Vec<Id> order(&arena);
        auto add = [&](Id id) {
            auto found = ids.find(id); if (found != ids.end()) return found->second;
            auto next = Id(order.size()); ids.emplace(id, next); order.push_back(id); return next;
        };
        keyInt(out, unsigned(roots.size())); for (auto r : roots) keyInt(out, add(r));
        for (std::size_t i = 0; i < order.size(); ++i) {
            const auto& n = node(order[i]); budget.tick(n.cards.size() + n.order.size() + n.mapping.size() + 1);
            keyInt(out, unsigned(n.kind)); keyInt(out, n.markers);
            if (n.kind == Kind::Protect || n.kind == Kind::Delete || n.kind == Kind::DeleteEarlier ||
                n.kind == Kind::Family || n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) labelKey(out, n.label);
            keyInt(out, unsigned(n.order.size()));
            if (n.kind == Kind::Bag || n.kind == Kind::Family || n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) {
                Vec<std::pmr::string> entries(&arena);
                for (auto c : n.order) { std::pmr::string row(&arena); labelKey(row, c); entries.push_back(std::move(row)); }
                std::sort(entries.begin(), entries.end()); for (const auto& row : entries) out.append(row);
            } else for (auto c : n.order) labelKey(out, c);
            keyInt(out, unsigned(n.children.size())); for (auto c : n.children) keyInt(out, add(c));
            keyInt(out, unsigned(n.terms.size()));
            for (const auto& t : n.terms) { keyInt(out, add(t.child)); keyMass(out, t.mass); }
            keyInt(out, unsigned(n.mapping.size()));
            // Relabeling changes integer sort order. Sort by public label bytes.
            Vec<std::pmr::string> maps(&arena);
            for (const auto& p : n.mapping) {
                std::pmr::string row(&arena); labelKey(row, p.first); labelKey(row, p.second); maps.push_back(std::move(row));
            }
            std::sort(maps.begin(), maps.end()); for (const auto& row : maps) out.append(row);
            keyInt(out, unsigned(n.counts.size()));
            for (const auto& t : n.counts) { keyInt(out, t.unopened); keyInt(out, t.ghosts); keyMass(out, t.mass); }
        }
        return out;
    }

    Id word(const Cards& cards) {
        checkCards(cards.size());
        Node n(Kind::Word, &arena); n.order = cards; n.cards = cards; return make(std::move(n));
    }
    Id bag(const Cards& cards) {
        checkCards(cards.size());
        Node n(Kind::Bag, &arena); n.order = cards; sortCharge(cards.size());
        std::sort(n.order.begin(), n.order.end()); n.cards = n.order;
        if (n.order.empty() || n.order.front() == n.order.back()) n.kind = Kind::Word;
        return make(std::move(n));
    }
    Id concat(Id left, Id right) {
        const auto& a = node(left); const auto& b = node(right);
        if (a.cards.empty()) return right;
        if (b.cards.empty()) return left;
        checkCards(a.cards.size() + b.cards.size());
        if (a.kind == Kind::Word && b.kind == Kind::Word) {
            Cards joined(a.order, &arena); joined.insert(joined.end(), b.order.begin(), b.order.end()); return word(joined);
        }
        Node n(Kind::Concat, &arena); n.children = {left, right}; n.cards = a.cards;
        n.cards.insert(n.cards.end(), b.cards.begin(), b.cards.end()); return make(std::move(n));
    }
    Result family(Label marker, const Cards& pool, unsigned markers, const Vec<CountTerm>& input) {
        checkCards(pool.size() + markers);
        if (std::find(pool.begin(), pool.end(), marker) != pool.end()) throw std::invalid_argument("family marker in terminal pool");
        using State = std::pair<unsigned, unsigned>;
        std::pmr::map<State, Rational> masses(&arena);
        for (const auto& t : input) {
            budget.tick();
            if (t.ghosts > budget.limits.cards) throw Failure("card_limit");
            if (t.unopened > markers || t.unopened > pool.size() + t.ghosts) throw std::invalid_argument("invalid counted roles");
            addMass(masses, (t.unopened == 0 || pool.empty()) ? State(0, 0) : State(t.unopened, t.ghosts), t.mass);
        }
        Rational z;
        for (const auto& t : masses) z = Rational::add(z, t.second, budget);
        if (!z) return {z, empty_};
        Node n(Kind::Family, &arena); n.label = marker; n.markers = markers; n.order = pool;
        sortCharge(pool.size()); std::sort(n.order.begin(), n.order.end());
        n.cards = n.order; n.cards.insert(n.cards.end(), markers, marker);
        for (const auto& t : masses) n.counts.push_back({t.first.first, t.first.second, Rational::divide(t.second, z, budget)});
        maximumCountTerms = std::max(maximumCountTerms, n.counts.size());
        return {z, make(std::move(n))};
    }
    Id shuffle(const Vec<Id>& input) {
        Vec<Id> parts(&arena), pending(input, &arena);
        while (!pending.empty()) {
            budget.tick(); Id id = pending.back(); pending.pop_back(); const auto& n = node(id);
            if (n.cards.empty()) continue;
            if (n.kind == Kind::Shuffle) pending.insert(pending.end(), n.children.begin(), n.children.end());
            else parts.push_back(id);
        }
        if (parts.empty()) return empty_;
        if (parts.size() == 1) return parts.front();
        // Certify common-head/two-card roles before applying counting.
        bool paired = true; Label marker = 0; Cards pool(&arena);
        for (auto id : parts) {
            const auto& n = node(id);
            if (n.kind != Kind::Word || n.order.size() != 2 || (!pool.empty() && n.order.front() != marker)) { paired = false; break; }
            marker = n.order.front(); pool.push_back(n.order.back());
        }
        if (paired && std::find(pool.begin(), pool.end(), marker) == pool.end()) {
            Vec<CountTerm> terms(&arena); terms.push_back({unsigned(pool.size()), 0, Rational(1)});
            return family(marker, pool, unsigned(pool.size()), terms).root;
        }
        Id familyId = 0; unsigned families = 0;
        for (auto id : parts) if (node(id).kind == Kind::Family) { familyId = id; ++families; }
        if (families == 1) {
            const auto& f = node(familyId);
            bool absorb = std::all_of(f.counts.begin(), f.counts.end(), [](const CountTerm& t) { return t.ghosts == 0; });
            std::size_t extra = 0;
            for (auto id : parts) if (id != familyId) {
                const auto& n = node(id); budget.tick(n.cards.size() + 1);
                if ((n.kind != Kind::Word && n.kind != Kind::Bag) ||
                    !std::all_of(n.cards.begin(), n.cards.end(), [&](Label c) { return c == f.label; })) absorb = false;
                extra += n.cards.size();
            }
            if (absorb) { checkCards(f.cards.size() + extra); return family(f.label, f.order, f.markers + unsigned(extra), f.counts).root; }
        }
        Cards bagCards(&arena); Vec<Id> retained(&arena);
        for (auto id : parts) {
            const auto& n = node(id); budget.tick(n.cards.size() + 1);
            if (n.kind == Kind::Bag || (n.kind == Kind::Word && n.cards.front() == n.cards.back()))
                bagCards.insert(bagCards.end(), n.cards.begin(), n.cards.end());
            else retained.push_back(id);
        }
        if (!bagCards.empty()) retained.push_back(bag(bagCards));
        if (retained.size() == 1) return retained.front();
        sortCharge(retained.size()); std::sort(retained.begin(), retained.end());
        Node n(Kind::Shuffle, &arena); n.children = retained;
        for (auto id : retained) {
            const auto& cards = node(id).cards; checkCards(n.cards.size() + cards.size());
            n.cards.insert(n.cards.end(), cards.begin(), cards.end());
        }
        return make(std::move(n));
    }
    Id belowTop(Id root, Label card) {
        const auto& n = node(root); Cards single({card}, &arena);
        if (n.cards.empty()) return word(single);
        checkCards(n.cards.size() + 1);
        Label head; Id tail;
        if (knownHead(root, head, tail)) {
            Vec<Id> parts({tail, word(single)}, &arena);
            return concat(word(Cards({head}, &arena)), shuffle(parts));
        }
        Node out(Kind::Protect, &arena); out.cards = n.cards; out.cards.push_back(card);
        if (n.kind == Kind::Protect) {
            Vec<Id> parts({n.children[1], word(single)}, &arena);
            out.children = {n.children[0], shuffle(parts)};
        } else out.children = {root, word(single)};
        return make(std::move(out));
    }
    Id erase(Id root, Label card) {
        if (!multiplicity(root, card)) throw std::invalid_argument("missing selected public card");
        Vec<std::pair<Id, bool>> path(&arena); Id cur = root;
        while (node(cur).kind == Kind::Concat) {
            const auto& n = node(cur);
            bool left = multiplicity(n.children[0], card) != 0, right = multiplicity(n.children[1], card) != 0;
            if (left && right) break;
            path.push_back({left ? n.children[1] : n.children[0], left});
            cur = left ? n.children[0] : n.children[1];
        }
        const auto& n = node(cur); Id result;
        if (n.kind == Kind::Family && card != n.label) {
            Cards pool = without(n.order, card); Vec<CountTerm> terms(n.counts, &arena);
            for (auto& t : terms) { if (t.ghosts == budget.limits.cards) throw Failure("card_limit"); ++t.ghosts; }
            result = family(n.label, pool, n.markers, terms).root;
        } else if (n.kind == Kind::Bag) result = bag(without(n.cards, card));
        else if (n.kind == Kind::Word && (multiplicity(cur, card) == 1 || n.cards.front() == n.cards.back())) result = word(without(n.order, card));
        else {
            Node out(Kind::Delete, &arena); out.children = {cur}; out.label = card; out.cards = without(n.cards, card);
            result = make(std::move(out));
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) result = it->second ? concat(result, it->first) : concat(it->first, result);
        return result;
    }
    // Uniform deletion among this label's occurrences EXCEPT its last one.
    // A compact transducer, not a mixture enumerating every eligible rank.
    Id eraseEarlier(Id root, Label card) {
        if (multiplicity(root, card) < 2) throw Failure("event_contract");
        Node out(Kind::DeleteEarlier, &arena); out.children = {root}; out.label = card;
        out.cards = without(node(root).cards, card); return make(std::move(out));
    }
    Result mixture(const Vec<Term>& input) {
        std::pmr::map<Id, Rational> masses(&arena);
        for (const auto& t : input) { node(t.child); addMass(masses, t.child, t.mass); }
        Rational total;
        for (const auto& p : masses) total = Rational::add(total, p.second, budget);
        if (!total) return {total, empty_};
        if (masses.size() == 1) return {total, masses.begin()->first};
        Node n(Kind::Mix, &arena); n.cards = node(masses.begin()->first).cards;
        for (const auto& p : masses) {
            budget.tick(n.cards.size() + 1);
            if (node(p.first).cards != n.cards) throw std::invalid_argument("mixture inventory differs");
            n.terms.push_back({p.first, Rational::divide(p.second, total, budget)});
        }
        return {total, make(std::move(n))};
    }
    Id mapped(Id root, const Vec<std::pair<Label, Label>>& mapping) {
        const auto& original = node(root);
        auto transform = [&](Label c) {
            budget.tick(mapping.size() + 1);
            for (const auto& p : mapping) if (p.first == c) return p.second;
            return c;
        };
        std::pmr::set<Label> unique(&arena);
        for (const auto& p : mapping) if (!unique.insert(p.first).second) throw std::invalid_argument("duplicate public map key");
        Node n(Kind::Map, &arena); n.children = {root}; bool changed = false;
        for (auto c : original.cards) { Label next = transform(c); n.cards.push_back(next); changed |= next != c;
            if (n.mapping.empty() || n.mapping.back().first != c) n.mapping.push_back({c, next}); }
        if (!changed) return root;
        if (original.kind == Kind::Word) {
            Cards order(&arena); for (auto c : original.order) order.push_back(transform(c)); return word(order);
        }
        if (original.kind == Kind::Bag) return bag(n.cards);
        // In particular, do not commute a many-to-one map through deletion.
        return make(std::move(n));
    }

    bool selectionPossible(Id root, const Cards& matches, Label selected, unsigned rank) {
        budget.tick(node(root).cards.size() * (matches.size() + 1) + 1);
        unsigned count = 0;
        for (auto c : node(root).cards) if (std::binary_search(matches.begin(), matches.end(), c)) ++count;
        return rank < count && multiplicity(root, selected) && std::binary_search(matches.begin(), matches.end(), selected);
    }
    Id selectionNode(Id root, const Cards& matches, Label selected, unsigned rank, Kind kind = Kind::SelectErase) {
        Node n(kind, &arena); n.children = {root}; n.label = selected;
        n.markers = rank; n.order = matches;
        n.cards = kind == Kind::SelectErase ? without(node(root).cards, selected) : node(root).cards;
        return make(std::move(n));
    }
    // Inventory can prove rank evidence certain without enumerating
    // any prefix: all eligible occurrences have the selected INTERNAL symbol.
    // Different causal roles remain different symbols, even for equal cards.
    bool certainSelection(Id root, const Cards& matches, Label selected) {
        budget.tick(node(root).cards.size() * (matches.size() + 1) + 1);
        for (auto c : node(root).cards)
            if (c != selected && std::binary_search(matches.begin(), matches.end(), c)) return false;
        return true; // Caller has already checked rank/selected feasibility.
    }
    Result eraseSelected(Id root, const Cards& matches, Label selected, unsigned rank) {
        if (!std::is_sorted(matches.begin(), matches.end()) || std::adjacent_find(matches.begin(), matches.end()) != matches.end())
            throw Failure("event_contract");
        if (!selectionPossible(root, matches, selected, rank)) return {Rational(), empty_};
        // Unique eligible occurrence: ranked and uniform deletion coincide.
        // Multiple equal occurrences MUST retain their positional selector.
        if (multiplicity(root, selected) == 1 && certainSelection(root, matches, selected))
            return {Rational(1), erase(root, selected)};
        auto id = selectionNode(root, matches, selected, rank);
        return evaluate(id, 0, true);
    }
    // Observe an eligible occurrence WITHOUT moving it. A selector's visible
    // presentation can reveal positional evidence before the player chooses.
    Result observeSelected(Id root, const Cards& matches, Label selected, unsigned rank) {
        if (!std::is_sorted(matches.begin(), matches.end()) || std::adjacent_find(matches.begin(), matches.end()) != matches.end())
            throw Failure("event_contract");
        if (!selectionPossible(root, matches, selected, rank)) return {Rational(), empty_};
        if (certainSelection(root, matches, selected)) return {Rational(1), root};
        auto id = selectionNode(root, matches, selected, rank, Kind::SelectObserve);
        auto result = evaluate(id, 0, true);
        if (result.likelihood == Rational(1)) result.root = root;
        return result;
    }

    // Pop and rank-selection likelihoods share ONE explicit dependency stack.
    // Selected removal is not uniform deletion: keep the requested occurrence
    // rank, even when equal public cards occupy different constrained regions.
    Result evaluate(Id root, Label observed, bool massQuery = false) {
        struct Frame { Query key; unsigned phase; bool mass = false; };
        Vec<Frame> stack(&arena); stack.push_back({{root, observed}, 0, massQuery});
        while (!stack.empty()) {
            budget.tick(1 + logarithm(cache_.size() + 1));
            auto key = stack.back().key;
            const auto& n = node(key.first); Label c = key.second;
            auto dependency = [&](Id id, Label label) { stack.push_back({{id, label}, 0}); };
            auto massDependency = [&](Id id) { stack.push_back({{id, 0}, 0, true}); };
            auto matched = [&](Label label) { return std::binary_search(n.order.begin(), n.order.end(), label); };
            if (stack.back().mass) {
                if (selectionMass_.find(key.first) != selectionMass_.end()) { stack.pop_back(); continue; }
                if (n.kind != Kind::SelectErase && n.kind != Kind::SelectObserve) throw std::logic_error("selection query on ordinary node");
                auto base = n.children[0]; const auto& b = node(base);
                if (certainSelection(base, n.order, n.label)) {
                    selectionMass_.emplace(key.first, Rational(1)); stack.pop_back(); continue;
                }
                if (b.kind == Kind::Bag || b.kind == Kind::Word) {
                    Rational z;
                    if (b.kind == Kind::Bag) {
                        unsigned count = 0; for (auto label : b.cards) if (matched(label)) ++count;
                        z = Rational::ratio(multiplicity(base, n.label), count, budget);
                    } else {
                        unsigned rank = n.markers;
                        for (auto label : b.order) if (matched(label)) {
                            if (!rank) { z = Rational(label == n.label ? 1 : 0); break; }
                            --rank;
                        }
                    }
                    selectionMass_.emplace(key.first, z); stack.pop_back(); continue;
                }
                if (!stack.back().phase) {
                    stack.back().phase = 1;
                    Label previous = 0; bool first = true;
                    for (auto label : b.cards) if (first || label != previous) {
                        dependency(base, label); previous = label; first = false;
                    }
                    continue;
                }
                if (stack.back().phase == 1) {
                    stack.back().phase = 2;
                    Label previous = 0; bool first = true;
                    for (auto label : b.cards) if (first || label != previous) {
                        previous = label; first = false;
                        auto part = cached(base, label);
                        if (!part.likelihood || (matched(label) && !n.markers)) continue;
                        auto rank = n.markers - unsigned(matched(label));
                        if (selectionPossible(part.root, n.order, n.label, rank))
                            massDependency(selectionNode(part.root, n.order, n.label, rank, n.kind));
                    }
                    continue;
                }
                Rational z; Label previous = 0; bool first = true;
                for (auto label : b.cards) if (first || label != previous) {
                    previous = label; first = false;
                    auto part = cached(base, label); if (!part.likelihood) continue;
                    Rational continuation;
                    if (matched(label) && !n.markers) continuation = Rational(label == n.label ? 1 : 0);
                    else {
                        auto rank = n.markers - unsigned(matched(label));
                        if (selectionPossible(part.root, n.order, n.label, rank))
                            continuation = selectionMass_.at(selectionNode(part.root, n.order, n.label, rank, n.kind));
                    }
                    z = Rational::add(z, Rational::multiply(part.likelihood, continuation, budget), budget);
                }
                selectionMass_.emplace(key.first, z); stack.pop_back(); continue;
            }
            if (cache_.find(key) != cache_.end()) { stack.pop_back(); continue; }
            if (!multiplicity(key.first, c)) { save(key, {Rational(), empty_}); stack.pop_back(); continue; }
            if (!stack.back().phase) {
                stack.back().phase = 1;
                switch (n.kind) {
                    case Kind::Concat: case Kind::Protect: dependency(n.children[0], c); break;
                    case Kind::Shuffle: for (auto child : n.children) dependency(child, c); break;
                    case Kind::Delete: case Kind::DeleteEarlier: dependency(n.children[0], c); dependency(n.children[0], n.label); break;
                    case Kind::Mix: for (const auto& t : n.terms) dependency(t.child, c); break;
                    case Kind::Map: for (const auto& p : n.mapping) if (p.second == c) dependency(n.children[0], p.first); break;
                    case Kind::SelectErase: case Kind::SelectObserve:
                        dependency(n.children[0], c); dependency(n.children[0], n.label); massDependency(key.first); break;
                    default: break;
                }
                continue;
            }
            if ((n.kind == Kind::Delete || n.kind == Kind::DeleteEarlier) && stack.back().phase == 1) {
                stack.back().phase = 2;
                auto first = cached(n.children[0], n.label);
                if (first.likelihood) { dependency(first.root, c); continue; }
            }
            if ((n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) && stack.back().phase == 1) {
                stack.back().phase = 2;
                auto keep = cached(n.children[0], c);
                if (keep.likelihood && !(matched(c) && !n.markers)) {
                    auto rank = n.markers - unsigned(matched(c));
                    if (selectionPossible(keep.root, n.order, n.label, rank))
                        massDependency(selectionNode(keep.root, n.order, n.label, rank, n.kind));
                }
                auto erased = cached(n.children[0], n.label);
                if (n.kind == Kind::SelectErase && !n.markers && erased.likelihood) dependency(erased.root, c);
                continue;
            }
            Result result{Rational(), empty_}; Vec<Term> terms(&arena);
            switch (n.kind) {
                case Kind::Word:
                    if (n.order.front() == c) result = {Rational(1), word(Cards(n.order.begin() + 1, n.order.end(), &arena))};
                    break;
                case Kind::Bag:
                    result = {Rational::ratio(multiplicity(key.first, c), n.cards.size(), budget), bag(without(n.cards, c))}; break;
                case Kind::Concat: case Kind::Protect: {
                    auto child = cached(n.children[0], c);
                    if (child.likelihood) {
                        Id tail = n.kind == Kind::Concat ? concat(child.root, n.children[1]) : shuffle(Vec<Id>({child.root, n.children[1]}, &arena));
                        result = {child.likelihood, tail};
                    }
                    break;
                }
                case Kind::Shuffle:
                    for (unsigned i = 0; i < n.children.size(); ++i) {
                        Id id = n.children[i]; auto child = cached(id, c);
                        if (!child.likelihood) continue;
                        Vec<Id> children(n.children, &arena); children[i] = child.root;
                        auto mass = Rational::multiply(Rational::ratio(node(id).cards.size(), n.cards.size(), budget), child.likelihood, budget);
                        terms.push_back({shuffle(children), mass});
                    }
                    result = mixture(terms); break;
                case Kind::Delete: case Kind::DeleteEarlier: {
                    auto base = n.children[0]; auto m = multiplicity(base, n.label) - unsigned(n.kind == Kind::DeleteEarlier);
                    auto first = cached(base, c);
                    if (first.likelihood && (c != n.label || m > 1)) {
                        auto keep = c == n.label ? Rational::ratio(m - 1, m, budget) : Rational(1);
                        auto tail = n.kind == Kind::DeleteEarlier ? eraseEarlier(first.root, n.label) : erase(first.root, n.label);
                        terms.push_back({tail, Rational::multiply(first.likelihood, keep, budget)});
                    }
                    auto skipped = cached(base, n.label);
                    if (skipped.likelihood) {
                        auto second = cached(skipped.root, c);
                        if (second.likelihood) terms.push_back({second.root, Rational::multiply(
                            Rational::multiply(skipped.likelihood, second.likelihood, budget), Rational::ratio(1, m, budget), budget)});
                    }
                    result = mixture(terms); break;
                }
                case Kind::Mix:
                    for (const auto& t : n.terms) { auto child = cached(t.child, c); if (child.likelihood)
                        terms.push_back({child.root, Rational::multiply(t.mass, child.likelihood, budget)}); }
                    result = mixture(terms); break;
                case Kind::Map:
                    for (const auto& p : n.mapping) if (p.second == c) { auto child = cached(n.children[0], p.first);
                        if (child.likelihood) terms.push_back({mapped(child.root, n.mapping), child.likelihood}); }
                    result = mixture(terms); break;
                case Kind::Family: result = familyPop(key.first, c); break;
                case Kind::SelectErase: case Kind::SelectObserve: {
                    auto z = selectionMass_.at(key.first); if (!z) break;
                    auto first = cached(n.children[0], c);
                    if (first.likelihood && !(matched(c) && !n.markers)) {
                        auto rank = n.markers - unsigned(matched(c));
                        if (selectionPossible(first.root, n.order, n.label, rank)) {
                            auto tail = selectionNode(first.root, n.order, n.label, rank, n.kind);
                            auto mass = Rational::divide(Rational::multiply(first.likelihood, selectionMass_.at(tail), budget), z, budget);
                            if (mass) terms.push_back({tail, mass});
                        }
                    }
                    auto skipped = cached(n.children[0], n.label);
                    if (n.kind == Kind::SelectObserve && !n.markers && c == n.label && first.likelihood) {
                        terms.push_back({first.root, Rational::divide(first.likelihood, z, budget)});
                    }
                    if (n.kind == Kind::SelectErase && !n.markers && skipped.likelihood) {
                        auto second = cached(skipped.root, c);
                        if (second.likelihood) terms.push_back({second.root,
                            Rational::divide(Rational::multiply(skipped.likelihood, second.likelihood, budget), z, budget)});
                    }
                    result = mixture(terms); break;
                }
            }
            save(key, result); stack.pop_back();
        }
        return massQuery ? Result{selectionMass_.at(root), root} : cached(root, observed);
    }
    Result pop(Id root, Label observed) { return evaluate(root, observed); }

    Result unorderedDraw(Id root, const Cards& observed) {
        checkCards(observed.size());
        Cards batch(observed, &arena); sortCharge(batch.size()); std::sort(batch.begin(), batch.end());
        const auto& original = node(root).cards;
        if (!std::includes(original.begin(), original.end(), batch.begin(), batch.end())) return {Rational(), empty_};
        std::pmr::map<Id, Rational> current(&arena); current.emplace(root, Rational(1));
        for (std::size_t step = 0; step < batch.size(); ++step) {
            std::pmr::map<Id, Rational> next(&arena);
            for (const auto& state : current) {
                Cards removed(&arena), remaining(&arena);
                const auto& now = node(state.first).cards;
                budget.tick(original.size() + now.size() + batch.size() + 1);
                std::set_difference(original.begin(), original.end(), now.begin(), now.end(), std::back_inserter(removed));
                std::set_difference(batch.begin(), batch.end(), removed.begin(), removed.end(), std::back_inserter(remaining));
                auto last = std::unique(remaining.begin(), remaining.end()); remaining.erase(last, remaining.end());
                for (auto c : remaining) {
                    auto part = pop(state.first, c);
                    addMass(next, part.root, Rational::multiply(state.second, part.likelihood, budget));
                }
            }
            current = std::move(next);
        }
        Vec<Term> terms(&arena); for (const auto& p : current) terms.push_back({p.first, p.second});
        return mixture(terms);
    }
    Result reveal(Id root, const Cards& topFirst) {
        if (topFirst.size() != node(root).cards.size()) return {Rational(), empty_};
        Rational z(1); Id cur = root;
        for (auto c : topFirst) { auto step = pop(cur, c); z = Rational::multiply(z, step.likelihood, budget);
            if (!z) return {z, empty_};
            cur = step.root;
        }
        return {z, word(topFirst)}; // Revealing is not drawing/removing the cards.
    }

    Cards sample(Id root, std::mt19937_64& rng) {
        struct Task { Id root; bool combine; };
        Vec<Task> stack(&arena); Vec<Cards> values(&arena); stack.push_back({root, false});
        while (!stack.empty()) {
            budget.tick(); auto task = stack.back(); stack.pop_back(); const auto& n = node(task.root);
            if (!task.combine) {
                if (n.kind == Kind::Word || n.kind == Kind::Bag) {
                    Cards out(n.order, &arena); budget.tick(out.size() + 1);
                    if (n.kind == Kind::Bag) permute(out, rng);
                    values.push_back(std::move(out));
                } else if (n.kind == Kind::Family) values.push_back(sampleFamily(n, rng));
                else if ((n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) &&
                         certainSelection(n.children[0], n.order, n.label)) {
                    stack.push_back({task.root, true});
                    stack.push_back({n.children[0], false});
                } else if ((n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) && node(n.children[0]).kind == Kind::Bag) {
                    values.push_back(sampleRankedBag(*this, n, n.kind == Kind::SelectErase,
                        [&](const auto& masses) { return categorical(masses, rng); }));
                } else if (n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) {
                    Cards out(&arena); Id cur = task.root;
                    while (!node(cur).cards.empty()) {
                        Cards options(node(cur).cards, &arena);
                        options.erase(std::unique(options.begin(), options.end()), options.end());
                        Vec<Rational> masses(&arena); Vec<Id> tails(&arena);
                        for (auto label : options) { auto part = pop(cur, label); masses.push_back(part.likelihood); tails.push_back(part.root); }
                        auto chosen = categorical(masses, rng); out.push_back(options[chosen]); cur = tails[chosen];
                    }
                    values.push_back(std::move(out));
                }
                else if (n.kind == Kind::Mix) {
                    Vec<Rational> masses(&arena); for (const auto& t : n.terms) masses.push_back(t.mass);
                    stack.push_back({n.terms[categorical(masses, rng)].child, false});
                } else {
                    stack.push_back({task.root, true});
                    for (auto it = n.children.rbegin(); it != n.children.rend(); ++it) stack.push_back({*it, false});
                }
                continue;
            }
            if (values.size() < n.children.size()) throw std::logic_error("invalid sample stack");
            auto first = values.size() - n.children.size(); Cards out(&arena);
            if (n.kind == Kind::Concat) {
                out = std::move(values[first]); const auto& other = values[first + 1];
                budget.tick(out.size() + other.size() + 1); out.insert(out.end(), other.begin(), other.end());
            } else if (n.kind == Kind::SelectErase || n.kind == Kind::SelectObserve) {
                out = std::move(values[first]); unsigned rank=n.markers; bool found=false;
                for (auto pos=out.begin(); pos!=out.end(); ++pos) {
                    budget.tick(); if (*pos!=n.label) continue;
                    if (rank) { --rank; continue; }
                    if (n.kind==Kind::SelectErase) out.erase(pos);
                    found=true; break;
                }
                if (!found) throw Failure("certain_selector_sample_contract");
            } else if (n.kind == Kind::Delete || n.kind == Kind::DeleteEarlier) {
                out = std::move(values[first]); Vec<unsigned> matches(&arena);
                budget.tick(out.size() + 1);
                for (unsigned i = 0; i < out.size(); ++i) if (out[i] == n.label) matches.push_back(i);
                if (n.kind == Kind::DeleteEarlier) matches.pop_back();
                auto j = randomBelow(UInt(matches.size()), rng, budget).limb[0]; out.erase(out.begin() + matches[j]);
            } else if (n.kind == Kind::Map) {
                out = std::move(values[first]);
                for (auto& c : out) { budget.tick(n.mapping.size() + 1);
                    for (const auto& p : n.mapping) if (p.first == c) { c = p.second; break; } }
            } else {
                Vec<unsigned> offsets(n.children.size(), 0, &arena), schedule(&arena);
                if (n.kind == Kind::Protect) { out.push_back(values[first].front()); offsets[0] = 1; }
                for (unsigned i = 0; i < n.children.size(); ++i)
                    for (auto j = offsets[i]; j < values[first + i].size(); ++j) { budget.tick(); schedule.push_back(i); }
                permute(schedule, rng);
                for (auto i : schedule) { budget.tick(); out.push_back(values[first + i][offsets[i]++]); }
            }
            values.erase(values.begin() + first, values.end()); values.push_back(std::move(out));
        }
        if (values.size() != 1) throw std::logic_error("invalid final sample stack");
        return std::move(values.front());
    }
};

} // namespace spire_belief
