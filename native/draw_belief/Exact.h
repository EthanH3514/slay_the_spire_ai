#pragma once

// Bounded non-negative exact arithmetic. No floating point, external bignum
// dependency, recursive division, or unchecked growing integer allocation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <random>
#include <stdexcept>
#include <string>

namespace spire_belief {

struct Failure : std::runtime_error {
    const std::string status;
    explicit Failure(const char* reason) : std::runtime_error(reason), status(reason) {}
};

struct Limits {
    std::size_t bytes = 16 * 1024 * 1024;
    std::size_t nodes = 20000;
    std::size_t cards = 4096;
    std::uint64_t work = 1000000;
    unsigned bits = 4096;
    std::uint64_t randomWords = 100000;
    unsigned rejectionAttempts = 64;
};

// All dynamically allocated kernel containers use this resource. The engine
// object and fixed-size arithmetic temporaries are additional bounded storage;
// caller-owned input/output, a simulator and the C runtime are not in this cap.
class Arena final : public std::pmr::memory_resource {
    std::size_t cap_;
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes > cap_ - current) throw Failure("space_limit");
        void* p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        current += bytes;
        peak = std::max(peak, current);
        return p;
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        current -= bytes;
        std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
public:
    std::size_t current = 0, peak = 0;
    explicit Arena(std::size_t cap) : cap_(cap) {}
};

struct Budget {
    Limits limits;
    std::uint64_t work = 0, randomWords = 0;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();

    explicit Budget(Limits value = {}) : limits(value) {
        if (!value.bytes || !value.nodes || !value.cards || !value.work ||
            !value.bits || value.bits > 4096 || !value.randomWords || !value.rejectionAttempts)
            throw std::invalid_argument("invalid draw-belief limits");
    }
    void begin(std::chrono::steady_clock::time_point until = std::chrono::steady_clock::time_point::max()) {
        work = randomWords = 0;
        deadline = until;
    }
    void tick(std::uint64_t amount = 1) {
        if (amount > limits.work - work) throw Failure("work_limit");
        work += amount;
        if (std::chrono::steady_clock::now() >= deadline) throw Failure("deadline");
    }
    void precision(unsigned bits) {
        tick();
        if (bits > limits.bits) throw Failure("precision_limit");
    }
};

// Little-endian base-2^32 limbs. A value occupies fixed storage even at a small
// configured precision. Keeping 4096-bit storage fixed also bounds intermediates.
struct UInt {
    static constexpr unsigned capacity = 128;
    std::array<std::uint32_t, capacity> limb{};
    unsigned used = 0;

    UInt(std::uint64_t v = 0) {
        limb[0] = std::uint32_t(v);
        limb[1] = std::uint32_t(v >> 32);
        used = limb[1] ? 2 : limb[0] ? 1 : 0;
    }
    explicit operator bool() const { return used != 0; }
    void trim() { while (used && !limb[used - 1]) --used; }
    unsigned bits() const {
        if (!used) return 0;
        unsigned result = 32 * (used - 1);
        for (auto v = limb[used - 1]; v; v >>= 1) ++result;
        return result;
    }
    int compare(const UInt& b) const {
        if (used != b.used) return used < b.used ? -1 : 1;
        for (unsigned i = used; i--;) if (limb[i] != b.limb[i]) return limb[i] < b.limb[i] ? -1 : 1;
        return 0;
    }
    bool operator==(const UInt& b) const { return compare(b) == 0; }
    bool operator!=(const UInt& b) const { return compare(b) != 0; }
    bool operator<(const UInt& b) const { return compare(b) < 0; }

    static UInt add(const UInt& a, const UInt& b, Budget& budget) {
        UInt out;
        unsigned n = std::max(a.used, b.used);
        budget.tick(n + 1);
        std::uint64_t carry = 0;
        for (unsigned i = 0; i < n; ++i) {
            std::uint64_t v = std::uint64_t(a.limb[i]) + b.limb[i] + carry;
            out.limb[i] = std::uint32_t(v); carry = v >> 32;
        }
        out.used = n;
        if (carry) {
            if (n == capacity) throw Failure("precision_limit");
            out.limb[out.used++] = std::uint32_t(carry);
        }
        budget.precision(out.bits());
        return out;
    }
    static UInt subtract(const UInt& a, const UInt& b, Budget& budget) {
        budget.tick(a.used + b.used + 1);
        if (a < b) throw std::invalid_argument("negative exact mass");
        UInt out;
        std::uint64_t borrow = 0;
        for (unsigned i = 0; i < a.used; ++i) {
            std::uint64_t sub = std::uint64_t(b.limb[i]) + borrow;
            out.limb[i] = std::uint32_t(std::uint64_t(a.limb[i]) - sub);
            borrow = a.limb[i] < sub;
        }
        out.used = a.used; out.trim();
        return out;
    }
    static UInt multiply(const UInt& a, const UInt& b, Budget& budget) {
        if (!a || !b) return {};
        budget.tick(std::uint64_t(a.used) * b.used + 1);
        // The full product needs at most a.bits()+b.bits() bits. A possible
        // one-bit excess is checked in the fixed-width multiplication below.
        if (a.bits() + b.bits() - 1 > budget.limits.bits) throw Failure("precision_limit");
        UInt out;
        for (unsigned i = 0; i < a.used; ++i) {
            std::uint64_t carry = 0;
            for (unsigned j = 0; j < b.used; ++j) {
                const unsigned k = i + j;
                if (k >= capacity) throw Failure("precision_limit");
                std::uint64_t value = std::uint64_t(a.limb[i]) * b.limb[j] + out.limb[k] + carry;
                out.limb[k] = std::uint32_t(value); carry = value >> 32;
            }
            const unsigned k = i + b.used;
            if (carry) {
                if (k >= capacity) throw Failure("precision_limit");
                out.limb[k] = std::uint32_t(carry);
            }
        }
        out.used = std::min(capacity, a.used + b.used); out.trim();
        budget.precision(out.bits());
        return out;
    }
    static UInt shifted(const UInt& a, unsigned shift, Budget& budget) {
        if (!a) return {};
        budget.precision(a.bits() + shift);
        budget.tick(a.used + shift / 32 + 1);
        UInt out;
        unsigned whole = shift / 32, part = shift % 32;
        for (unsigned i = 0; i < a.used; ++i) {
            std::uint64_t value = std::uint64_t(a.limb[i]) << part;
            out.limb[i + whole] |= std::uint32_t(value);
            if ((value >> 32) && i + whole + 1 < capacity) out.limb[i + whole + 1] |= std::uint32_t(value >> 32);
        }
        out.used = (a.bits() + shift + 31) / 32; out.trim();
        return out;
    }
    void halve(Budget& budget) {
        budget.tick(used + 1);
        std::uint32_t carry = 0;
        for (unsigned i = used; i--;) {
            auto next = limb[i] & 1;
            limb[i] = (limb[i] >> 1) | (carry << 31); carry = next;
        }
        trim();
    }
    static std::pair<UInt, UInt> divide(const UInt& a, const UInt& b, Budget& budget) {
        if (!b) throw std::invalid_argument("exact division by zero");
        budget.tick(a.used + b.used + 1);
        if (a < b) return {UInt(), a};
        if (b == UInt(1)) return {a, UInt()};
        if (a == b) return {UInt(1), UInt()};
        if (b.used == 1) {
            UInt quotient;
            std::uint64_t remainder = 0;
            for (unsigned i = a.used; i--;) {
                auto part = (remainder << 32) | a.limb[i];
                quotient.limb[i] = std::uint32_t(part / b.limb[0]);
                remainder = part % b.limb[0];
            }
            quotient.used = a.used; quotient.trim();
            return {quotient, UInt(remainder)};
        }
        if (a.used <= 2 && b.used <= 2) {
            auto av = std::uint64_t(a.limb[0]) | (std::uint64_t(a.limb[1]) << 32);
            auto bv = std::uint64_t(b.limb[0]) | (std::uint64_t(b.limb[1]) << 32);
            return {UInt(av / bv), UInt(av % bv)};
        }
        unsigned shift = a.bits() - b.bits();
        UInt divisor = shifted(b, shift, budget), remainder = a, quotient;
        for (unsigned j = shift + 1; j--;) {
            budget.tick(remainder.used + divisor.used + 1);
            if (!(remainder < divisor)) {
                remainder = subtract(remainder, divisor, budget);
                quotient.limb[j / 32] |= std::uint32_t(1) << (j % 32);
            }
            if (j) divisor.halve(budget);
        }
        quotient.used = shift / 32 + 1; quotient.trim();
        return {quotient, remainder};
    }
    static UInt gcd(UInt a, UInt b, Budget& budget) {
        budget.tick(a.used + b.used + 1);
        if (a == b) return a;
        while (b) { auto next = divide(a, b, budget).second; a = b; b = next; }
        return a;
    }
    static UInt parse(const std::string& text, Budget& budget) {
        if (text.empty()) throw std::invalid_argument("empty exact integer");
        UInt out;
        for (char c : text) {
            if (c < '0' || c > '9') throw std::invalid_argument("invalid exact integer");
            out = add(multiply(out, UInt(10), budget), UInt(c - '0'), budget);
        }
        return out;
    }
    std::string decimal(Budget& budget) const {
        if (!*this) return "0";
        UInt v = *this;
        std::string out;
        while (v) {
            budget.tick(v.used + 1);
            std::uint64_t carry = 0;
            for (unsigned i = v.used; i--;) {
                std::uint64_t part = (carry << 32) | v.limb[i];
                v.limb[i] = std::uint32_t(part / 10); carry = part % 10;
            }
            out.push_back(char('0' + carry)); v.trim();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
};

struct Rational {
    UInt numerator, denominator = UInt(1);
    Rational(std::uint64_t value = 0) : numerator(value) {}
    Rational(UInt n, UInt d, Budget& budget) : numerator(n), denominator(d) {
        if (!d) throw std::invalid_argument("zero probability denominator");
        budget.precision(n.bits()); budget.precision(d.bits());
        if (!n) { denominator = UInt(1); return; }
        if (d == UInt(1)) return;
        if (n == d) { numerator = denominator = UInt(1); return; }
        auto common = UInt::gcd(n, d, budget);
        numerator = UInt::divide(n, common, budget).first;
        denominator = UInt::divide(d, common, budget).first;
    }
    explicit operator bool() const { return bool(numerator); }
    bool operator==(const Rational& b) const { return numerator == b.numerator && denominator == b.denominator; }
    static Rational ratio(std::uint64_t n, std::uint64_t d, Budget& budget) { return Rational(UInt(n), UInt(d), budget); }
    static Rational add(const Rational& a, const Rational& b, Budget& budget) {
        budget.tick();
        if (!a) return b;
        if (!b) return a;
        auto common = UInt::gcd(a.denominator, b.denominator, budget);
        auto ad = UInt::divide(a.denominator, common, budget).first;
        auto bd = UInt::divide(b.denominator, common, budget).first;
        return Rational(UInt::add(UInt::multiply(a.numerator, bd, budget), UInt::multiply(b.numerator, ad, budget), budget),
                        UInt::multiply(ad, b.denominator, budget), budget);
    }
    static Rational multiply(const Rational& a, const Rational& b, Budget& budget) {
        budget.tick();
        if (!a || !b) return {};
        if (a == Rational(1)) return b;
        if (b == Rational(1)) return a;
        auto g1 = UInt::gcd(a.numerator, b.denominator, budget);
        auto g2 = UInt::gcd(b.numerator, a.denominator, budget);
        return Rational(UInt::multiply(UInt::divide(a.numerator, g1, budget).first, UInt::divide(b.numerator, g2, budget).first, budget),
                        UInt::multiply(UInt::divide(a.denominator, g2, budget).first, UInt::divide(b.denominator, g1, budget).first, budget), budget);
    }
    static Rational divide(const Rational& a, const Rational& b, Budget& budget) {
        budget.tick();
        if (!b) throw std::invalid_argument("zero probability normalization");
        if (b == Rational(1)) return a;
        Rational inverse; inverse.numerator = b.denominator; inverse.denominator = b.numerator;
        return multiply(a, inverse, budget);
    }
    std::string text(Budget& budget) const { return numerator.decimal(budget) + "/" + denominator.decimal(budget); }
};

inline UInt randomBelow(const UInt& exclusive, std::mt19937_64& rng, Budget& budget) {
    if (!exclusive) throw std::invalid_argument("empty exact categorical range");
    if (exclusive == UInt(1)) return {};
    unsigned bits = UInt::subtract(exclusive, UInt(1), budget).bits();
    for (unsigned attempt = 0; attempt < budget.limits.rejectionAttempts; ++attempt) {
        UInt pick;
        pick.used = (bits + 31) / 32;
        for (unsigned i = 0; i < pick.used; i += 2) {
            budget.tick();
            if (budget.randomWords == budget.limits.randomWords) throw Failure("random_limit");
            ++budget.randomWords;
            auto word = rng();
            pick.limb[i] = std::uint32_t(word);
            if (i + 1 < pick.used) pick.limb[i + 1] = std::uint32_t(word >> 32);
        }
        if (bits % 32) pick.limb[pick.used - 1] &= (std::uint32_t(1) << (bits % 32)) - 1;
        pick.trim(); budget.tick(pick.used + exclusive.used + 1);
        if (pick < exclusive) return pick;
    }
    throw Failure("random_rejection_limit");
}

} // namespace spire_belief
