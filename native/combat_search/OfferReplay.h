#pragma once
#include "../draw_belief/SimulatorHooks.h"
#include "../draw_belief/Continuation.h"

namespace spire_offer_replay {
using J = spire_search::J;
using namespace sts;
using namespace spire_belief;
inline constexpr auto version = "public-offer-replay-v1";

// A Discovery window stops action execution at its first offer. Bind the
// observed offer to THAT historical action, regardless of its outer trigger.
// The ordinal is owned by each reconstructed world, not shared sampler state.
// No event is armed after the recorded history ends.
struct Program {
    struct Step { std::string action; std::optional<trace::EventOffer> offer; J final; };
    std::vector<Step> steps;
    std::size_t events = 0, work = 0;
    std::size_t validationWork = 0;
    bool validationCharged = false;
    unsigned firstPool = 0;

    Program(const J& anchor, const std::vector<PublicStep>& history) {
        J previous = anchor;
        for (const auto& item : history) {
            if (item.observations.empty()) throw Failure("offer_replay_missing_observation");
            const auto action = J::parse(item.action);
            const auto observed = J::parse(item.observations.back());
            Step step{item.action, {}, observed};
            if (observed.at("input") == int(InputState::CARD_SELECT) &&
                (observed.at("selection").value("task", "") == "DISCOVERY" ||
                 observed.at("selection").value("task", "") == "CODEX")) {
                // These are the actual generation pools in the supported
                // Ironclad simulator: direct potion, opening Toolbox, or the
                // Discovery card (manual, Havoc, Mayhem, duplication, etc.).
                // The hook additionally checks the actual pool at consumption.
                CardType type = CardType::INVALID;
                // Codex always uses the Ironclad mixed combat pool, irrespective of trigger.
                if (observed.at("selection").value("task", "") == "CODEX") type = CardType::INVALID;
                else if (action.value("kind", "") == "opening_prelude") type = CardType::STATUS;
                else if (action.value("kind", "") == "use_potion") {
                    const auto id = static_cast<Potion>(previous.at("potions").at(
                        action.at("potion_index").get<int>()).get<int>());
                    switch (id) {
                        case Potion::ATTACK_POTION: type=CardType::ATTACK; break;
                        case Potion::SKILL_POTION: type=CardType::SKILL; break;
                        case Potion::POWER_POTION: type=CardType::POWER; break;
                        case Potion::COLORLESS_POTION: type=CardType::STATUS; break;
                        default: break; // Indirect auto-play uses Discovery's mixed pool.
                    }
                }
                const auto& candidates=observed.at("selection").at("candidates");
                if (candidates.size()!=3) throw Failure("offer_replay_candidate_count");
                std::array<CardId,3> cards{};
                for (std::size_t i=0;i<cards.size();++i) {
                    bool found=false;
                    for (int c=1;c<=int(CardId::ZAP);++c)
                        if (spire_search::key(getCardEnumName(static_cast<CardId>(c)))==candidates[i].at("id")) {
                            cards[i]=static_cast<CardId>(c);found=true;break;
                        }
                    if (!found) throw Failure("unmapped_public_offer");
                }
                step.offer.emplace(type,cards,true);++events;
                if (!firstPool) firstPool=step.offer->poolSize;
                // The pool scan and card-name decoding occur once here, not
                // on every proposed world. Each world only copies three IDs.
                validationWork+=step.offer->poolSize+3*int(CardId::ZAP);
                work+=3;
            }
            steps.push_back(std::move(step));previous=observed;
        }
    }

    void charge(Budget& budget) {
        budget.tick(work+(validationCharged?0:validationWork));
        validationCharged=true;
    }

    bool arm(std::size_t& cursor, const std::string& action, trace::MechanismRecorder& recorder) const {
        if (cursor>=steps.size()) return false;
        const auto& step=steps[cursor++];
        if (action!=step.action || recorder.eventOffer) throw Failure("offer_replay_step_contract");
        recorder.eventOffer=step.offer;
        return true;
    }
};
} // namespace spire_offer_replay
