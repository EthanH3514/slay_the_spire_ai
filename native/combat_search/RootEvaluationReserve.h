#pragma once
#include "Engine.h"

namespace spire_tree {
inline constexpr auto reservedVersion="public-history-tree-v6-full-round-reserve";

struct RootEvaluationReserve {
    int rounds=0,planning=0;
    std::int64_t required=0;
    static RootEvaluationReserve calculate(int steps,int candidates,int evalRounds,int horizon) {
        if(steps<1||candidates<2||evalRounds<1||horizon<1||evalRounds>steps/candidates)
            throw std::invalid_argument("root_evaluation_reserve_contract");
        RootEvaluationReserve out;
        out.rounds=evalRounds/2+evalRounds%2;
        // candidates * evalRounds <= steps, so even INT_MAX horizons fit int64.
        out.required=std::int64_t(candidates)*out.rounds*horizon;
        out.planning=int(std::min<std::int64_t>(steps/2,std::max<std::int64_t>(0,steps-out.required)));
        return out;
    }
    J diagnostic() const {
        return {{"version","full-root-round-reserve-v1"},{"minimum_full_rounds",rounds},
            {"worst_case_evaluation_steps",required},{"planning_step_cap",planning},
            {"zero_planning_uses_flat_search",planning==0},
            {"guarantee","step reservation only; sample, reconstruction and wall-time caps still apply"}};
    }
};
} // namespace spire_tree
