#pragma once
#include "ConditionalRoots.h"
#include "HistoryTree.h"
#include "TreeOverrideProbe.h"
#include "PairedContinuation.h"
#include "JointContinuation.h"
#include "FocusedExpansion.h"
#include "RootEvaluationReserve.h"
#include <array>

namespace spire_search {
inline J searchHistoryTree(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                          const J& budget,const J& metadata,bool detail=false,bool guarded=false,
                          const J* overrideProbe=nullptr,bool paired=false) {
    auto started=Clock::now();
    if(paired&&(!guarded||overrideProbe))throw std::invalid_argument("paired_tree_entry_contract");
    const bool aligned=spire_objective::enabled(budget);
    const auto requested=budget.value("tree_algorithm",std::string(paired?spire_tree::pairedVersion:
        guarded?spire_tree::guardedVersion:spire_tree::version));
    const auto construction=spire_tree::ConstructionVariant::parse(requested);
    const bool joint=requested==spire_tree::jointVersion||construction.enabled;
    const bool reserved=requested==spire_tree::reservedVersion||joint;
    const bool efficient=requested==spire_tree::efficientVersion||reserved;
    if(aligned?(!paired||(requested!=spire_tree::objectiveVersion&&!efficient)||overrideProbe):
        (requested==spire_tree::objectiveVersion||efficient))throw std::invalid_argument("objective_tree_entry_contract");
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    const auto probe=spire_tree::OverrideProbe::parse(overrideProbe,detail,guarded,budget.at("max_seconds"));
    for(const auto* k:{"samples_per_action","rollout_horizon","max_branch_steps","max_samples","max_value_evaluations"})
        if(!budget.at(k).is_number_integer()||budget.at(k)<1||budget.at(k)>2147483647)
            throw std::invalid_argument("tree_search_budget");
    const auto& input=roots.evidence();const auto& candidates=input.candidates;
    const int count=int(candidates.size());
    if(count<2)throw std::invalid_argument("tree_requires_two_candidates");
    double seconds=budget.at("max_seconds").is_null()?-1:budget.at("max_seconds").get<double>();
    if(!budget.at("max_seconds").is_null()&&(!std::isfinite(seconds)||seconds<0))
        throw std::invalid_argument("tree_search_deadline");
    const auto until=seconds<0?Clock::time_point::max():started+
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto checkTime=[&](){if(Clock::now()>=until)throw std::runtime_error("tree_deadline");};
    const int evalRounds=std::min({number(budget,"samples_per_action"),number(budget,"max_samples")/count,
        number(budget,"max_value_evaluations")/count,number(budget,"max_branch_steps")/count});
    spire_tree::RootEvaluationReserve reservation;
    if(reserved)reservation=spire_tree::RootEvaluationReserve::calculate(number(budget,"max_branch_steps"),
        count,evalRounds,number(budget,"rollout_horizon"));
    if(seeds.size()!=std::size_t(evalRounds+limits.trainingRollouts+(paired?spire_tree::verificationProposals:0)))
        throw std::invalid_argument("tree_seed_plan_size");
    int samples=0,steps=0,values=0,policies=0,prepared=0,trainingSamples=0,trainingSteps=0,trainingValues=0;
    int treeSteps=0,evaluationTreeSteps=0,ruleLeaves=0,evaluationRuleSteps=0;
    J rounds=J::array(),horizons=J::array(),traces=J::array();
    J firstOverride,probeResult;bool probeActive=false;
    unsigned planningTraces=0,evaluationTraces=0;
    unsigned newNodes=0,maxNewNodes=0,rulePriors=0,evaluationOverrides=0,guardLeaves=0;
    spire_tree::Tree tree(limits);
    std::unique_ptr<spire_tree::ObjectiveStats> objectiveStats;
    if(aligned)objectiveStats=std::make_unique<spire_tree::ObjectiveStats>(number(budget,"max_branch_steps"),spire_objective::name(budget));
    auto* rootNode=tree.root(candidates);
    std::unique_ptr<spire_tree::FocusedRoots> focusedRoots;
    if(construction.focused)focusedRoots=std::make_unique<spire_tree::FocusedRoots>(rootNode);
    std::array<unsigned,4> expansionHistogram{};
    std::unique_ptr<spire_tree::PairedCandidates> proposed;
    std::unique_ptr<spire_tree::JointCandidates> jointCandidates;
    spire_tree::JointPolicy jointPolicy;JointWork jointWork;
    spire_tree::PairedProposal proposal;PairedWork singleWork;unsigned eligible=0;
    PairedWork& verification=joint?static_cast<PairedWork&>(jointWork):singleWork;
    unsigned maxEvaluationChanges=0,multiEvaluationPaths=0;
    verification.objective=aligned;
    verification.objectiveScreen.objectiveVersion=spire_objective::name(budget);
    if(joint){jointCandidates=std::make_unique<spire_tree::JointCandidates>(limits.nodes);jointCandidates->link(rootNode,nullptr,nullptr);}
    else if(paired){proposed=std::make_unique<spire_tree::PairedCandidates>(limits.nodes);proposed->link(rootNode,nullptr,nullptr);}
    auto finish=[&](const std::string& stop,const std::string& refusal="") {
        auto stats=tree.diagnostics(detail);
        if(guarded)stats.update({{"version",spire_tree::guardedVersion},
            {"expansion_policy","one-new-node-then-rule"},{"continuation_policy","repeated-rule-reference-v1"},
            {"minimum_continuation_visits",spire_tree::minimumContinuationVisits},
            {"training_new_nodes",newNodes},{"max_new_nodes_per_trajectory",maxNewNodes},
            {"rule_prior_calls",rulePriors},{"evaluation_override_steps",evaluationOverrides},
            {"evaluation_guard_leaves",guardLeaves}});
        if(paired)stats.update({{"version",spire_tree::pairedVersion},
            {"continuation_policy","single-fixed-batch-paired-rule-improvement-v1"},
            {"verification",joint?jointWork.diagnostic(detail):verification.diagnostic(detail)},{"proposal",proposal.diagnostic(detail)},
            {"eligible_proposals",eligible},{"verification_link_bytes",joint?jointCandidates->bytes():proposed->bytes()},
            {"verification_steps",verification.steps},{"verification_samples",verification.samples},
            {"verification_values",verification.values},{"verification_policies",verification.policies}});
        if(aligned)stats.update({{"version",requested},{"combat_objective",objectiveStats->diagnostics()},
            {"minimum_nomination_visits",efficient?1:2},{"construction_fraction",efficient?"3/4":"3/8"},
            {"continuation_policy","single-objective-paired-rule-improvement-v1"}});
        if(reserved)stats["root_evaluation_reserve"]=reservation.diagnostic();
        if(joint)stats.update({{"continuation_policy","single-carrier-joint-conditional-policy-v1"},
            {"joint_policy",jointPolicy.diagnostic(detail)},
            {"max_evaluation_distinct_overrides",maxEvaluationChanges},
            {"evaluation_multi_node_trajectories",multiEvaluationPaths}});
        if(construction.enabled)stats.update({
            {"expansion_policy",construction.newNodeCap==3?"three-new-nodes-then-rule":"one-new-node-then-rule"},
            {"construction",{{"new_node_cap",construction.newNodeCap},
                {"completed_creation_histogram",expansionHistogram},{"depths",spire_tree::constructionDepths(rootNode)},
                {"root_allocation",focusedRoots?"two-pass-top-two-with-quarter-exploration":"guarded-UCT"},
                {"focus",focusedRoots?focusedRoots->diagnostic():J()}}}});
        stats.update({{"training_samples",trainingSamples},{"training_steps",trainingSteps},
            {"training_values",trainingValues},{"evaluation_samples",samples-trainingSamples},
            {"evaluation_steps",steps-trainingSteps},{"evaluation_values",values-trainingValues},
            {"tree_policy_steps",treeSteps},{"evaluation_tree_steps",evaluationTreeSteps},
            {"evaluation_rule_steps",evaluationRuleSteps},{"rule_leaf_trajectories",ruleLeaves},
            {"root_plans",prepared},{"evaluation_rounds",rounds.size()},
            {"training_horizon",std::min(limits.trainingHorizon,number(budget,"rollout_horizon"))}});
        bool failed=!refusal.empty();
        const int totalSamples=samples+verification.samples;
        J result={{"stop",stop},{"refusal",failed?J(refusal):J(nullptr)},{"unavailable",""},
            {"samples",totalSamples},{"steps",steps+verification.steps},{"values",values+verification.values},
            {"policies",policies+verification.policies},
            {"discarded",failed?totalSamples:0},{"peak",totalSamples?1:0},
            {"rounds",failed?J::array():rounds},{"horizons",failed?J::array():horizons},
            {"traces",failed?J::array():traces},{"tree",stats},{"root_sampling",roots.diagnostics()}};
        if(detail&&guarded)result["first_override"]=firstOverride;
        if(probe.enabled)result["override_probe"]=probeResult;
        return result;
    };
    try {
        checkTime();roots.validate(input.policy,candidates,metadata);
        if(roots.certificate(budget,metadata,seeds).size()>16*1024*1024)
            throw std::runtime_error("tree_root_certificate_limit");
        if(reserved&&reservation.planning==0) {
            // No game-specific whitelist or rule fallback: execute the EXACT
            // existing flat evaluator with its full total budget/common seeds.
            tree.freeze();std::vector<RootSeeds> evalSeeds(seeds.begin(),seeds.begin()+evalRounds);
            // Keep the ORIGINAL request clock. A nested searchConditioned call
            // would restart the deadline after validation/certificate work.
            auto flat=flatSearch(roots,input.policy,input.relics,input.character,evalSeeds,candidates,
                budget,metadata,detail,started,input.actionVersion=="native-public-occurrences-v1");
            flat["root_sampling"]=roots.diagnostics();
            samples=flat.at("samples");steps=flat.at("steps");values=flat.at("values");policies=flat.at("policies");
            evaluationRuleSteps=policies;prepared=roots.diagnostics().at("prepared_rounds");
            rounds=flat.at("rounds");horizons=flat.at("horizons");
            flat["tree"]=finish(flat.at("stop"))["tree"];
            return flat;
        }
        auto prepare=[&](const RootSeeds& seed){checkTime();roots.prepare(prepared,seed,until);++prepared;};
        auto trajectory=[&](bool learn,int horizon,const J* first,int round=-1,int candidate=-1) {
            checkTime();++samples;if(learn)++trainingSamples;
            auto world=roots.branch(prepared-1);
            auto project=[&]{return roots.projectView(*world.battle,input.relics,world.knownTop,input.character);};
            auto view=project();auto legal=actions(*world.battle,true);
            if(view!=input.policy||!roots.sameActions(legal,candidates))throw std::runtime_error("tree_root_changed");
            auto* node=rootNode;
            spire_tree::JointUsage jointUsage;
            const bool useJoint=joint&&!learn&&jointPolicy.first&&verification.approved()&&first&&
                *first==J::parse(jointPolicy.first->action);
            std::array<std::pair<spire_tree::Node*,spire_tree::Edge*>,64> path{};unsigned pathSize=0;
            bool usedRule=false;J traceSteps=J::array();
            int probeDepth=-1,probeStepsBefore=0,probePoliciesBefore=0;
            auto row=[&](double value){const int terminal=outcome(*world.battle);
                if(aligned)return spire_objective::record(view,input.policy,terminal,value,spire_objective::name(budget));
                return J{{"value",value},{"hp",world.battle->player.curHp},{"outcome",terminal},
                    {"resource_hp",world.battle->player.curHp+inventoryReserve(view)-inventoryReserve(input.policy)}};};
            unsigned trajectoryNewNodes=0;
            // Separate diagnostic caps: planning must not hide ALL evaluation
            // traces. No change to tree policy, seeds, backup or counted work.
            const bool captureTrace=detail&&(learn?planningTraces<16:evaluationTraces<32);
            for(int depth=0;depth<(probeActive?probeDepth+probe.suffixSteps:horizon);++depth) {
                checkTime();tree.check(node,legal);
                const bool forced=depth==0&&first;
                J prior;
                if(guarded&&node&&!forced){
                    ++policies;++rulePriors;prior=combatChoice(view,legal,metadata);
                    if(paired&&learn){if(joint)jointCandidates->prior(tree,node,prior);else proposed->prior(tree,node,prior);}
                }
                auto* edge=forced?tree.edge(node,*first):
                    (paired&&!learn?(joint?(useJoint?jointPolicy.select(node):nullptr):
                        (node&&node==proposal.node&&verification.approved()?proposal.challenger:nullptr)):
                        (guarded?tree.selectGuarded(node,learn,prior):tree.select(node,learn)));
                J action;
                if(forced)action=*first;
                else if(edge)action=J::parse(edge->action);
                else {
                    usedRule=true;
                    if(!prior.is_null())action=prior;
                    else {++policies;action=combatChoice(view,legal,metadata);}
                    if(!learn){++evaluationRuleSteps;if(guarded&&node)++guardLeaves;}
                }
                if(guarded&&!learn&&!forced&&edge&&action!=prior){
                    ++evaluationOverrides;
                    if(joint)jointUsage.add(node);
                    if(detail&&firstOverride.is_null()){
                        // Independent of the ordinary trace caps: always retain the FIRST
                        // override in round/candidate/depth order, never chosen by outcome.
                        firstOverride={{"round",round},{"candidate",candidate},{"depth",depth},
                            {"public",view},{"action",action},{"rule_prior",prior},
                            {"support",tree.decisionDiagnostic(node)},{"prefix",traceSteps}};
                        if(probe.enabled){
                            if(firstOverride!=probe.expected)throw std::runtime_error("override_probe_target_changed");
                            if(probe.suffixSteps<horizon-depth)throw std::runtime_error("override_probe_horizon_too_short");
                            probeActive=true;probeDepth=depth;probeStepsBefore=steps;probePoliciesBefore=policies;
                            probeResult={{"version",spire_tree::overrideProbeVersion},{"target",firstOverride},
                                {"first_action",probe.useRule?"rule":"accepted"},
                                {"continuation",probe.ruleSuffix?"rule":"frozen"},
                                {"original_horizon",horizon},{"max_suffix_steps",probe.suffixSteps}};
                            if(probe.useRule){action=prior;edge=tree.edge(node,action);}
                        }
                    }
                }
                if(edge){++treeSteps;if(!learn)++evaluationTreeSteps;
                    if(learn){if(pathSize==path.size())throw std::runtime_error("tree_path_limit");path[pathSize++]={node,edge};}}
                if(captureTrace||probeActive||(detail&&guarded&&!learn&&firstOverride.is_null())){
                    traceSteps.push_back({{"depth",depth},{"node",node?J(node->id):J(nullptr)},
                    {"action",action},{"tree_policy",bool(edge)},{"public",view},
                    {"forced_first",forced},{"support",tree.decisionDiagnostic(node)}});
                    if(guarded)traceSteps.back()["rule_prior"]=prior;
                }
                ++steps;if(learn)++trainingSteps;
                const auto observed=roots.advancePublic(world,action);view=project();checkTime();
                // Verify byte limits even at terminal/horizon leaves.
                auto key=spire_tree::observationKey(observed,limits.observationBytes);
                if(probeActive&&(depth+1==horizon||outcome(*world.battle))&&!probeResult.contains("at_original_horizon")){
                    probeResult["at_original_horizon"]=row(combatValue(view,outcome(*world.battle)));
                    probeResult["horizon_public"]=view;
                }
                if(outcome(*world.battle))break;
                guard(view,metadata);
                if(depth+1==(probeActive?probeDepth+probe.suffixSteps:horizon))break;
                legal=actions(*world.battle,true);
                bool created=false;
                auto* parent=node;
                // Even rule-reference actions must traverse known histories in v3;
                // otherwise the approved deeper node would never be reachable.
                auto* via=paired&&!learn&&parent?tree.edge(parent,action):edge;
                node=probeActive&&probe.ruleSuffix?nullptr:tree.child(parent,via,key,legal,learn,&created);
                if(paired&&learn&&created){if(joint)jointCandidates->link(node,parent,via);else proposed->link(node,parent,via);}
                if(guarded&&created){
                    ++trajectoryNewNodes;++newNodes;maxNewNodes=std::max(maxNewNodes,trajectoryNewNodes);
                    // The cap-th created node is a retained, unvisited frontier.
                    // v8 can immediately select actions in the first two new
                    // nodes; all older versions retain their one-node behavior.
                    if(trajectoryNewNodes>=construction.newNodeCap)node=nullptr;
                }
            }
            ++values;if(learn)++trainingValues;
            const int terminal=outcome(*world.battle);const double value=combatValue(view,terminal);
            if(!std::isfinite(value))throw std::runtime_error("tree_nonfinite_value");
            checkTime();
            const auto resultRow=row(value);
            if(learn){const double utility=std::clamp((value+3.)/6.,0.,1.);
                for(unsigned i=0;i<pathSize;++i){tree.backup(path[i].first,path[i].second,utility);
                    if(aligned)objectiveStats->add(path[i].second,resultRow);}}
            ruleLeaves+=usedRule;
            if(construction.enabled&&learn)++expansionHistogram.at(trajectoryNewNodes);
            if(joint&&!learn){maxEvaluationChanges=std::max(maxEvaluationChanges,jointUsage.count);
                multiEvaluationPaths+=jointUsage.count>=2;}
            if(captureTrace){++(learn?planningTraces:evaluationTraces);
                traces.push_back({{"phase",learn?"planning":"evaluation"},{"diagnostic_version","tree-decision-trace-v1"},
                    {"steps",traceSteps},{"final",view},{"outcome",terminal},{"horizon",horizon}});
                if(guarded)traces.back().update({{"new_nodes",trajectoryNewNodes},{"tree_path_length",pathSize}});
            }
            if(probeActive)probeResult.update({{"final",row(value)},{"final_public",view},{"steps",traceSteps},
                {"suffix_steps",steps-probeStepsBefore},{"suffix_policy_calls",policies-probePoliciesBefore},
                {"suffix_value_evaluations",2},
                {"prefix_search_steps",probeStepsBefore},{"prefix_policy_calls",probePoliciesBefore},
                {"termination",terminal?"terminal":"suffix_step_limit"}});
            return resultRow;
        };
        // Separate seed namespaces: first evalRounds seeds match flat's paired
        // evaluation worlds; the suffix belongs exclusively to tree learning.
        const int reservedOutcomes=count*evalRounds;
        const int trainingLimit=reserved?reservation.planning:
            std::min(number(budget,"max_branch_steps")/2,number(budget,"max_branch_steps")-reservedOutcomes);
        const int constructionLimit=paired?int(std::int64_t(trainingLimit)*3/(efficient?4:8)):trainingLimit;
        const int trainingHorizon=std::min(limits.trainingHorizon,number(budget,"rollout_horizon"));
        for(int i=0;i<limits.trainingRollouts;++i) {
            if(constructionLimit-trainingSteps<trainingHorizon||number(budget,"max_samples")-samples<=reservedOutcomes||
               number(budget,"max_value_evaluations")-values<=reservedOutcomes)break;
            // Choose the carrier BEFORE constructing this latent sampled world.
            // No future draw/PRNG information is available to the allocator.
            J first=focusedRoots?J::parse(focusedRoots->next(*objectiveStats)->action):J();
            prepare(seeds[evalRounds+i]);trajectory(true,trainingHorizon,first.is_null()?nullptr:&first);
        }
        tree.freeze();
        if(joint) {
            jointPolicy=jointCandidates->propose(tree,*objectiveStats,trainingHorizon,eligible);checkTime();
            verifyJointContinuation(roots,seeds,evalRounds+limits.trainingRollouts,prepared,tree,jointPolicy,
                trainingHorizon,trainingLimit-trainingSteps,number(budget,"max_samples")-samples-reservedOutcomes,
                number(budget,"max_value_evaluations")-values-reservedOutcomes,metadata,until,jointWork,limits.observationBytes);
        } else if(paired) {
            proposal=proposed->propose(tree,trainingHorizon,eligible,objectiveStats.get(),efficient?1:2);
            verifyContinuation(roots,seeds,evalRounds+limits.trainingRollouts,prepared,proposal,trainingHorizon,
                trainingLimit-trainingSteps,number(budget,"max_samples")-samples-reservedOutcomes,
                number(budget,"max_value_evaluations")-values-reservedOutcomes,metadata,until,verification,
                spire_tree::verificationPairs,limits.observationBytes);
        }
        for(int r=0;r<evalRounds;++r) {
            int horizon=std::min(number(budget,"rollout_horizon"),(number(budget,"max_branch_steps")-steps-verification.steps)/count);
            if(horizon<1)return finish("step_budget");
            if(number(budget,"max_samples")-samples-verification.samples<count)return finish("sample_budget");
            if(number(budget,"max_value_evaluations")-values-verification.values<count)return finish("value_budget");
            prepare(seeds[r]);J rows=J::array();
            int candidateIndex=0;
            for(const auto& candidate:candidates){
                rows.push_back(trajectory(false,horizon,&candidate,r,candidateIndex++));
                // A probe is an independent replay, not a search result. No changed
                // suffix is backed up or allowed to change later horizons/ranking.
                if(probeActive)return finish("diagnostic_probe");
            }
            rounds.push_back(rows);horizons.push_back(horizon);
        }
        if(probe.enabled)return finish("tree_refused","override_probe_target_not_reached");
        return finish("requested_samples",rounds.empty()?"no_complete_round":"");
    } catch(const std::exception& e) {
        return finish(std::string(e.what())=="tree_deadline"?"deadline":"tree_refused",e.what());
    }
}
} // namespace spire_search
