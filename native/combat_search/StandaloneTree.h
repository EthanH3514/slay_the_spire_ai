#pragma once
#include "ConditionalRoots.h"
#include "HistoryTree.h"
#include "ExplorationProfile.h"
#include "BranchingProfile.h"
#include "CombatObjective.h"
#include "ObjectiveAudit.h"
#include "DeepPrior.h"
#include "ColdLeaf.h"
#include "NeuralPrior.h"
#include "ValueLeaf.h"
#include "PairedValueAudit.h"
#include "RootRecency.h"
#include "RootRecommendation.h"
#include "TerminalProof.h"
#include <array>
#include <map>
#include <memory>

namespace spire_lazy_rule {inline thread_local bool enabled=false;}

namespace spire_tree {
inline constexpr auto standaloneVersion="public-history-tree-v9-standalone";
inline constexpr auto deepQuarterVersion="public-history-tree-v12-deep-uct-quarter";
inline constexpr auto scoreOrderVersion="public-history-tree-v13-score-ordered";
inline constexpr auto rootObjectiveVersion="public-history-tree-v14-root-objective";

// Public histories/statistics only; no simulator, sampled state or RNG interface.
class StandaloneTree {
    Tree tree_;
    Node* root_;
    std::map<const Edge*,std::vector<spire_objective::Outcome>> rows_;
    bool distributionMode_;
    unsigned predicted_=0,distributionAtoms_=0;
    std::map<const Edge*,std::vector<spire_value_distribution::Distribution>> distributions_;
    unsigned cap_,completed_=0,backups_=0,created_=0,maxCreated_=0;
    std::unique_ptr<ExplorationProfile> profile_;
    std::unique_ptr<ObjectiveAudit> objectiveAudit_;
    bool deepQuarter_;
    bool branching_;
    bool scoreOrder_;
    bool rootObjective_,deepPrior_,rootRecency_,frozen_=false;
public:
    using Path=std::vector<std::pair<Node*,Edge*>>;
    StandaloneTree(Limits limits,const J& legal,unsigned cap,bool profile=false,bool deepQuarter=false,bool branching=false,bool scoreOrder=false,bool objectiveAudit=false,bool rootObjective=false,bool deepPrior=false,bool rootRecency=false,bool distributionMode=false)
        :tree_(limits),root_(tree_.root(legal)),distributionMode_(distributionMode),cap_(cap),
         profile_(profile?std::make_unique<ExplorationProfile>():nullptr),deepQuarter_(deepQuarter),branching_(branching),scoreOrder_(scoreOrder),rootObjective_(rootObjective),deepPrior_(deepPrior),rootRecency_(rootRecency) {
        if(int(deepQuarter_)+int(scoreOrder_)+int(rootObjective_)+int(deepPrior_)+int(rootRecency_)>1)throw std::invalid_argument("standalone_single_factor");
        if(rootRecency_&&cap>256)throw std::invalid_argument("root_recency_cap");
        if(!root_)throw std::runtime_error("standalone_root_arena");
        if(objectiveAudit){
            if(!profile||!branching)throw std::invalid_argument("objective_audit_requires_profiles");
            objectiveAudit_=std::make_unique<ObjectiveAudit>(cap,limits.depth);
        }
        for(auto* e=root_->edges;e;e=e->next){rows_.emplace(e,std::vector<spire_objective::Outcome>{});distributions_[e]={};}
    }
    Node* root()const{return root_;}
    Edge* rootAction()const{
        if(!rootObjective_)return tree_.select(root_,true);
        spire_perf::Scope timing(spire_perf::Phase::Tree);
        if(frozen_)throw std::runtime_error("tree_frozen_mutation");
        Edge* least=root_->edges;
        for(auto* e=least->next;e;e=e->next)if(e->visits<least->visits)least=e;
        // Selection is read-only. Only a completed backup advances the schedule.
        if(least->visits<2||(completed_-2*rows_.size())%2==1)return least;
        return recommend();
    }
    bool needsScores(const Node* n)const{
        if(!scoreOrder_||!n||!n->depth||!n->visits)return false;
        for(auto* e=n->edges;e;e=e->next)if(!e->visits)return true;
        return false;
    }
    Edge* select(Node* n,const J& legal,const J& prior,const std::vector<double>* scores=nullptr)const{
        if(needsScores(n)){
            if(!scores)throw std::invalid_argument("score_order_missing_scores");
            return tree_.selectScoreOrdered(n,legal,prior,*scores);
        }
        tree_.check(n,legal);
        if(deepPrior_&&n&&n->depth){
            spire_perf::Scope timing(spire_perf::Phase::Tree);
            if(frozen_)throw std::runtime_error("tree_frozen_mutation");
            return deepPriorChoice(n,tree_.edge(n,prior));
        }
        return deepQuarter_&&n&&n->depth?tree_.selectDeepQuarter(n,prior):tree_.selectGuarded(n,true,prior);
    }
    void checkRoot(const J& legal)const{tree_.check(root_,legal);}
    Node* descend(Node* parent,Edge* edge,const std::string& observed,const J& legal,unsigned& newNodes){
        bool made=false;auto* child=tree_.child(parent,edge,observed,legal,true,&made);
        if(made){++created_;++newNodes;maxCreated_=std::max(maxCreated_,newNodes);}
        return newNodes>=3?nullptr:child;
    }
    void backup(const Path& path,const spire_objective::Outcome& row){
        if(path.empty()||path.front().first!=root_||completed_>=cap_)
            throw std::runtime_error("standalone_backup_contract");
        auto found=rows_.find(path.front().second);
        if(found==rows_.end()||!std::isfinite(row.value)||!std::isfinite(row.hp)||
           !std::isfinite(row.resource)||row.terminal<0||row.terminal>3)
            throw std::runtime_error("standalone_outcome_contract");
        const double utility=std::clamp((row.value+3.)/6.,0.,1.);
        if(objectiveAudit_)objectiveAudit_->completed(path,row);
        found->second.push_back(row);
        for(const auto& pair:path){
            if(profile_)profile_->beforeBackup(pair.first,pair.second);
            tree_.backup(pair.first,pair.second,utility);++backups_;
        }
        ++completed_;
    }
    void backupDistribution(const Path& path,const spire_value_distribution::Distribution& row){
        if(!distributionMode_||path.empty()||path.front().first!=root_||completed_>=cap_||cap_>spire_boundary_value::limit())
            throw std::runtime_error("distribution_backup_contract");
        row.check();if(distributionAtoms_+row.atoms.size()>spire_boundary_value::atoms())throw std::runtime_error("distribution_atom_limit");
        auto found=distributions_.find(path.front().second);if(found==distributions_.end())throw std::runtime_error("distribution_root_edge");
        found->second.push_back(row);distributionAtoms_+=row.atoms.size();predicted_+=row.predicted;
        const double utility=spire_resource_utility::backup(row.meanValue());
        for(const auto& pair:path){tree_.backup(pair.first,pair.second,utility);++backups_;}++completed_;
    }
    unsigned completed()const{return completed_;}
    unsigned covered()const{unsigned n=0;for(const auto& pair:rows_)n+=distributionMode_?!distributions_.at(pair.first).empty():!pair.second.empty();return n;}
    Edge* recommend(bool visitsFirst=false)const{
        if(covered()!=rows_.size())return nullptr;
        Edge* best=nullptr;spire_objective::Key key{};
        for(auto* e=root_->edges;e;e=e->next){auto k=distributionMode_?spire_value_distribution::summarize(distributions_.at(e)):(rootRecency_?recencyKey(rows_.at(e)):spire_objective::summarize(rows_.at(e)));
            if(!best||spire_root_recommendation::better(visitsFirst,e->visits,k,best->visits,key)){best=e;key=k;}}
        return best;
    }
    J evidence()const{
        J result=J::array();
        if(distributionMode_){
            for(auto* e=root_->edges;e;e=e->next){const auto& rows=distributions_.at(e);J outcomes=J::array();
                for(const auto& r:rows)outcomes.push_back(r.evidence);
                result.push_back({{"action",J::parse(e->action)},{"visits",e->visits},{"distributions",outcomes},
                    {"distribution_key",rows.empty()?J():J(spire_value_distribution::summarize(rows))}});
            }return result;
        }
        for(auto* e=root_->edges;e;e=e->next){const auto& rows=rows_.at(e);J outcomes=J::array();
            for(const auto& r:rows)outcomes.push_back({{"value",r.value},{"hp",r.hp},
                {"resource_hp",r.resource},{"outcome",r.terminal}});
            result.push_back({{"action",J::parse(e->action)},{"visits",e->visits},{"outcomes",outcomes},
                {"objective_key",rows.empty()?J():J(spire_objective::summarize(rows))}});
            if(rootRecency_)result.back()["recommendation_estimate"]=recencyEvidence(rows);
        }return result;
    }
    void freeze(){frozen_=true;tree_.freeze();}
    J diagnostic(bool detail)const{
        auto d=tree_.diagnostics(detail);
        d.update({{"version",standaloneVersion},{"completed_simulations",completed_},
            {"covered_root_actions",covered()},{"root_action_count",rows_.size()},
            {"backed_up_edges",backups_},{"new_nodes",created_},{"max_new_nodes_per_simulation",maxCreated_},
            {"new_node_cap",3},{"exploration","public-rule-prior-then-scalar-UCT"},
            {"root_exploration","canonical-unvisited-then-scalar-UCT-before-world"},
            {"recommendation","empirical-adaptive-mixture-distributional-objective"},
            {"tie_breaker","canonical-action-json"},{"calibrated_confidence",false},
            {"flat_delegations",0},{"verification_simulations",0}});
        if(distributionMode_)d.update({{"recommendation",spire_value_distribution::version},
            {"predicted_simulations",predicted_},{"observed_terminal_simulations",completed_-predicted_},
            {"distribution_atoms",distributionAtoms_},{"distribution_atom_limit",spire_boundary_value::atoms()}});
        if(profile_)d["exploration_profile"]=profile_->result(tree_);
        if(branching_)d["branching_profile"]=branchingProfile(tree_);
        if(objectiveAudit_)d["objective_audit"]=objectiveAudit_->result();
        if(deepQuarter_)d.update({{"version",deepQuarterVersion},{"deep_uct_bonus_scale",.25},
            {"root_uct_bonus_scale",1.},{"exploration","public-rule-prior-then-quarter-bonus-UCT"}});
        if(scoreOrder_)d.update({{"version",scoreOrderVersion},{"deep_uct_bonus_scale",1.},
            {"root_uct_bonus_scale",1.},{"exploration","public-rule-prior-then-score-ordered-unvisited-then-scalar-UCT"},
            {"unvisited_order","public-rule-score-desc-canonical-ties"},{"mandatory_action_coverage",true}});
        if(rootObjective_){
            const auto warm=std::min<std::size_t>(completed_,2*rows_.size()),post=completed_-warm;
            d.update({{"version",rootObjectiveVersion},{"deep_uct_bonus_scale",1.},
                {"root_exploration","two-pass-then-objective-least-visits-alternating-before-world"},
                {"root_warmup_min_visits",2},{"root_warmup_completed",warm},
                {"root_objective_completed",(post+1)/2},{"root_exploration_completed",post/2},
                {"root_allocation_confidence_bound",false}});
        }
        if(deepPrior_)d.update({{"version",deepPriorVersion},{"exploration",deepPriorSelection},
            {"root_uct_bonus_scale",1.},{"deep_puct_c",1.},{"deep_prior_rule_mass",.5},
            {"unvisited_value","parent-empirical-mean-selection-only"},
            {"mandatory_deep_action_coverage",false},{"stored_pseudo_visits",0}});
        if(rootRecency_)d.update({{"version",rootRecencyVersion},{"recommendation",rootRecencyEstimator},
            {"recency_scope","root-recommendation-only"},{"recency_simulation_cap",256},{"weighted_samples_are_rollouts",false}});
        return d;
    }
};
} // namespace spire_tree

namespace spire_search {
template<bool DeepQuarter=false,bool ScoreOrder=false,bool RootObjective=false,bool DeepPrior=false,bool RootRecency=false>
inline J searchStandaloneTree(ConditionalRoots& roots,const std::vector<RootSeeds>& seeds,
                              const J& budget,const J& metadata,bool detail=false,
                              const std::function<void(const spire_tree::StandaloneTree&,int,bool)>& observer={}) {
    const auto started=Clock::now();
    static_assert(int(DeepQuarter)+int(ScoreOrder)+int(RootObjective)+int(DeepPrior)+int(RootRecency)<=1,"one search factor only");
    constexpr auto algorithm=RootRecency?spire_tree::rootRecencyVersion:DeepPrior?spire_tree::deepPriorVersion:RootObjective?spire_tree::rootObjectiveVersion:ScoreOrder?spire_tree::scoreOrderVersion:DeepQuarter?spire_tree::deepQuarterVersion:spire_tree::standaloneVersion;
    if(budget.value("tree_algorithm","")!=algorithm||!spire_objective::enabled(budget))
        throw std::invalid_argument("standalone_version_contract");
    if constexpr(DeepQuarter){
        for(const auto* k:{"root_coupling","frozen_audit","deep_uct_bonus_scale","root_uct_bonus_scale"})
            if(budget.contains(k))throw std::invalid_argument("deep_quarter_fixed_factor_contract");
    }
    if constexpr(ScoreOrder){
        for(const auto* k:{"root_coupling","frozen_audit","deep_uct_bonus_scale","root_uct_bonus_scale","opening_width","score_weights"})
            if(budget.contains(k))throw std::invalid_argument("score_order_fixed_factor_contract");
    }
    if constexpr(RootObjective){
        for(const auto* k:{"root_coupling","frozen_audit","deep_uct_bonus_scale","root_uct_bonus_scale",
                "opening_width","score_weights","root_warmup_min_visits","root_exploration_fraction"})
            if(budget.contains(k))throw std::invalid_argument("root_objective_fixed_factor_contract");
    }
    if constexpr(DeepPrior){
        for(const auto* k:{"root_coupling","frozen_audit","deep_uct_bonus_scale","root_uct_bonus_scale",
                "opening_width","score_weights","root_warmup_min_visits","root_exploration_fraction",
                "deep_puct_c","deep_prior_rule_mass","unvisited_value"})
            if(budget.contains(k))throw std::invalid_argument("deep_prior_fixed_factor_contract");
    }
    if constexpr(RootRecency){
        for(const auto* k:{"root_coupling","frozen_audit","deep_uct_bonus_scale","root_uct_bonus_scale",
                "opening_width","score_weights","root_warmup_min_visits","root_exploration_fraction",
                "deep_puct_c","deep_prior_rule_mass","unvisited_value","recency_weights","recency_decay"})
            if(budget.contains(k))throw std::invalid_argument("root_recency_fixed_factor_contract");
    }
    const auto limits=spire_tree::Limits::parse(budget.at("tree"));
    for(const auto* k:{"samples_per_action","rollout_horizon","max_branch_steps","max_samples","max_value_evaluations"})
        if(!budget.at(k).is_number_integer()||budget.at(k)<1||budget.at(k)>2147483647)
            throw std::invalid_argument("standalone_search_budget");
    const auto& input=roots.evidence();const auto& legalRoot=input.candidates;
    if(legalRoot.size()<2||legalRoot.size()>2147483647u)throw std::invalid_argument("standalone_requires_choices");
    const int count=int(legalRoot.size()),stepCap=number(budget,"max_branch_steps");
    const int simulationCap=std::min({limits.trainingRollouts,number(budget,"max_samples"),
        number(budget,"max_value_evaluations"),stepCap});
    const int horizon=std::min(number(budget,"rollout_horizon"),stepCap/count);
    if(seeds.size()!=std::size_t(simulationCap))throw std::invalid_argument("standalone_seed_plan_size");
    double seconds=budget.at("max_seconds").is_null()?-1:budget.at("max_seconds").get<double>();
    if(!budget.at("max_seconds").is_null()&&(!std::isfinite(seconds)||seconds<0))
        throw std::invalid_argument("standalone_deadline_contract");
    const auto until=seconds<0?Clock::time_point::max():started+
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto checkTime=[&](){if(Clock::now()>=until)throw std::runtime_error("standalone_deadline");};
    if(budget.contains("exploration_profile")&&!budget.at("exploration_profile").is_boolean())
        throw std::invalid_argument("exploration_profile_boolean");
    if(budget.contains("branching_profile")&&(!budget.at("branching_profile").is_boolean()||
        (budget.at("branching_profile")==true&&!budget.value("exploration_profile",false))))
        throw std::invalid_argument("branching_profile_requires_exploration_profile");
    if(budget.contains("objective_audit")&&(!budget.at("objective_audit").is_boolean()||
        (budget.at("objective_audit")==true&&(!budget.at("max_seconds").is_null()||
          !budget.value("exploration_profile",false)||!budget.value("branching_profile",false)))))
        throw std::invalid_argument("objective_audit_requires_untimed_profiles");
    spire_resource_utility::Scope resourceUtility(budget);
    spire_boundary_value::Scope boundaryScope(budget);
    const spire_root_recommendation::Config rootRecommendation(budget);
    spire_boundary_value::Counts boundaryCounts;
    spire_allocation_probe::Scope allocationProbe(budget);
    spire_neural_prior::Prior neural(budget);
    spire_value_resource_bound::Scope resourceBound(budget,input.policy,metadata);
    spire_value_leaf::Evaluator evaluator(budget);
    spire_terminal_proof::Collector terminalProof(budget,input.policy,metadata,neural.root());
    spire_puct_exploration::Scope explorationScope(budget);
    spire_value_audit::Audit audit(budget);
    spire_tree::StandaloneTree tree(limits,legalRoot,simulationCap,budget.value("exploration_profile",false),DeepQuarter,
        budget.value("branching_profile",false),ScoreOrder,budget.value("objective_audit",false),RootObjective,DeepPrior,RootRecency,evaluator.enabled());
    const std::string actorMode=budget.value("leaf_actor",std::string());
    spire_actor::Counts actorCounts;
    int samples=0,steps=0,values=0,policies=0,prepared=0,treeSteps=0,leafSteps=0,replacementSteps=0;
    std::uint64_t orderingBatches=0,orderingScores=0,extraDiscardScores=0;
    J traces=J::array();
    auto finish=[&](const std::string& stop,const std::string& error=""){
        tree.freeze();auto* best=error.empty()?tree.recommend(rootRecommendation.visits):nullptr;
        std::string refusal=error;
        if(!best&&refusal.empty())refusal="standalone_incomplete_root_coverage";
        if(observer&&best)observer(tree,steps,true);
        auto stats=tree.diagnostic(detail);
        if(rootRecommendation.enabled)stats["root_recommendation"]=rootRecommendation.receipt();
        if(rootRecommendation.visits)stats["recommendation"]=spire_root_recommendation::visitsObjective;
        if(boundaryScope.config.enabled)stats["boundary_value"]=boundaryCounts.json(boundaryScope.config,roots.failureCode());
        if(neural.enabled())stats["neural_prior"]=neural.diagnostics();
        if(allocationProbe.enabled)stats["allocation_probe"]=allocationProbe.receipt();
        if(evaluator.enabled())stats["value_leaf"]=evaluator.diagnostic();
        if(evaluator.collection().enabled())stats["frontier_collection"]=evaluator.collection().report();
        if(neural.changes()){stats["exploration"]="positive-public-prior-PUCT";stats["root_exploration"]="canonical-coverage-then-PUCT-before-world";}
        if(!actorMode.empty()){stats["actor_mode"]=actorMode;stats["actor_counts"]=actorCounts.json();}
        if constexpr(ScoreOrder)stats.update({{"ordering_batches",orderingBatches},
            {"ordering_score_slots",orderingScores},{"ordering_extra_discard_scores",extraDiscardScores}});
        stats.update({{"simulation_horizon",horizon},{"simulation_cap",simulationCap},
            {"tree_policy_steps",treeSteps},{"rule_leaf_steps",leafSteps},{"rule_calls",policies},
            {"interrupted_simulations",samples-int(tree.completed())},{"unused_steps",stepCap-steps},
            {"ignored_legacy_allocation_fields",{"samples_per_action","tree.training_horizon"}}});
        J answer={{"contract",algorithm},{"action",best?J::parse(best->action):J()},
            {"stop",stop},{"refusal",refusal.empty()?J():J(refusal)},{"unavailable",""},
            {"samples",samples},{"steps",steps},{"values",values},{"policies",policies},
            {"discarded",best?samples-int(tree.completed()):samples},{"peak",samples?1:0},
            {"root_evidence",tree.evidence()},{"tree",stats},{"traces",traces},
            {"root_sampling",roots.diagnostics()}};
        if(spire_potion_replacement::enabled)answer["potion_replacement"]={
            {"version",spire_potion_replacement::version},{"menus",spire_potion_replacement::menus},
            {"excluded_action_occurrences",spire_potion_replacement::excluded},
            {"completed_replacements",replacementSteps},{"additional_primitive_steps",replacementSteps},
            {"decision_steps",steps-replacementSteps},{"primitive_steps",steps},{"primitive_step_cap",stepCap},
            {"evaluated_between_actions",false}};
        if(resourceUtility.enabled)answer["resource_utility"]=spire_resource_utility::version;
        if(resourceBound.enabled())answer["value_resource_bound"]=resourceBound.diagnostic();
        if(terminalProof.enabled){
            auto proof=terminalProof.finish(answer["action"],legalRoot,bool(best));
            answer["action"]=proof.at("selected_action");answer["terminal_proof"]=std::move(proof);
        }
        if(audit.enabled()&&best)answer["paired_value_audit"]=audit.finish(roots,metadata);
        return answer;
    };
    try {
        checkTime();roots.validate(input.policy,legalRoot,metadata);
        if(roots.certificate(budget,metadata,seeds).size()>16*1024*1024)
            throw std::runtime_error("standalone_certificate_limit");
        if(!horizon||simulationCap<count)return finish("coverage_budget");
        while(samples<simulationCap){
            checkTime();if(stepCap-steps<horizon*(spire_potion_replacement::enabled?2:1))return finish("step_budget");
            // Scheduling must not receive the next sampled latent world.
            auto* suggested=neural.enabled()?neural.select(tree.root(),input.policy,legalRoot,neural.root()):nullptr;
            auto* first=suggested?suggested:tree.rootAction();
            const int index=samples++;evaluator.collection().begin(index);
            roots.prepare(prepared,seeds.at(index),until);++prepared;
            auto world=roots.branch(prepared-1);
            auto project=[&]{return roots.projectView(*world.battle,input.relics,world.knownTop,input.character);};
            auto view=project();checkTime();
            spire_rich::Memory richMemory;J rich;
            if(neural.enabled()){richMemory=neural.start(*world.battle);rich=neural.root();}
            if(view!=input.policy)throw std::runtime_error("standalone_root_projection");
            auto legal=actions(*world.battle,true);tree.checkRoot(legal);
            spire_tree::StandaloneTree::Path path;path.reserve(limits.depth);
            terminalProof.begin();
            auto* node=tree.root();unsigned newNodes=0;
            J traceSteps=J::array();const bool capture=detail&&traces.size()<8;
            for(int depth=0;depth<horizon;++depth){
                checkTime();
                if(boundaryCounts.cut(boundaryScope.config.direct,unsigned(depth),node!=nullptr))break;
                spire_tree::Edge* edge=nullptr;J action;
                if(depth==0){edge=first;action=J::parse(first->action);}
                else {++policies;J prior;std::vector<double> scores;
                    if constexpr(ScoreOrder){
                        if(tree.needsScores(node)){
                            auto ranked=scoredCombatChoice(view,legal,metadata);prior=std::move(ranked.prior);scores=std::move(ranked.scores);
                            ++orderingBatches;orderingScores+=scores.size();extraDiscardScores+=ranked.extraDiscardScores;
                        }else prior=combatChoice(view,legal,metadata);
                    }else if(!spire_lazy_rule::enabled||!neural.changes()) prior=combatChoice(view,legal,metadata);
                    checkTime();edge=node?neural.select(node,view,legal,rich):nullptr;
                    if(!edge){
                        // Only the fallback uses this prior. ScoreOrder retains its
                        // original eager rule scores, and probe/disabled neural modes
                        // retain their original rule path and validation.
                        if constexpr(!ScoreOrder)
                            if(spire_lazy_rule::enabled&&neural.changes())prior=combatChoice(view,legal,metadata);
                        edge=node?tree.select(node,legal,prior,ScoreOrder?&scores:nullptr):nullptr;
                    }
                    if(boundaryScope.config.direct&&!edge)throw std::logic_error("direct_actor_forbidden");
                    action=edge?J::parse(edge->action):(actorMode.empty()?prior:spire_cold::leaf(view,legal,prior,actorMode,actorCounts));}
                if(std::find(legal.begin(),legal.end(),action)==legal.end())
                    throw std::runtime_error("standalone_illegal_action");
                if(evaluator.collection().enabled()&&depth>=16&&edge)throw std::runtime_error("frontier_non_greedy_continuation");
                if(edge){++treeSteps;path.emplace_back(node,edge);}else ++leafSteps;
                if(capture)traceSteps.push_back({{"depth",depth},{"node",node?J(node->id):J()},
                    {"action",action},{"tree_policy",bool(edge)},{"public",view}});
                terminalProof.before(view,rich,action);
                std::optional<spire_terminal_proof::EntropyStamp> entropy;
                if(terminalProof.tracking())entropy.emplace(*world.battle);
                const bool replacing=spire_potion_replacement::enabled&&action.at("kind")=="discard_potion";
                const int refill=replacing?replacementSource(*world.battle,number(action,"potion_index")):-1;
                if(replacing&&refill<0)throw std::runtime_error("discard_without_immediate_replacement");
                ++steps;auto observed=roots.advancePublic(world,action);
                if(replacing){
                    // Both physical actions use the public event machinery. No
                    // leaf value or further choice occurs between the two.
                    if(steps>=stepCap)throw std::runtime_error("replacement_step_budget");
                    const J acquire={{"kind","use_potion"},{"potion_index",refill}};
                    ++steps;++replacementSteps;auto more=roots.advancePublic(world,acquire);
                    observed.insert(observed.end(),more.begin(),more.end());
                    if(capture)traceSteps.back()["replacement_followup"]=acquire;
                }
                view=project();checkTime();
                if(entropy)terminalProof.after(entropy->unchanged(*world.battle),observed);
                neural.observe(*world.battle,richMemory,rich,observed);neural.key(observed,view,rich);
                auto key=spire_tree::observationKey(observed,limits.observationBytes);
                if(outcome(*world.battle))break;
                guard(view,metadata);
                if(evaluator.enabled())evaluator.probe(depth+1,view,actions(*world.battle,true),rich);
                if(depth+1==horizon)break;
                legal=actions(*world.battle,true);
                node=tree.descend(node,edge,key,legal,newNodes);
            }
            ++values;const int terminal=outcome(*world.battle);const double value=combatValue(view,terminal);
            if(!std::isfinite(value))throw std::runtime_error("standalone_nonfinite_value");
            checkTime();auto row=spire_objective::record(view,input.policy,terminal,value);
            row=spire_objective::applyVersion(std::move(row),spire_objective::name(budget));
            spire_objective::Outcome result{row.at("value"),row.at("hp"),row.at("resource_hp"),row.at("outcome")};
            if(evaluator.enabled()){
                auto distribution=evaluator.evaluate(view,input.policy,terminal,terminal?J::array():actions(*world.battle,true),rich);
                row=distribution.evidence;tree.backupDistribution(path,distribution);
            }else tree.backup(path,result);
            terminalProof.complete(view,terminal);
            if(observer)observer(tree,steps,false);
            if(capture)traces.push_back({{"simulation",index},{"steps",traceSteps},{"outcome",row},
                {"path_length",path.size()},{"new_nodes",newNodes}});
            if(audit.enabled())audit.observe(index,std::string(first->action),row,view,terminal?J::array():actions(*world.battle,true),rich,world);
        }
        return finish("simulation_cap");
    }catch(const std::exception& e){
        const std::string error=e.what();
        // Only a typed deadline may retain already-complete coverage. All other
        // failures invalidate the recommendation (never authorize a fallback).
        if(error=="standalone_deadline"||roots.failureCode()=="deadline")return finish("deadline");
        return finish("tree_refused",error);
    }
}
} // namespace spire_search
