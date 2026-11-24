// Included INSIDE SimulatorSession by the reproducible integration overlay.
// Ordinary builds do not include this service and keep their original ABI.
private:
    spire_rich::Memory richMemory;
    bool publicSessionEnabled=false;
    std::uint64_t publicSessionEpoch=0;
    std::unique_ptr<spire_session::SessionHistory> publicSession;
    spire_search::J lastPublicBattleResult;
    void resetPublicSession() {publicSession.reset();}
    void startPublicSession() {
        richMemory=spire_rich::Memory{};
        publicSession=std::make_unique<spire_session::SessionHistory>(
            *simulator->battleSim.bc,++publicSessionEpoch,spire_search::J::array());
    }
    void commitPublicSession(const spire_search::J& action=spire_search::J(),
        const std::vector<public_sampling::PublicTheftEvent>* events=nullptr) {
        spire_search::J relics=spire_search::J::array();
        for(const auto& r:simulator->gc->relics.relics)relics.push_back(spire_search::relicKey(r.id));
        publicSession->setRelics(std::move(relics));publicSession->commit(action,events);
        richMemory.observe(*simulator->battleSim.bc);
    }
public:
    void enablePublicSession(bool enabled) {
        if(simulator&&simulator->battleSim.initialized)
            throw std::runtime_error("session mode must be selected before combat");
        publicSessionEnabled=enabled;resetPublicSession();
    }
    void requireStructuredPublicAction() const {
        if(publicSessionEnabled&&simulator->battleSim.initialized&&simulator->gc->screenState==ScreenState::BATTLE)
            throw std::runtime_error("public session requires take_public_action; console bypass refused");
    }
    void requireCloneablePublicSession() const {
        if(publicSession&&simulator->battleSim.initialized&&!public_state::effectsResolved(*simulator->battleSim.bc))
            throw std::runtime_error("paused public session must be reconstructed, not exactly queue-cloned");
    }
    std::string autoplayTargetScope() const {
        if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        return publicSession->autoplayTargetScope().dump();
    }
    std::string publicSessionStatus() const {
        using spire_search::J;
        return publicSession?publicSession->status().dump():
            J{{"version",spire_session::sessionVersion},{"enabled",publicSessionEnabled},
              {"epoch",publicSessionEpoch},{"anchor",false},{"revision",0}}.dump();
    }
    std::string publicSessionView() const {
        if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        return publicSession->view().dump();
    }
    std::string richPublicView() const {
        if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        publicSession->fresh();
        auto order=spire_search::J::array();
        for(const auto& r:simulator->gc->relics.relics)order.push_back({{"id",int(r.id)}});
        return spire_rich::project(*simulator->battleSim.bc,order,richMemory.previous).dump();
    }
    std::string publicBattleResult() const {return lastPublicBattleResult.dump();}
    void takePublicAction(const std::string& command,const std::string& encoded) {
        ensureReady();
        if(!publicSessionEnabled)throw std::runtime_error("public_session_disabled");
        lastPublicBattleResult=nullptr;
        using spire_search::J;
        if(simulator->battleSim.initialized&&simulator->gc->screenState==ScreenState::BATTLE) {
            if(!publicSession)throw std::runtime_error("missing_battle_opening_journal");
            publicSession->fresh();auto action=J::parse(encoded);int top=-1;
            std::vector<public_sampling::PublicTheftEvent> theftEvents;
            {public_sampling::PublicTheftScope theftScope(*simulator->battleSim.bc,theftEvents);
             spire_session::advanceObserved(*simulator->battleSim.bc,action,top,publicSession->recorder());}
            commitPublicSession(action,&theftEvents);observeEnemies(&theftEvents);
            if(simulator->battleSim.isBattleComplete()) {
                const auto& b=*simulator->battleSim.bc;
                // Public terminal telemetry BEFORE postcombat rewards/healing or
                // another boss battle replaces the context. No RNG/raw piles.
                lastPublicBattleResult={{"version","public-battle-result-v1"},
                    {"outcome",spire_search::outcome(b)},{"hp",b.player.curHp},
                    {"potion_reserve",spire_search::inventoryReserve(publicSession->view())}};
            }
            if(simulator->battleSim.isBattleComplete()&&!combatOnly) {
                simulator->battleSim.exitBattle(*simulator->gc);
                simulator->battleSim.initialized=false;resetPublicSession();
                if(simulator->gc->screenState==ScreenState::BATTLE) {
                    startPublicSession();
                    {spire_belief::trace::MechanismRecorder::Scope scope(publicSession->recorder());
                     simulator->battleSim.initBattle(*simulator->gc);}
                    commitPublicSession();observeEnemies();
                }
            }
        } else {
            // Opening hooks must be armed BEFORE a map/event action starts a
            // battle. Reconstructing an opening from the observed hand later
            // would lose the public order history.
            startPublicSession();
            {spire_belief::trace::MechanismRecorder::Scope scope(publicSession->recorder());takeAction(command);}
            if(simulator->battleSim.initialized&&simulator->gc->screenState==ScreenState::BATTLE)
                commitPublicSession();
            else resetPublicSession();
        }
    }
    std::string searchSessionNeuralPrior(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(!b.contains("neural_prior"))throw std::invalid_argument("missing_neural_prior_mode");
        auto order=J::array();for(const auto& r:simulator->gc->relics.relics)order.push_back({{"id",int(r.id)}});
        spire_neural_prior::Scope scope(J::parse(richPublicView()),std::move(order));
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,b,J::parse(metadata),trace,
            false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionPublic(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;
        // Deliberately keep the GIL for this first service. A second Python
        // thread cannot mutate/reset a session while its history is sampled.
        // Independent worker PROCESSES remain parallel.
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),trace).dump();
    }
    std::string searchSessionTree(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),trace,true).dump();
    }
    std::string searchSessionGuardedTree(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),trace,true,true).dump();
    }
    std::string searchSessionPairedTree(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),trace,true,true,nullptr,true).dump();
    }
    std::string searchSessionStandaloneTree(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;
        if(J::parse(budget).value("tree_algorithm","")!=spire_tree::standaloneVersion)
            throw std::runtime_error("standalone_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionDeepUct(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::deepQuarterVersion)throw std::runtime_error("deep_uct_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionRootObjective(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::rootObjectiveVersion)throw std::runtime_error("root_objective_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string diagnoseSessionContinuation(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,const std::string& diagnostic,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto spec=J::parse(diagnostic);
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,J::parse(budget),
            J::parse(metadata),trace,false,false,nullptr,false,nullptr,true,&spec).dump();
    }
    std::string searchSessionRootRecency(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::rootRecencyVersion)throw std::runtime_error("root_recency_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionDeepPrior(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::deepPriorVersion)throw std::runtime_error("deep_prior_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionScoreOrder(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::scoreOrderVersion)throw std::runtime_error("score_order_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionFrozenRoot(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::frozenRootVersion)throw std::runtime_error("frozen_root_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string searchSessionRoundTree(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,bool trace) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto b=J::parse(budget);
        if(b.value("tree_algorithm","")!=spire_tree::roundVersion)throw std::runtime_error("root_round_binding_version");
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            b,J::parse(metadata),trace,false,false,nullptr,false,nullptr,true).dump();
    }
    std::string auditTreeProposal(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,const std::string& diagnostic) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto request=J::parse(diagnostic);
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),false,false,false,nullptr,false,&request).dump();
    }
    std::string diagnoseFirstTreeOverride(const std::vector<std::array<std::uint64_t,7>>& seeds,
        const std::string& expected,const std::string& actions,const std::string& budget,
        const std::string& metadata,const std::string& diagnostic) {
        ensureReady();if(!publicSessionEnabled||!publicSession)throw std::runtime_error("missing_public_session");
        using spire_search::J;auto probe=J::parse(diagnostic);
        return publicSession->search(J::parse(expected),J::parse(actions),seeds,
            J::parse(budget),J::parse(metadata),true,true,true,&probe).dump();
    }
