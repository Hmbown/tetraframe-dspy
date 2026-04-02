from __future__ import annotations

from tetraframe.artifacts import (
    ArbiterArtifact,
    CartographyArtifact,
    CornerDraftArtifact,
    CornerInputView,
    CornerMode,
    CornerReconstructionArtifact,
    DistilledSeedArtifact,
    EvidenceDiscriminatorArtifact,
    HardenedCornerArtifact,
    PairwiseRelationArtifact,
    PredicateSelectionArtifact,
    PredicateSpec,
    RejectedPredicate,
    RelationType,
    StageTraceArtifact,
    TetraFrameRunArtifact,
    TransformedFrameArtifact,
    VerificationMetricArtifact,
    VerificationReportArtifact,
    WritingAdapterArtifact,
    CodingAdapterArtifact,
    ResearchAdapterArtifact,
    PlanningAdapterArtifact,
)
from tetraframe.guards import make_corner_input_view


def build_sample_run_artifact() -> TetraFrameRunArtifact:
    distilled = DistilledSeedArtifact(
        run_id="run_sample0001",
        raw_seed="We need a universal reasoning framework that avoids anchoring bias from sequential thesis→antithesis generation by producing independent tetralemmatic corners and transforming them into useful outputs for writing, coding, research, and planning.",
        normalized_project_seed="Design a universal reasoning program that prevents sequential anchoring by generating four independent tetralemmatic corners from an operational predicate and transforming them into domain-specific outputs.",
        stakes=[
            "reduce anchoring bias",
            "preserve genuine alternative structures",
            "produce usable outputs across domains",
        ],
        constraints=[
            "must operate for writing, coding, research, and planning",
            "must preserve branch independence during generation",
            "must verify rigor and transformation quality",
        ],
        unknowns=[
            "which predicate best captures the intervention",
            "how to measure branch contamination reliably",
            "which transformation best outperforms compromise",
        ],
        hidden_assumptions=[
            "branch independence can be operationalized in software",
            "parallel exploration and structured synthesis can be separated",
            "tetralemmatic outputs can be adapted into useful domain artifacts",
        ],
        candidate_predicates=[
            "Independent parallel tetralemmatic branching reduces anchoring bias relative to sequential thesis→antithesis generation.",
            "Tetralemmatic branching should be the first reasoning primitive for cross-domain systems.",
            "Useful outputs should be generated from a transformed framing rather than raw corners.",
            "Branch independence is enforceable and measurable in software.",
        ],
        frame_risk_score=0.89,
        evaluation_criteria=[
            "branch independence",
            "divergence quality",
            "rigor of both and neither",
            "transformation quality",
            "domain usefulness",
        ],
        novelty_criteria=[
            "no sequential anchoring",
            "non-averaging transformed frame",
            "explicit cartography",
        ],
    )
    selection = PredicateSelectionArtifact(
        primary_predicate=PredicateSpec(
            text="Using independent parallel tetralemmatic corner generation as the exploration primitive yields less anchoring bias and higher transformation quality than sequential thesis→antithesis generation.",
            subject="reasoning program",
            relation="yields less anchoring bias and higher transformation quality than",
            object="sequential thesis→antithesis generation",
            measurable_terms=["anchoring bias", "divergence quality", "transformation quality"],
            scope=["framing-heavy reasoning tasks", "design", "planning", "writing"],
            objective_type="comparative process claim",
            operational_tests=[
                "compare pairwise corner divergence against a sequential baseline",
                "compare contamination score against thesis-first baseline",
                "compare transformation-quality score on held-out seeds",
            ],
        ),
        sub_predicates=[
            PredicateSpec(text="Branch independence is enforceable in code through isolated views, fresh rollout IDs, and no cross-branch visibility."),
            PredicateSpec(text="A transformed frame P* should be generated after explicit cartography, not by averaging corners."),
        ],
        rejected_predicates=[
            RejectedPredicate(text="Universal reasoning framework", reason="too broad", rewrite_suggestion="state the exact comparative process claim"),
            RejectedPredicate(text="Anchoring bias is bad", reason="normative but non-operational", rewrite_suggestion="state how bias changes under different generation topologies"),
        ],
        rationale="The selected predicate is comparative, falsifiable, and rich enough to support all four corners without collapsing into a slogan.",
        operationalization_notes=[
            "measure divergence before cross-branch exposure",
            "treat both and neither as validity-typed corners rather than softer opinions",
            "score P* for non-averaging transformation",
        ],
    )

    corner_inputs = {mode: make_corner_input_view(distilled, selection, mode) for mode in CornerMode}

    corner_drafts = {
        CornerMode.P: CornerDraftArtifact(
            corner_mode=CornerMode.P,
            core_claim="Independent parallel tetralemmatic branching should replace thesis-first generation at the exploration stage because it prevents early anchoring and preserves structurally distinct alternatives.",
            assumptions=[
                "sequential antithesis is biased by the thesis it answers",
                "independence can be enforced materially rather than rhetorically",
            ],
            strongest_case="A thesis-first pipeline narrows the search space before alternatives exist. Parallel independent corners preserve hypothesis diversity, enable real frame discovery, and produce a stronger transformed frame later because synthesis begins from independently generated structures rather than rehearsed opposition.",
            scope_conditions=["high-framing tasks", "multi-step design", "planning under ambiguity"],
            falsifiers=[
                "A sequential baseline achieves equal contamination and transformation scores on matched seeds.",
                "Parallel branches collapse into near-duplicates despite isolation controls.",
            ],
            evidence_needs=[
                "trace logs comparing pairwise divergence before synthesis",
                "benchmark results comparing contamination scores across branch topologies",
            ],
            uncertainty="Effect size depends on model family and whether branch isolation is enforced at runtime.",
            unique_signal="Exploration topology itself is the intervention; later synthesis quality depends on preserving alternative search structures early.",
            validity_basis_label="affirmation",
            validity_basis_explanation="The selected predicate is affirmed directly as a comparative process claim.",
        ),
        CornerMode.NOT_P: CornerDraftArtifact(
            corner_mode=CornerMode.NOT_P,
            core_claim="Parallel tetralemmatic branching is not the essential intervention; disciplined sequential generation with strong critique can match or exceed its quality while costing less and preserving more useful context.",
            assumptions=[
                "quality depends more on critique and verification than on initial topology",
                "later reasoning benefits from conditioned context",
            ],
            strongest_case="If the real failure is weak critique rather than generation order, then parallelization is expensive theater. A strong sequential pipeline can generate a thesis, then a deliberately adversarial dismantling, then a frame-dissolving critique, and still reach a transformed frame without paying the coordination and latency costs of four isolated branches.",
            scope_conditions=["low-ambiguity tasks", "cost-sensitive systems", "domains with strong evaluators"],
            falsifiers=[
                "Sequential systems consistently underperform on divergence and contamination metrics even after strong critique.",
                "Sequential conditioning causes recoverable but persistent frame omission across tasks.",
            ],
            evidence_needs=[
                "matched-cost benchmark comparing sequential critique vs isolated parallel branching",
                "latency and token-cost traces against final usefulness scores",
            ],
            uncertainty="Sequential systems may fail precisely on the broad, under-specified seeds where alternative structures matter most.",
            unique_signal="The bottleneck may be verification strength, not initial branch topology.",
            validity_basis_label="rejection",
            validity_basis_explanation="The core predicate is rejected by treating topology as secondary to verification quality.",
        ),
        CornerMode.BOTH: CornerDraftArtifact(
            corner_mode=CornerMode.BOTH,
            core_claim="Independent parallel branching is necessary during exploration, while conditioned sequential interaction becomes necessary during synthesis; both P and not-P hold when separated by role in the pipeline.",
            assumptions=[
                "exploration and synthesis are different reasoning roles",
                "the best control variable changes across roles",
            ],
            strongest_case="At exploration time, independence prevents anchoring and protects structural alternatives. At synthesis time, those same branches must be brought into relation through explicit comparison, contradiction mapping, and transformation. The system therefore needs non-sequentiality first and structured sequential conditioning later.",
            scope_conditions=["two-phase pipelines", "systems that separate search from synthesis"],
            falsifiers=[
                "A single topology works best for both exploration and synthesis without trade-offs.",
                "Sequential conditioning during synthesis reintroduces the same anchoring defects measured in exploration.",
            ],
            evidence_needs=[
                "ablation comparing exploration-first independence vs synthesis-first conditioning",
                "per-stage score breakdowns for divergence, contradiction honesty, and transformation quality",
            ],
            uncertainty="The boundary between exploration and synthesis can blur in interactive workflows.",
            unique_signal="The key hidden variable is pipeline role, not a global preference for parallel or sequential reasoning everywhere.",
            validity_basis_label="role_split",
            validity_basis_explanation="P and not-P co-hold because exploration and synthesis impose different control requirements.",
        ),
        CornerMode.NEITHER: CornerDraftArtifact(
            corner_mode=CornerMode.NEITHER,
            core_claim="The original predicate is partially misframed because it treats exploration topology and synthesis topology as a single binary decision; neither 'parallel is best' nor 'sequential is best' is the right predicate.",
            assumptions=[
                "the real design variable is topology by stage",
                "the seed bundles search, verification, and adaptation into one predicate",
            ],
            strongest_case="The seed collapses multiple variables into one contest: initial branch generation, later synthesis, verification quality, and domain adaptation. That makes the original predicate overloaded. The correct question is not whether reasoning should be parallel or sequential in the abstract, but which topology maximizes branch independence during exploration and transformation quality during synthesis under cost constraints.",
            scope_conditions=["systems with stage-dependent goals", "multi-objective reasoning programs"],
            falsifiers=[
                "A single topology dominates across exploration, synthesis, and adaptation without trade-offs.",
                "Stage-specific topology provides no measurable advantage over a single global topology.",
            ],
            evidence_needs=[
                "per-stage benchmark with exploration and synthesis metrics separated",
                "error analysis showing frame failures caused by predicate overload",
            ],
            uncertainty="The replacement predicate may still need domain-specific refinement for cost or safety constraints.",
            unique_signal="The original predicate is overloaded because it treats a staged control problem as a single comparative claim.",
            validity_basis_label="overloaded_predicate",
            validity_basis_explanation="The original predicate bundles exploration topology, synthesis topology, and evaluation strategy into one overloaded claim.",
            replacement_predicate="A reasoning program should use the topology that maximizes branch independence during exploration and transformation quality during synthesis, rather than choose one topology globally.",
            replacement_frame="Separate exploration control, synthesis control, and evaluation control before comparing pipeline designs.",
        ),
    }

    hardened = {
        CornerMode.P: HardenedCornerArtifact(
            **corner_drafts[CornerMode.P].model_dump(),
            internal_attack=["The claim may over-attribute failure to order when weak critique is the real cause.", "Parallelism alone does not guarantee meaningful divergence."],
            patched_claim="For framing-heavy tasks, isolated parallel tetralemmatic branching should replace thesis-first generation at the exploration stage because it reduces premature anchoring and preserves structurally distinct alternatives that later support better transformation.",
            patched_assumptions=[
                "order effects matter most in framing-heavy tasks",
                "branch isolation is implemented materially, not just by instruction",
            ],
            clarified_scope_conditions=["framing-heavy tasks", "ambiguous design spaces", "open-ended planning"],
            confidence_boundaries=["less relevant when the task is already well-specified", "less relevant when a strong external evaluator dominates quality"],
            minimal_falsifiers=[
                "A matched sequential baseline achieves equal divergence, contamination, and transformation scores on framing-heavy seeds.",
                "Independent branches converge to the same structure after isolation controls are enabled.",
            ],
            tightened_language="The intervention is not 'more perspectives'; it is independence at exploration time.",
            unresolved_weaknesses=["runtime isolation may be expensive on large models"],
            confidence_score=0.81,
        ),
        CornerMode.NOT_P: HardenedCornerArtifact(
            **corner_drafts[CornerMode.NOT_P].model_dump(),
            internal_attack=["This may understate how hard it is to generate genuine alternatives after a thesis is fixed."],
            patched_claim="Parallel tetralemmatic branching is not the only viable intervention; when critique and evaluation are unusually strong and costs matter, disciplined sequential generation can match much of its value without four isolated branches.",
            patched_assumptions=["high-quality critique is available", "the task has moderate framing ambiguity"],
            clarified_scope_conditions=["cost-sensitive deployments", "moderately specified tasks", "pipelines with strong evaluators"],
            confidence_boundaries=["weak critique collapses this corner", "broad seeds reduce its plausibility"],
            minimal_falsifiers=[
                "Sequential critique remains worse on contamination and transformation quality after cost matching.",
                "Thesis-conditioned generation systematically misses structures later recovered only by isolated branches.",
            ],
            tightened_language="The claim is about sufficiency under strong critique, not about parallelism being useless.",
            unresolved_weaknesses=["hard to guarantee adversarial quality without reintroducing anchoring"],
            confidence_score=0.67,
        ),
        CornerMode.BOTH: HardenedCornerArtifact(
            **corner_drafts[CornerMode.BOTH].model_dump(),
            internal_attack=["The role split can drift into a banal 'both sides matter' move if stage boundaries remain vague."],
            patched_claim="A high-quality reasoning program should use isolation-first parallel branching during exploration and explicit cross-branch conditioning during synthesis; both P and not-P hold only because exploration and synthesis are distinct roles with different failure modes.",
            patched_assumptions=["roles are explicit and observable in the pipeline", "the transition from exploration to synthesis is logged"],
            clarified_scope_conditions=["staged reasoning systems", "pipelines with a visible synthesis handoff"],
            confidence_boundaries=["weak if the pipeline never separates exploration from synthesis"],
            minimal_falsifiers=[
                "One topology dominates both exploration and synthesis after stage separation.",
                "Role separation does not improve transformation or contradiction honesty.",
            ],
            tightened_language="The basis is role split, not compromise.",
            unresolved_weaknesses=["requires clean stage interfaces to stay rigorous"],
            confidence_score=0.85,
        ),
        CornerMode.NEITHER: HardenedCornerArtifact(
            **corner_drafts[CornerMode.NEITHER].model_dump(),
            internal_attack=["Overload diagnosis can become evasive if it never yields a replacement predicate."],
            patched_claim="The original predicate is overloaded because it asks one global topology question where the real design problem is stage-specific control of exploration, synthesis, and evaluation. Neither a blanket pro-parallel nor blanket pro-sequential predicate is well-posed.",
            patched_assumptions=["stage-specific control variables are measurable", "overload is causing reasoning failure rather than mere complexity"],
            clarified_scope_conditions=["multi-stage reasoning systems", "pipelines optimizing more than one objective"],
            confidence_boundaries=["less compelling when the task truly has one dominant objective and one reasoning stage"],
            minimal_falsifiers=[
                "A single global topology dominates across exploration, synthesis, and evaluation on matched tasks.",
                "Stage-specific decomposition adds no predictive or design value.",
            ],
            tightened_language="The diagnosis is overload, not indecision.",
            unresolved_weaknesses=["replacement predicate may need cost terms for deployment use"],
            confidence_score=0.88,
        ),
    }

    cartography = CartographyArtifact(
        pairwise_relations=[
            PairwiseRelationArtifact(source_corner=CornerMode.P, target_corner=CornerMode.NOT_P, relation_type=RelationType.CONTRADICTION, rationale="P treats topology as primary; not-P treats critique strength as primary.", evidence_discriminator="Run matched-cost sequential vs isolated parallel benchmarks with contamination and transformation scores.", reversible=False, invariant_tags=["quality control matters"]),
            PairwiseRelationArtifact(source_corner=CornerMode.P, target_corner=CornerMode.BOTH, relation_type=RelationType.COMPLEMENTARITY, rationale="Both preserves P for exploration but limits its scope to one role.", evidence_discriminator="Separate exploration and synthesis metrics by stage.", reversible=True, invariant_tags=["exploration matters"]),
            PairwiseRelationArtifact(source_corner=CornerMode.P, target_corner=CornerMode.NEITHER, relation_type=RelationType.TRANSFORMATION, rationale="Neither dissolves the global topology frame while preserving P's insight about exploration independence.", evidence_discriminator="Compare single-topology vs stage-specific topology pipelines.", reversible=False, invariant_tags=["stage-specific control"]),
            PairwiseRelationArtifact(source_corner=CornerMode.NOT_P, target_corner=CornerMode.BOTH, relation_type=RelationType.COMPLEMENTARITY, rationale="Both preserves not-P's point that synthesis needs conditioning.", evidence_discriminator="Measure whether synthesis-time conditioning improves contradiction honesty after independent exploration.", reversible=True, invariant_tags=["synthesis conditioning"]),
            PairwiseRelationArtifact(source_corner=CornerMode.NOT_P, target_corner=CornerMode.NEITHER, relation_type=RelationType.DISSOLUTION, rationale="Neither absorbs not-P by relocating its force from global topology to synthesis/evaluation stages.", evidence_discriminator="Test whether critique quality explains most variance after stage separation.", reversible=False, invariant_tags=["verification strength"]),
            PairwiseRelationArtifact(source_corner=CornerMode.BOTH, target_corner=CornerMode.NEITHER, relation_type=RelationType.TRANSFORMATION, rationale="Neither reframes both's role split as a new predicate over stage-specific control variables.", evidence_discriminator="Compare role-split explanation against overload diagnosis across benchmark cases.", reversible=False, invariant_tags=["stage decomposition"]),
        ],
        contradiction_map=["P contradicts not-P on whether exploration topology is the primary intervention."],
        complementarity_map=["Both preserves P for exploration and not-P for synthesis."],
        paradox_map=["The system needs non-sequentiality to avoid anchoring, then explicit sequential comparison to transform the branches."],
        category_error_map=["The original predicate treats a staged control problem as a single global comparison."],
        frame_validity_map=["P is valid for exploration-heavy tasks.", "not-P is locally valid under strong critique and cost pressure.", "both is valid only under an explicit role split.", "neither is valid because the original predicate is overloaded."],
        evidence_discriminator_map=[
            EvidenceDiscriminatorArtifact(discriminator="matched-cost benchmark by stage", corner_favors=[CornerMode.P, CornerMode.NOT_P, CornerMode.BOTH], evidence_needed=["exploration divergence score", "synthesis contradiction-honesty score", "overall transformation quality"]),
            EvidenceDiscriminatorArtifact(discriminator="single-topology vs stage-specific-topology comparison", corner_favors=[CornerMode.NEITHER, CornerMode.BOTH], evidence_needed=["per-stage error analysis", "frame-failure taxonomy"]),
        ],
        invariant_map=["Reasoning quality improves when hidden control variables are made explicit.", "Transformation quality depends on preserving alternative structures before synthesis."],
        reversible_implications=["If synthesis is separated from exploration, not-P can survive without killing P."],
        irreversible_implications=["Once the global topology frame is dissolved, the original predicate cannot be the final frame."],
        structural_miss_map={
            "P": "Exploration topology is the core intervention.",
            "not-P": "Critique and verification may dominate topology on some tasks.",
            "both": "Different pipeline roles can require opposite control rules.",
            "neither": "The original predicate overloads several design variables into one comparison.",
        },
    )

    arbiter = ArbiterArtifact(
        reconstructions=[
            CornerReconstructionArtifact(corner_mode=CornerMode.P, strongest_fair_restatement="The exploration stage should begin with isolated parallel corners because order effects materially narrow the search space.", unsupported_premises=[]),
            CornerReconstructionArtifact(corner_mode=CornerMode.NOT_P, strongest_fair_restatement="Topology is not the whole story; strong critique may recover much of the same value at lower cost.", unsupported_premises=[]),
            CornerReconstructionArtifact(corner_mode=CornerMode.BOTH, strongest_fair_restatement="Exploration and synthesis are different roles, so opposite control rules can both be correct in a single pipeline.", unsupported_premises=[]),
            CornerReconstructionArtifact(corner_mode=CornerMode.NEITHER, strongest_fair_restatement="The real design question is stage-specific control, not a single global topology choice.", unsupported_premises=[]),
        ],
        opposition=["P and not-P oppose each other on the primary intervention variable."],
        contradiction=["P says exploration topology is primary; not-P says strong critique can make it secondary."],
        complementarity=["Both preserves the best local truths from P and not-P under role split."],
        paradox=["The program must reject sequencing at one stage and require sequencing at another."],
        dissolution=["Neither dissolves the single global topology frame."],
        transformation=["A stage-specific control predicate emerges that cannot be reduced to any single original corner."],
        arbiter_notes="No corner should be collapsed into a generic debate. P contributes exploration control, not-P contributes evaluation pressure, both reveals stage structure, and neither repairs the frame itself.",
    )

    transformed = TransformedFrameArtifact(
        transformed_predicate="A reasoning program should separate exploration topology from synthesis topology: use isolation-first tetralemmatic branching to maximize branch independence during exploration, then explicit cartography and constrained cross-branch comparison to maximize transformation quality during synthesis.",
        transformed_frame="The design problem is not 'parallel versus sequential reasoning' in the abstract. It is staged control: preserve independent structure during exploration, expose contradiction and complementarity during cartography, and only then synthesize a transformed frame and domain outputs.",
        survivors_from_p=["Exploration should begin with isolated branches.", "Branch independence is a primary control variable."],
        survivors_from_not_p=["Verification strength still matters.", "Conditioned comparison becomes valuable during synthesis."],
        hidden_structure_from_both=["Exploration and synthesis are distinct roles with different optimal control rules."],
        dissolved_false_frame_from_neither=["There is no single globally optimal topology across all reasoning stages.", "The original predicate was overloaded."],
        non_averaging_explanation="P* is not a compromise between parallel and sequential reasoning. It replaces the global topology frame with a staged-control frame in which different rules govern exploration and synthesis.",
        operational_tests=[
            "Measure divergence and contamination before any branch comparison.",
            "Measure contradiction honesty and transformation quality after cartography.",
            "Compare stage-specific topology against single-topology baselines on matched seeds.",
        ],
        boundary_conditions=[
            "Less benefit on tightly specified tasks with trivial framing.",
            "Requires a logged handoff from exploration to synthesis.",
        ],
        failure_modes=[
            "Branches are isolated only superficially and still converge.",
            "Synthesis begins before cartography, recreating early anchoring.",
            "The transformed frame adds novelty without support from the corners.",
        ],
        confidence=0.87,
    )

    writing = WritingAdapterArtifact(
        central_claim="Reasoning programs should separate exploration from synthesis instead of forcing every stage into a pro/con sequence.",
        rival_readings=[
            "Topology is the main lever.",
            "Critique strength is the main lever.",
            "The real lever is stage-specific control.",
        ],
        tension_map=[
            "independence vs contextual conditioning",
            "search breadth vs coordination cost",
            "novelty vs support",
        ],
        outline=[
            "Define the failure of thesis-first anchoring.",
            "Show why independent corners change exploration.",
            "Map the contradiction with the sequential critique view.",
            "Introduce stage-specific control as P*.",
            "Translate P* into a revision workflow.",
        ],
        voice_options=["technical-analytical", "editorial polemic", "research memo"],
        stress_test=[
            "Could strong sequential critique recover the same structures?",
            "Where does stage separation fail in practice?",
        ],
        revision_plan=[
            "tighten the distinction between exploration and synthesis",
            "add a concrete benchmark example",
            "remove any wording that sounds like compromise",
        ],
    )
    coding = CodingAdapterArtifact(
        architecture="Two-phase pipeline with isolated corner workers, cartography engine, transformer, and domain adapters.",
        modules=[
            "SeedDistillModule",
            "PredicateSelectModule",
            "CornerPGenerator",
            "CornerNotPGenerator",
            "CornerBothGenerator",
            "CornerNeitherGenerator",
            "HardenCornerModule",
            "CartographCornersModule",
            "FourCornerArbiterModule",
            "TransformFrameModule",
        ],
        interfaces=[
            "CornerInputView -> CornerDraftArtifact",
            "CornerDraftArtifact -> HardenedCornerArtifact",
            "{HardenedCornerArtifact} -> CartographyArtifact",
            "CartographyArtifact + ArbiterArtifact -> TransformedFrameArtifact",
        ],
        state_model="Artifacts are immutable stage outputs keyed by run_id and corner_mode; no corner may read any sibling artifact before stage 4.",
        verification_loop=[
            "compute contamination and divergence scores",
            "retry collapsed branches with anti-collapse hints",
            "score transformation quality before releasing adapters",
        ],
        tests=[
            "isolation view test",
            "canary contamination test",
            "near-duplicate corner retry test",
            "non-averaging transformation test",
        ],
        failure_modes=[
            "shared context leaks across branches",
            "both degrades into compromise",
            "neither diagnoses failure but forgets replacement predicate",
        ],
        iteration_plan=[
            "compile stage 0 and 1 first",
            "compile both and neither with stronger optimizers",
            "run GEPA on program traces to improve transformation",
        ],
    )
    research = ResearchAdapterArtifact(
        competing_hypotheses=[
            "Independent exploration topology is the main cause of better transformation.",
            "Verification strength dominates topology.",
            "Stage-specific control explains most of the variance.",
        ],
        discriminating_experiments=[
            "Compare single-topology vs stage-specific-topology pipelines on matched seeds.",
            "Hold critique strength constant while varying exploration topology.",
            "Hold topology constant while varying synthesis cartography depth.",
        ],
        evidence_agenda=[
            "collect divergence, contamination, contradiction-honesty, and transformation-quality scores",
            "store traces for failed branch-isolation runs",
        ],
        confound_map=[
            "model family may alter divergence independently of topology",
            "temperature and rollout policy may masquerade as independence",
            "task ambiguity moderates topology benefit",
        ],
        interpretation_grid=[
            "If stage-specific topology wins, prefer P*.",
            "If sequential critique matches all scores at lower cost, narrow P* scope.",
            "If neither topology matters, search for a hidden evaluation variable.",
        ],
        next_step_program=[
            "build benchmark seeds by domain",
            "label acceptable both bases and neither failure modes",
            "run ablations for no-hardening and shared-context corners",
        ],
    )
    planning = PlanningAdapterArtifact(
        option_set=[
            "ship full stage-specific topology pipeline",
            "ship isolated exploration only and keep current synthesis",
            "keep sequential baseline and strengthen critique loops",
        ],
        leverage_points=[
            "input isolation contract",
            "cartography quality",
            "non-averaging transformation guardrail",
        ],
        decision_thresholds=[
            "adopt full pipeline if aggregate verification >= 0.78 and contamination >= 0.90",
            "ship exploration-only variant if divergence gains appear but synthesis remains weak",
        ],
        scenario_map=[
            "high ambiguity -> full stage-specific topology",
            "low ambiguity, high cost pressure -> stronger sequential critique may suffice",
        ],
        reversibility_map=[
            "exploration isolation is reversible",
            "changing the top-level predicate schema is moderately reversible",
            "removing cartography is irreversible for transformation quality",
        ],
        execution_phases=[
            "implement typed artifacts and isolation views",
            "build corner generation + hardening",
            "add cartography and transformation",
            "compile and benchmark",
        ],
        monitoring_plan=[
            "track contamination score per run",
            "track both/neither rigor over benchmark sets",
            "audit fake novelty risk on transformed frames",
        ],
    )

    verification = VerificationReportArtifact(
        branch_independence=VerificationMetricArtifact(score=0.95, rationale="All corner inputs use isolated views and no explicit cross-references appear.", passed=True),
        divergence_quality=VerificationMetricArtifact(score=0.83, rationale="Incompatible corners remain meaningfully distinct after seed-token subtraction.", passed=True),
        rigor_of_both=VerificationMetricArtifact(score=0.87, rationale="The both corner uses a role split rather than compromise.", passed=True),
        rigor_of_neither=VerificationMetricArtifact(score=0.91, rationale="The neither corner diagnoses overload and proposes a replacement predicate/frame.", passed=True),
        contradiction_honesty=VerificationMetricArtifact(score=0.84, rationale="Direct contradiction is preserved rather than softened away.", passed=True),
        transformation_quality=VerificationMetricArtifact(score=0.89, rationale="P* is a staged-control reframing, not an average.", passed=True),
        actionability=VerificationMetricArtifact(score=0.86, rationale="All domain adapters contain concrete structures and thresholds.", passed=True),
        robustness=VerificationMetricArtifact(score=0.80, rationale="Boundary conditions and failure modes are explicit.", passed=True),
        fake_novelty_risk=VerificationMetricArtifact(score=0.82, rationale="Most transformed-frame concepts are grounded in the corners and cartography.", passed=True),
        slop_risk=VerificationMetricArtifact(score=0.88, rationale="Language is mostly specific and evidence-linked.", passed=True),
        aggregate_score=0.865,
        retry_recommendations=[],
    )

    traces = [
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage0.seed_distill", module_name="SeedDistillModule", signature_name="SeedDistillSignature", input_digest="in0", output_digest="out0", visible_input_fields=["raw_seed"], blocked_input_fields=[]),
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage1.predicate_select", module_name="PredicateSelectModule", signature_name="SplitPredicateSignature+ChoosePredicateSignature", input_digest="in1", output_digest="out1", visible_input_fields=["normalized_project_seed","candidate_predicates","constraints","evaluation_criteria"], blocked_input_fields=[]),
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage2.generate.P", module_name="CornerPGenerator", signature_name="GenerateCornerPSignature", input_digest="in2p", output_digest="out2p", visible_input_fields=list(corner_inputs[CornerMode.P].model_dump().keys()), blocked_input_fields=["other_corner_outputs","cartography","transformed_frame"]),
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage2.generate.not-P", module_name="CornerNotPGenerator", signature_name="GenerateCornerNotPSignature", input_digest="in2n", output_digest="out2n", visible_input_fields=list(corner_inputs[CornerMode.NOT_P].model_dump().keys()), blocked_input_fields=["other_corner_outputs","cartography","transformed_frame"]),
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage2.generate.both", module_name="CornerBothGenerator", signature_name="GenerateCornerBothSignature", input_digest="in2b", output_digest="out2b", visible_input_fields=list(corner_inputs[CornerMode.BOTH].model_dump().keys()), blocked_input_fields=["other_corner_outputs","cartography","transformed_frame"]),
        StageTraceArtifact(run_id="run_sample0001", stage_name="stage2.generate.neither", module_name="CornerNeitherGenerator", signature_name="GenerateCornerNeitherSignature", input_digest="in2z", output_digest="out2z", visible_input_fields=list(corner_inputs[CornerMode.NEITHER].model_dump().keys()), blocked_input_fields=["other_corner_outputs","cartography","transformed_frame"]),
    ]

    return TetraFrameRunArtifact(
        run_id="run_sample0001",
        distilled_seed=distilled,
        predicate_selection=selection,
        corner_inputs=corner_inputs,
        corner_drafts=corner_drafts,
        hardened_corners=hardened,
        cartography=cartography,
        arbiter=arbiter,
        transformed_frame=transformed,
        writing=writing,
        coding=coding,
        research=research,
        planning=planning,
        verification=verification,
        traces=traces,
    )
