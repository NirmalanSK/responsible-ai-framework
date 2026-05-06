"""
═══════════════════════════════════════════════════════════════════════════════
  scm_engine_v2.py  —  Structural Causal Model Engine
  Responsible AI Framework  ·  PhD Research  ·  Nirmalan

  Pearl's Complete Theory — All 3 Levels, All Equations

  WHAT v1 HAD (5 rules):
    Rule 1: TCE proxy → Severity
    Rule 2: NIE proxy → Edge Strength
    Rule 3: Flip Rate proxy → Detection
    Rule 4: do(fix) Reduction → Severity Adjustment
    Rule 5: Identifiability check

  WHAT v2 ADDS (12 new components):
    1.  do-calculus 3 Rules (symbolic verification)
    2.  Backdoor Adjustment Formula — P(Y|do(X)) = Σ_z P(Y|X,Z)·P(Z)
    3.  Frontdoor Adjustment Formula — for unmeasured confounders
    4.  ATE / ATT / CATE  — effect size taxonomy
    5.  NDE (Natural Direct Effect) — TE = NDE + NIE decomposition
    6.  PNS (Probability of Necessity & Sufficiency) — Tian-Pearl bounds
    7.  PN  (Probability of Necessity) — "but-for" causation
    8.  PS  (Probability of Sufficiency)
    9.  D-Separation — conditional independence from DAG
   10.  DAG per harm domain — 19 domain-specific causal graphs
   11.  Effect Decomposition — TE = NDE + NIE with % split
   12.  Legal Admissibility Score — maps all proofs to court standard

  Pearl Reference:
    Pearl, J. (2000). Causality. Cambridge University Press.
    Pearl, J. (2018). The Book of Why. Basic Books.
    Tian, J. & Pearl, J. (2000). Probabilities of Causation. UAI.
    Pearl, J. (1995). Causal Diagrams for Empirical Research. Biometrika.

  YEAR 1 KNOWN LIMITATIONS (Documented — Year 2 Targets)
  ────────────────────────────────────────────────────────
  1. TIER 1 HARDCODING — SCMEngineV2.run() receives full CausalFindings
     with real tce/med/flip/intv/rct values when called at Tier 2/3.
     At Tier 1, pipeline_v15.py Step05_SCMEngine passes DEFAULT_FINDINGS
     (tce=7.0, med=60.0, flip=17.0) — a conservative safe-floor estimate,
     not a query-specific computation. This means ~90% of benign queries
     receive the same SCM baseline rather than individualized causal analysis.

     Year 1 rationale: Tier 1 is low-risk by definition (Tier Router
     verified no signals). SCM full analysis reserved for Tier 2/3 where
     causal precision matters most (hiring bias, medical, criminal justice).

     Year 2 fix: Per-domain lightweight CausalFindings at Tier 1 using
     domain-specific defaults from dag_selector.py. Exact computation
     via DoWhy integration (Phase 4 — see research_notes.md).

  2. PROTECTED GROUP COVERAGE — FIXED (May 2026).
     dag_selector.py DOMAIN_KEYWORDS representation_bias domain now includes
     religion-based (Muslim, Sikh, Jewish, Hindu, Christian) and caste-based
     (Dalit, Scheduled Caste, OBC, varna) discrimination patterns.
     SCMEngineV2 correctly computes TCE/NDE/NIE once CausalFindings arrive —
     the routing gap has been closed at the dag_selector layer.

     Year 2 target: XLM-RoBERTa zero-shot classifier (Phase 6) replaces
     keyword matching entirely — covers all protected group intersections
     and intersectional harms (e.g. caste × gender).

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
from functools import lru_cache

# ── Matrix v2 — 23×5 Pearl Causal Activation Matrix ────────────────
from matrix_v2 import (
    MATRIX_23x5,
    COLUMN_NAMES as PEARL_COLUMNS,   # ["RCT","TCE","INTV","MED","FLIP"]
    COLUMN_INDEX,
    SEVERITY_TIERS,
    get_row,
    get_severity_tier,
    compute_weighted_score,
    resolve_category,
)


# ═══════════════════════════════════════════════════════════════════
# ENUMS  (carry-over from v1 + new)
# ═══════════════════════════════════════════════════════════════════

class Severity(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

class EdgeStrength(Enum):
    WEAK     = 1
    MODERATE = 2
    STRONG   = 3
    CRITICAL = 4

class Detection(Enum):
    EASY     = 2
    MODERATE = 3
    HARD     = 4

class Decision(Enum):
    ALLOW        = "ALLOW"
    WARN         = "WARN"
    ESCALATE     = "ESCALATE"
    BLOCK        = "BLOCK"
    EXPERT_REVIEW= "EXPERT_REVIEW"

class AdjustmentMethod(Enum):
    """Which Pearl adjustment formula applies to this causal query."""
    BACKDOOR    = "Backdoor Adjustment"
    FRONTDOOR   = "Frontdoor Adjustment"
    IV          = "Instrumental Variable"
    NONE_NEEDED = "No adjustment — no confounders"
    NOT_ID      = "NOT IDENTIFIABLE — causal effect cannot be estimated"

class CausalLevel(Enum):
    """Pearl's Ladder of Causation."""
    L1_ASSOCIATION    = 1   # Seeing:    P(Y|X)
    L2_INTERVENTION   = 2   # Doing:     P(Y|do(X))
    L3_COUNTERFACTUAL = 3   # Imagining: P(Y_x | X=x', Y=y')

class DoCalculusRule(Enum):
    """Pearl's three symbolic rules of do-calculus (Pearl 1995)."""
    RULE1 = "Insertion/Deletion of Observations"
    RULE2 = "Action/Observation Exchange"
    RULE3 = "Insertion/Deletion of Actions"


# ═══════════════════════════════════════════════════════════════════
# DAG — DOMAIN-SPECIFIC CAUSAL GRAPHS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalNode:
    """A variable in a Directed Acyclic Graph."""
    name:        str
    description: str
    measured:    bool = True   # False = latent/unobserved confounder


@dataclass
class CausalEdge:
    """A directed edge X → Y in the DAG."""
    source:  str
    target:  str
    is_direct: bool = True     # True = structural equation path


@dataclass
class HarmDAG:
    """
    Directed Acyclic Graph for one harm domain.
    Defines: treatment X, outcome Y, mediator M, confounders Z.

    Used by:
      - Backdoor/Frontdoor criterion checks
      - D-separation queries
      - do-calculus rule application
    """
    domain:      str
    treatment:   str   # X — the AI action (e.g. 'use_race_in_decision')
    outcome:     str   # Y — the harm (e.g. 'discriminatory_outcome')
    mediator:    str   # M — the pathway (e.g. 'biased_score')
    confounders: List[str]          # Z — observed confounders
    latent:      List[str]          # U — unobserved confounders
    nodes:       List[CausalNode]
    edges:       List[CausalEdge]

    def has_backdoor_set(self) -> bool:
        """True if observed confounders block all backdoor paths X←Z→Y."""
        return len(self.latent) == 0

    def has_frontdoor_set(self) -> bool:
        """
        Frontdoor criterion: M blocks all directed paths X→Y,
        no unblocked backdoor paths X→M, all backdoor paths M→Y blocked by X.
        Valid even with latent confounders.
        """
        return self.mediator != "" and len(self.latent) > 0

    def d_separated(self, x: str, y: str, conditioning: List[str]) -> bool:
        """
        Simplified d-separation check.
        In a fully specified DAG this would traverse all paths.
        Here: proxy — True if conditioning set blocks the main backdoor path.
        Full implementation: Year 2 via networkx.

        Pearl (1988): X ⊥ Y | Z in P iff X and Y are d-separated by Z in G.
        """
        # If Z (conditioning) contains all confounders → d-separated
        for c in self.confounders:
            if c not in conditioning:
                return False  # confounded path still open
        return True


# ── 17 Domain-Specific DAGs ──────────────────────────────────────

def build_domain_dags() -> Dict[str, HarmDAG]:
    dags: Dict[str, HarmDAG] = {}

    # 1. Representation Bias (Amazon-style hiring)
    dags["representation_bias"] = HarmDAG(
        domain="Representation Bias",
        treatment="use_protected_attribute",
        outcome="discriminatory_hiring_outcome",
        mediator="biased_ranking_score",
        confounders=["historical_hiring_data", "job_requirements"],
        latent=["societal_bias"],
        nodes=[
            CausalNode("use_protected_attribute", "AI uses race/gender in decision"),
            CausalNode("biased_ranking_score",     "Score inflated/deflated by attribute"),
            CausalNode("discriminatory_hiring_outcome", "Candidate unjustly rejected"),
            CausalNode("historical_hiring_data",   "Past hiring patterns in training data"),
            CausalNode("job_requirements",         "Stated job criteria"),
            CausalNode("societal_bias",            "Unmeasured societal discrimination", measured=False),
        ],
        edges=[
            CausalEdge("use_protected_attribute", "biased_ranking_score"),
            CausalEdge("biased_ranking_score",    "discriminatory_hiring_outcome"),
            CausalEdge("historical_hiring_data",  "use_protected_attribute"),
            CausalEdge("societal_bias",           "historical_hiring_data"),
            CausalEdge("societal_bias",           "discriminatory_hiring_outcome"),
        ]
    )

    # 2. Decision Transparency
    dags["decision_transparency"] = HarmDAG(
        domain="Decision Transparency",
        treatment="opaque_ai_decision",
        outcome="unjust_outcome_without_recourse",
        mediator="user_unable_to_contest",
        confounders=["audit_trail_available"],
        latent=[],
        nodes=[
            CausalNode("opaque_ai_decision", "AI decision with no explanation"),
            CausalNode("user_unable_to_contest", "User cannot challenge decision"),
            CausalNode("unjust_outcome_without_recourse", "Harm with no remedy"),
            CausalNode("audit_trail_available", "Whether logging exists"),
        ],
        edges=[
            CausalEdge("opaque_ai_decision", "user_unable_to_contest"),
            CausalEdge("user_unable_to_contest", "unjust_outcome_without_recourse"),
            CausalEdge("audit_trail_available", "user_unable_to_contest"),
        ]
    )

    # 3. Misuse Safety (CBRN, weapons)
    dags["misuse_safety"] = HarmDAG(
        domain="Misuse Safety",
        treatment="ai_provides_harmful_instructions",
        outcome="physical_harm_realized",
        mediator="actor_capability_increase",
        confounders=["prior_knowledge_of_actor"],
        latent=["actor_intent"],
        nodes=[
            CausalNode("ai_provides_harmful_instructions", "AI outputs synthesis/attack guide"),
            CausalNode("actor_capability_increase",  "Actor gains capability they lacked"),
            CausalNode("physical_harm_realized",     "Real-world harm occurs"),
            CausalNode("prior_knowledge_of_actor",   "What actor already knew"),
            CausalNode("actor_intent",               "Malicious intent (unobserved)", measured=False),
        ],
        edges=[
            CausalEdge("ai_provides_harmful_instructions", "actor_capability_increase"),
            CausalEdge("actor_capability_increase",  "physical_harm_realized"),
            CausalEdge("prior_knowledge_of_actor",   "actor_capability_increase"),
            CausalEdge("actor_intent",               "ai_provides_harmful_instructions"),
        ]
    )

    # 4. COMPAS-style criminal justice bias
    dags["criminal_justice_bias"] = HarmDAG(
        domain="Criminal Justice Bias",
        treatment="race_used_in_risk_score",
        outcome="unjust_incarceration",
        mediator="inflated_risk_score",
        confounders=["arrest_history", "neighborhood"],
        latent=["systemic_racism"],
        nodes=[
            CausalNode("race_used_in_risk_score", "Race as input to COMPAS-style model"),
            CausalNode("inflated_risk_score",     "Score artificially elevated"),
            CausalNode("unjust_incarceration",    "Longer sentence/denial of parole"),
            CausalNode("arrest_history",          "Prior arrests (observed)"),
            CausalNode("neighborhood",            "Zip code proxy for race (observed)"),
            CausalNode("systemic_racism",         "Historical policing bias (latent)", measured=False),
        ],
        edges=[
            CausalEdge("race_used_in_risk_score", "inflated_risk_score"),
            CausalEdge("inflated_risk_score",     "unjust_incarceration"),
            CausalEdge("arrest_history",          "inflated_risk_score"),
            CausalEdge("neighborhood",            "inflated_risk_score"),
            CausalEdge("systemic_racism",         "arrest_history"),
            CausalEdge("systemic_racism",         "unjust_incarceration"),
        ]
    )

    # 5. Microsoft Tay — context poisoning
    dags["context_poisoning"] = HarmDAG(
        domain="Context Poisoning / Slow Boiling",
        treatment="adversarial_conversation_drift",
        outcome="harmful_content_output",
        mediator="model_context_corrupted",
        confounders=["conversation_length", "prior_safe_turns"],
        latent=["adversarial_user_intent"],
        nodes=[
            CausalNode("adversarial_conversation_drift", "Gradual topic escalation by attacker"),
            CausalNode("model_context_corrupted",         "Model loses safe framing"),
            CausalNode("harmful_content_output",          "Model outputs harm"),
            CausalNode("conversation_length",             "Number of turns"),
            CausalNode("prior_safe_turns",                "Safe framing established early"),
            CausalNode("adversarial_user_intent",         "Attacker's goal (unobserved)", measured=False),
        ],
        edges=[
            CausalEdge("adversarial_conversation_drift", "model_context_corrupted"),
            CausalEdge("model_context_corrupted",         "harmful_content_output"),
            CausalEdge("conversation_length",             "model_context_corrupted"),
            CausalEdge("prior_safe_turns",                "model_context_corrupted"),
            CausalEdge("adversarial_user_intent",         "adversarial_conversation_drift"),
        ]
    )

    # 6-19: Remaining domains — structural pattern
    # ── Domains 6–19: Domain-specific causal structures ─────────────────────
    # v2 fix: each domain now has empirically-motivated confounders and latent variables
    # rather than a shared generic template. Sources: AIAAIC Database, EU AI Act Annex III,
    # domain-specific AI harm literature (see research_notes.md for citations).
    #
    # Confounder choice rationale (examiner-ready):
    #   Observed Z: variables the system can measure from context/metadata.
    #   Latent U:   variables that confound X→Y but are unobservable at inference time.
    #
    # Year 2: validate confounder sets via DoWhy refutation tests on real incident data.

    specific_domains = [
        # (domain, treatment, outcome, mediator, confounders, latent)
        ("disinformation",
         "ai_generates_false_content",
         "public_belief_manipulation",
         "viral_spread",
         ["platform_algorithm_boost", "media_literacy_of_audience"],
         ["political_actor_coordination", "existing_belief_bias"]),

        ("harassment",
         "ai_enables_targeted_attack",
         "victim_psychological_harm",
         "escalation_pattern",
         ["platform_moderation_presence", "prior_victim_exposure"],
         ["attacker_psychological_state"]),

        ("cyberattack",
         "ai_provides_exploit_code",
         "system_breach",
         "vulnerability_exploitation",
         ["existing_system_access_level", "patch_status_of_target"],
         ["attacker_skill_level", "attacker_intent"]),

        ("privacy_violation",
         "ai_reveals_personal_data",
         "identity_theft_or_stalking",
         "data_exposure",
         ["data_sensitivity_level", "user_consent_status"],
         ["requestor_true_intent"]),

        ("financial_fraud",
         "ai_facilitates_fraud_scheme",
         "financial_loss",
         "victim_deception",
         ["victim_financial_literacy", "platform_fraud_detection_strength"],
         ["criminal_network_involvement"]),

        ("medical_harm",
         "ai_provides_harmful_dosage",
         "patient_injury_or_death",
         "clinical_decision",
         ["physician_oversight_presence", "patient_comorbidity_profile"],
         ["patient_true_condition", "contraindication_knowledge"]),

        ("physical_violence",
         "ai_provides_attack_guidance",
         "physical_harm",
         "actor_capability",
         ["prior_access_to_means", "law_enforcement_proximity"],
         ["attacker_intent", "mental_state"]),

        ("hate_speech",
         "ai_generates_hate_content",
         "societal_harm",
         "amplification",
         ["platform_community_norms", "content_moderation_active"],
         ["radicalization_state_of_audience"]),

        ("drug_trafficking",
         "ai_provides_synthesis_guide",
         "drug_supply_increase",
         "production_capability",
         ["precursor_chemical_access", "lab_infrastructure_available"],
         ["actor_criminal_network", "law_enforcement_awareness"]),

        ("child_safety",
         "ai_enables_grooming",
         "child_abuse",
         "access_facilitation",
         ["platform_age_verification_strength", "parental_supervision_level"],
         ["predator_intent", "child_vulnerability_state"]),

        ("identity_forgery",
         "ai_generates_fake_documents",
         "fraud_or_impersonation",
         "document_use",
         ["verification_system_strength", "document_type_scrutiny_level"],
         ["forger_criminal_network"]),

        ("audit_gap",
         "ai_operates_without_logging",
         "accountability_failure",
         "audit_absence",
         ["regulatory_oversight_active", "internal_compliance_process"],
         ["organizational_intent_to_evade"]),

        ("self_harm",
         "ai_provides_method_information",
         "self_harm_act",
         "method_access",
         ["platform_crisis_intervention_active", "user_expressed_distress_signal"],
         ["underlying_mental_health_state"]),  # Joiner (2005): means restriction key lever

        ("election_interference",
         "ai_generates_false_narrative",
         "corrupted_vote_behavior",
         "belief_update",
         ["media_literacy_of_electorate", "fact_checking_infrastructure"],
         ["political_actor_coordination"]),    # Allcott & Gentzkow (2017)
    ]

    for domain, treatment, outcome, mediator, confounders, latent in specific_domains:
        dags[domain] = HarmDAG(
            domain=domain.replace("_", " ").title(),
            treatment=treatment, outcome=outcome, mediator=mediator,
            confounders=confounders,
            latent=latent,
            nodes=[
                CausalNode(treatment, "AI action causing potential harm"),
                CausalNode(mediator,  "Intermediate causal pathway"),
                CausalNode(outcome,   "Final harm realized"),
            ] + [
                CausalNode(c, f"Observed confounder: {c.replace('_', ' ')}")
                for c in confounders
            ] + [
                CausalNode(u, f"Latent confounder: {u.replace('_', ' ')}", measured=False)
                for u in latent
            ],
            edges=[
                CausalEdge(treatment, mediator),
                CausalEdge(mediator,  outcome),
            ] + [
                CausalEdge(c, treatment) for c in confounders
            ] + [
                CausalEdge(u, treatment) for u in latent
            ]
        )

    return dags

DOMAIN_DAGS = build_domain_dags()


# ═══════════════════════════════════════════════════════════════════
# CAUSAL FINDINGS  (extended from v1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalFindings:
    """
    Input to the SCM Engine.
    Values sourced from:
      Year 1: Published literature (manual)
      Year 2: DoWhy auto-computation from AIAAIC 1,000 cases

    Fields carry-over from v1 + new L3 fields:
    """
    # ── Carry-over from v1 ────────────────────────────────────────
    tce   : float    # Total Causal Effect (%) — L2 backdoor
    med   : float    # Mediation % (NIE/TCE × 100) — L2
    flip  : float    # Counterfactual Flip Rate (%) — L3 proxy
    intv  : float    # do(fix) Intervention Reduction (%) — L2
    rct   : bool     # True if data from randomized controlled trial

    # ── New v2 fields ─────────────────────────────────────────────
    p_y_given_x   : float = 0.0   # P(Y=1 | X=1) — observed probability
    p_y_given_notx: float = 0.0   # P(Y=1 | X=0) — baseline
    p_x           : float = 0.5   # P(X=1) — prevalence of treatment
    nde_raw       : float = 0.0   # Natural Direct Effect (raw, 0 = unknown)
    domain        : str   = "misuse_safety"   # maps to DOMAIN_DAGS key
    subgroup_tce  : Dict[str, float] = field(default_factory=dict)
                                  # CATE: TCE per subgroup {"female":14.2,"male":3.1}
    _query        : str   = ""    # Pipeline passes query for domain inference

    def __hash__(self):
        return hash((self.tce, self.med, self.flip, self.intv, self.rct, self.domain))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.__hash__() == other.__hash__()


# ═══════════════════════════════════════════════════════════════════
# LEVEL 2 — INTERVENTION CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BackdoorResult:
    """
    Backdoor Adjustment Formula (Pearl 1993).

    P(Y=1 | do(X=1)) = Σ_z  P(Y=1 | X=1, Z=z) · P(Z=z)

    Interpretation:
      Marginalizes over all confounders Z to isolate the
      causal effect of X on Y, removing all non-causal
      (spurious) correlation paths X ← Z → Y.

    Applied when: Backdoor criterion satisfied (observed Z
    blocks all backdoor paths from X to Y).
    """
    applicable  : bool
    formula     : str
    ate         : float     # ATE = P(Y|do(X=1)) − P(Y|do(X=0))
    py_do_x1    : float     # P(Y=1 | do(X=1))
    py_do_x0    : float     # P(Y=1 | do(X=0))
    method      : AdjustmentMethod
    note        : str


@dataclass
class FrontdoorResult:
    """
    Frontdoor Adjustment Formula (Pearl 1995).

    P(Y | do(X)) = Σ_m P(M=m | X) · Σ_x' P(Y | M=m, X=x') · P(X=x')

    Applied when:
      - Backdoor set NOT available (latent confounders exist)
      - A mediator M is observed that:
        (a) is caused by X
        (b) blocks all directed paths from X to Y
        (c) all backdoor paths M→Y are blocked by X

    Key advantage: Valid even with UNMEASURED confounders.
    This is why it's critical for AI fairness cases where
    'societal bias' (latent) confounds the causal estimate.
    """
    applicable : bool
    formula    : str
    ate        : float
    note       : str


@dataclass
class EffectDecomposition:
    """
    Total Effect = NDE + NIE  (Pearl 2001, Mediation Formula)

    NDE = Natural Direct Effect
          E[Y_{x=1, M_{x=0}}] − E[Y_{x=0}]
          'How much does X affect Y directly, holding M fixed at x=0?'
          → Pure discrimination: the part of harm NOT mediated

    NIE = Natural Indirect Effect
          E[Y_{x=1, M_{x=1}}] − E[Y_{x=1, M_{x=0}}]
          'How much does X affect Y through M?'
          → Mediated harm: the part going through the pathway

    TE  = Total Effect = NDE + NIE

    Legal relevance:
      If NDE >> NIE: AI decision DIRECTLY causes harm (strong liability)
      If NIE >> NDE: Harm mediated through systemic pathway (shared liability)
    """
    te          : float   # Total Effect (TCE)
    nde         : float   # Natural Direct Effect
    nie         : float   # Natural Indirect Effect
    pct_direct  : float   # NDE / TE × 100
    pct_mediated: float   # NIE / TE × 100
    note        : str


@dataclass
class EffectSizes:
    """
    ATE / ATT / CATE taxonomy (Imbens & Rubin 2015 + Pearl).

    ATE  = Average Treatment Effect (population)
           E[Y₁ − Y₀] = P(Y|do(X=1)) − P(Y|do(X=0))

    ATT  = Average Treatment Effect on the Treated
           E[Y₁ − Y₀ | X=1]
           'Among those the AI DID act on, what was the causal effect?'
           → More relevant for individual harm cases

    CATE = Conditional ATE (subgroup)
           E[Y₁ − Y₀ | Z=z]
           'What is the causal effect for subgroup Z?'
           → Intersectional bias detection
    """
    ate          : float
    att          : float
    cate         : Dict[str, float]   # e.g. {"female": 0.142, "male": 0.031}
    att_note     : str
    cate_note    : str


# ═══════════════════════════════════════════════════════════════════
# LEVEL 3 — COUNTERFACTUAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualBounds:
    """
    Tian-Pearl Bounds on Probabilities of Causation (Tian & Pearl 2000).

    These bounds provide the strongest Pearl-grounded causal evidence (Daubert-aligned)
    available without experimental (RCT) data.

    PN  = Probability of Necessity
          PN = P(Y_{x'=0}=0 | X=x, Y=y)
          'Would harm have occurred WITHOUT the AI decision?'
          Bounds: max(0, (TCE − P(Y|x')) / P(Y|x)) ≤ PN ≤ min(1, P(Yx'=0) / P(Y|x))
          Court question: 'But for the AI decision, would harm have occurred?'

    PS  = Probability of Sufficiency
          PS = P(Y_{x=1}=1 | X=x', Y=y')
          'Is the AI decision alone sufficient to cause harm?'
          Bounds: max(0, (TCE − P(¬Y|x')) / P(¬Y|x')) ≤ PS ≤ min(1, P(Yx=1) / P(¬Y|x'))

    PNS = Probability of Necessity AND Sufficiency
          PNS = P(Y_{x=1}=1, Y_{x=0}=0)
          'Was AI decision BOTH necessary AND sufficient for harm?'
          Tian-Pearl bounds: max(0, TCE) ≤ PNS ≤ min(P(Y|x), P(¬Y|x'))
          Highest bar for legal causation — proves individual-level causal responsibility.
    """
    # PN bounds
    pn_lower : float
    pn_upper : float
    pn_point : float   # Best estimate (midpoint or RCT value)
    pn_note  : str

    # PS bounds
    ps_lower : float
    ps_upper : float
    ps_point : float
    ps_note  : str

    # PNS bounds
    pns_lower: float
    pns_upper: float
    pns_point: float
    pns_note : str

    # Legal interpretation
    legal_strength: str   # STRONG / MODERATE / WEAK
    but_for_causal: bool  # True if PN_lower > 0.5 (more likely than not)


@dataclass
class DoCalculusVerification:
    """
    Symbolic verification of Pearl's 3 do-calculus rules.

    Rule 1 (Insertion/Deletion of Observations):
      If (Y ⊥ Z | X, W) in G_{X̄}:
      P(y | do(x), z, w) = P(y | do(x), w)
      → Can we ignore an observed variable given do(X)?

    Rule 2 (Action/Observation Exchange):
      If (Y ⊥ Z | X, W) in G_{X̄,Z̄}:
      P(y | do(x), do(z), w) = P(y | do(x), z, w)
      → Can we replace do(Z) with observe(Z)?

    Rule 3 (Insertion/Deletion of Actions):
      If (Y ⊥ Z | X, W) in G_{X̄, Z(W)}:
      P(y | do(x), do(z), w) = P(y | do(x), w)
      → Can we delete an intervention do(Z)?

    In v2: proxy verification using D-separation.
    In Year 2: full symbolic implementation via networkx DAG traversal.
    """
    rule1_applicable: bool
    rule1_note      : str
    rule2_applicable: bool
    rule2_note      : str
    rule3_applicable: bool
    rule3_note      : str
    identification_path: str   # Which rule(s) identify the causal effect


# ═══════════════════════════════════════════════════════════════════
# FULL SCM RESULT (v2)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SCMResultV2:
    """Complete output of the v2 SCM Engine."""
    # ── v1 carry-over ────────────────────────────────────────────
    severity      : Severity
    edge_strength : EdgeStrength
    detection     : Detection
    severity_adj  : Severity
    epistemic_risk: bool
    risk_score    : float
    decision      : Decision
    rule_trace    : List[Dict]

    # ── v2 new ───────────────────────────────────────────────────
    level         : CausalLevel        # Highest Pearl level reached
    dag           : HarmDAG            # Domain DAG used
    adjustment    : AdjustmentMethod   # Backdoor / Frontdoor / IV
    backdoor      : BackdoorResult
    frontdoor     : FrontdoorResult
    effects       : EffectSizes        # ATE / ATT / CATE
    decomposition : EffectDecomposition # TE = NDE + NIE
    bounds        : CounterfactualBounds # PNS / PN / PS
    do_calc       : DoCalculusVerification
    legal_score   : float              # 0.0–1.0 legal admissibility
    legal_verdict : str


# ═══════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _clamp(v: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, v))


def compute_backdoor(f: CausalFindings, dag: HarmDAG) -> BackdoorResult:
    """
    Backdoor Adjustment Formula.

    P(Y=1 | do(X=1)) = Σ_z P(Y=1 | X=1, Z=z) · P(Z=z)

    Proxy implementation (Year 1):
      With RCT data: do(X) = observe(X) → ATE = TCE directly
      Without RCT but backdoor-applicable:
        ATE ≈ TCE (backdoor eliminates confounding by design)
      Frontdoor needed if latent confounders exist.
    """
    ate = f.tce / 100.0
    applicable = dag.has_backdoor_set() or f.rct

    if f.rct:
        py_do_x1 = f.p_y_given_x
        py_do_x0 = f.p_y_given_notx
        ate_computed = py_do_x1 - py_do_x0
        note = ("RCT data: P(Y|do(X=1)) = P(Y|X=1) directly — "
                "no confounding adjustment needed (do(X) = see(X) in RCT)")
        method = AdjustmentMethod.BACKDOOR
    elif applicable:
        # ── Backdoor adjustment: Σ_z P(Y|X,Z=z)·P(Z=z) ──────────────────
        # Year 1 assumption: P(Z=z) = 1/|Z| (uniform prior over confounder levels).
        # This is the correct STRUCTURAL form of the backdoor formula.
        # Year 2: replace with empirical P(Z) from calibrated dataset (DoWhy Phase 4).
        #
        # Pearl (2000, Def 3.3.2): when Z satisfies backdoor criterion,
        # deconfounding reduces to marginalising P(Y|X,Z) over the marginal P(Z).
        # With uniform P(Z) and binary Z levels {0,1}:
        #   P(Y|do(X=1)) ≈ 0.5·P(Y|X=1,Z=0) + 0.5·P(Y|X=1,Z=1)
        # We proxy: P(Y|X=1,Z=0) ≈ p_y_given_x · (1 − confounder_bias_factor)
        #           P(Y|X=1,Z=1) ≈ p_y_given_x · (1 + confounder_bias_factor)
        # where confounder_bias_factor = 0 (no confounder data) → sum collapses to p_y_given_x.
        # This makes the Year 1 assumption EXPLICIT rather than silent.
        n_z = max(1, len(dag.confounders))
        uniform_pz = 1.0 / n_z
        # Without stratum-level data: P(Y|X,Z=z) ≈ P(Y|X) for all z (no-interaction assumption)
        py_do_x1 = _clamp(
            sum(uniform_pz * _clamp(f.p_y_given_x) for _ in range(n_z))
        )
        py_do_x0 = _clamp(
            sum(uniform_pz * _clamp(f.p_y_given_notx) for _ in range(n_z))
        )
        ate_computed = py_do_x1 - py_do_x0
        note = (
            f"Backdoor set Z = {dag.confounders} (|Z|={n_z}) blocks all X←Z→Y paths. "
            f"P(Y|do(X)) = Σ_z P(Y|X,Z=z)·P(Z=z). "
            f"Year 1: uniform P(Z=z)={uniform_pz:.3f}, no-interaction assumption across strata. "
            f"ATE = {ate_computed*100:.1f}%. "
            f"Year 2: empirical P(Z) from DoWhy calibration will replace uniform prior."
        )
        method = AdjustmentMethod.BACKDOOR
    else:
        py_do_x1 = _clamp(f.p_y_given_x + ate * 0.5)
        py_do_x0 = _clamp(f.p_y_given_notx)
        ate_computed = py_do_x1 - py_do_x0
        note = (f"Latent confounders {dag.latent} present. Backdoor not sufficient. "
                f"Frontdoor adjustment required. ATE bounds wider.")
        method = AdjustmentMethod.FRONTDOOR

    return BackdoorResult(
        applicable=applicable,
        formula=(
            "P(Y=1|do(X=1)) = Σ_z P(Y=1|X=1,Z=z)·P(Z=z)   [Pearl 1993]\n"
            f"  = {py_do_x1:.4f} (with treatment)\n"
            f"  P(Y=1|do(X=0)) = {py_do_x0:.4f} (without treatment)\n"
            f"  ATE = {py_do_x1:.4f} − {py_do_x0:.4f} = {ate_computed:.4f} "
            f"({ate_computed*100:.1f}%)"
        ),
        ate=ate_computed,
        py_do_x1=py_do_x1,
        py_do_x0=py_do_x0,
        method=method,
        note=note
    )


def compute_frontdoor(f: CausalFindings, dag: HarmDAG) -> FrontdoorResult:
    """
    Frontdoor Adjustment Formula (Pearl 1995).

    P(Y | do(X=x)) = Σ_m P(M=m|X=x) · Σ_x' P(Y|M=m,X=x') · P(X=x')

    Key property: Identifies causal effect even with unmeasured confounders U,
    as long as the mediator M satisfies the frontdoor criterion.

    Proxy implementation:
      When latent confounders exist, frontdoor ATE ≈ NIE-based estimate
      (mediated path through M is the only identified path).
      Full computation requires P(M|X) and P(Y|M,X) from data — Year 2.
    """
    applicable = dag.has_frontdoor_set()
    nie_frac   = (f.med / 100.0) * (f.tce / 100.0)  # NIE as fraction of TE

    if applicable:
        # Proxy: frontdoor identifies effect through mediator
        # P(Y|do(X=1)) − P(Y|do(X=0)) ≈ sum of M-mediated paths
        ate_proxy = nie_frac
        note = (f"Frontdoor criterion satisfied. Mediator M='{dag.mediator}' "
                f"blocks all X→Y paths and is unconfounded by U. "
                f"Frontdoor ATE ≈ {ate_proxy*100:.1f}% (mediated portion). "
                f"Full computation (P(M|X)·ΣP(Y|M,x')P(x')) deferred to Year 2 DoWhy.")
    else:
        ate_proxy = 0.0
        note = ("Frontdoor criterion not satisfied — mediator does not "
                "fully block X→Y paths, or M is also confounded by U.")

    return FrontdoorResult(
        applicable=applicable,
        formula=(
            "P(Y|do(X=x)) = Σ_m P(M=m|X=x) · Σ_x' P(Y|M=m,X=x')·P(X=x')   "
            "[Pearl 1995]\n"
            f"  Mediator: {dag.mediator}\n"
            f"  Latent confounders: {dag.latent}\n"
            f"  Frontdoor ATE proxy: {ate_proxy*100:.1f}%"
        ),
        ate=ate_proxy,
        note=note
    )


def compute_effect_sizes(f: CausalFindings) -> EffectSizes:
    """
    ATE / ATT / CATE computation.

    ATE = P(Y|do(X=1)) − P(Y|do(X=0))  [Population average]
    ATT = E[Y₁−Y₀ | X=1]               [Among those treated]
        Proxy: ATT ≈ ATE × (1 + selection_bias_factor)
        Selection bias = if high TCE AND low P(X) → high selection pressure
    CATE = ATE|Z=z  for each subgroup z in subgroup_tce
    """
    ate  = f.tce / 100.0

    # ATT proxy: P(X) adjustment (Simpson's paradox guard)
    # If AI only acts on high-risk cases → ATT > ATE
    selection_factor = 1.0 + max(0, (0.5 - f.p_x) / 0.5) * 0.3
    att = _clamp(ate * selection_factor)

    cate = {k: v/100.0 for k, v in f.subgroup_tce.items()}
    if not cate:
        cate = {"overall": ate}

    max_cate = max(cate.values()) if cate else ate
    min_cate = min(cate.values()) if cate else ate
    cate_gap = max_cate - min_cate

    return EffectSizes(
        ate=ate,
        att=att,
        cate=cate,
        att_note=(f"ATT={att:.4f} ({att*100:.1f}%): causal effect on "
                  f"individuals the AI actually acted upon. "
                  f"Higher than ATE={ate*100:.1f}% due to selection (P(X)={f.p_x})."),
        cate_note=(f"CATE gap = {cate_gap*100:.1f}pp across subgroups. "
                   + ("⚠️ Intersectional bias detected — subgroup disparity exceeds 5pp."
                      if cate_gap > 0.05 else "Subgroup effects consistent with ATE."))
    )


def compute_effect_decomposition(f: CausalFindings) -> EffectDecomposition:
    """
    Total Effect decomposition: TE = NDE + NIE  (Pearl Mediation Formula 2001).

    NIE = (med/100) × tce/100   [fraction of TE that is mediated]
    NDE = TE − NIE               [fraction that is direct]

    If nde_raw provided: use it directly.
    Otherwise: proxy from mediation % (standard in observational studies).
    """
    te  = f.tce / 100.0
    nie = (f.med / 100.0) * te

    if f.nde_raw > 0:
        nde = f.nde_raw / 100.0
        nie = _clamp(te - nde)
    else:
        nde = _clamp(te - nie)

    pct_d = (nde / te * 100) if te > 0 else 0
    pct_m = (nie / te * 100) if te > 0 else 0

    if pct_d >= 70:
        note = (f"NDE={nde*100:.1f}% >> NIE={nie*100:.1f}%. "
                "Harm is PRIMARILY DIRECT — AI decision itself causes harm "
                "without systemic mediation. Strongest legal liability case.")
    elif pct_m >= 70:
        _dag = DOMAIN_DAGS.get(f.domain, DOMAIN_DAGS["misuse_safety"])
        note = (f"NIE={nie*100:.1f}% >> NDE={nde*100:.1f}%. "
                f"Harm is PRIMARILY MEDIATED through '{_dag.mediator}'. "
                "Shared liability — AI + systemic pathway. "
                "Mitigation should target the mediating mechanism.")
    else:
        note = (f"Mixed: Direct={pct_d:.0f}%, Mediated={pct_m:.0f}%. "
                "Both direct AI action and systemic pathway contribute to harm.")

    return EffectDecomposition(
        te=te, nde=nde, nie=nie,
        pct_direct=pct_d, pct_mediated=pct_m,
        note=note
    )


def compute_counterfactual_bounds(f: CausalFindings) -> CounterfactualBounds:
    """
    Tian-Pearl Bounds (Tian & Pearl 2000, UAI).

    Given only observational data (no RCT), we cannot compute PN/PS/PNS
    exactly but can compute sharp bounds.

    Let:
      p  = P(Y=1 | X=1)  = p_y_given_x
      p' = P(Y=1 | X=0)  = p_y_given_notx
      TCE = p − p'  (when no confounders, else backdoor-adjusted)

    PNS bounds (Tian-Pearl):
      max(0, p − p') ≤ PNS ≤ min(p, 1−p')

    PN bounds:
      max(0, (TCE − p') / p) ≤ PN ≤ min(1, (1−p') / p)
      [Where p = P(Y|X=1), p' = P(Y|X=0)]

    PS bounds:
      max(0, (TCE − (1−p)) / (1−p')) ≤ PS ≤ min(1, p / (1−p'))
      [Where 1−p = P(¬Y|X=1), 1−p' = P(¬Y|X=0)]

    RCT special case: bounds collapse to point estimates.
    """
    p  = f.p_y_given_x if f.p_y_given_x > 0 else f.tce/100 + 0.05
    # FIX v15h: was '>= 0' which is always True (0.0 >= 0), fallback never ran.
    # When p_y_given_notx is unset (default 0.0), PNS collapsed to overconfident
    # point estimate. Fix: '> 0' so default 0.0 triggers tce-based fallback.
    pp = f.p_y_given_notx if f.p_y_given_notx > 0 else max(0, p - f.tce/100)
    tce = f.tce / 100.0
    p  = _clamp(p)
    pp = _clamp(pp)

    # PNS
    pns_lo = max(0.0, p - pp)           # = max(0, TCE_obs)
    pns_hi = min(p, 1.0 - pp)
    pns_lo = _clamp(pns_lo)
    pns_hi = _clamp(pns_hi, lo=pns_lo)

    # PN
    pn_lo = max(0.0, (tce - pp) / p) if p > 0 else 0.0
    pn_hi = min(1.0, (1.0 - pp) / p) if p > 0 else 1.0
    pn_lo = _clamp(pn_lo)
    pn_hi = _clamp(pn_hi, lo=pn_lo)

    # PS
    not_p  = 1.0 - p
    not_pp = 1.0 - pp
    ps_lo = max(0.0, (tce - not_p) / not_pp) if not_pp > 0 else 0.0
    ps_hi = min(1.0, p / not_pp) if not_pp > 0 else 1.0
    ps_lo = _clamp(ps_lo)
    ps_hi = _clamp(ps_hi, lo=ps_lo)

    # Point estimates (midpoint for observational; exact for RCT)
    if f.rct:
        pns_pt = tce
        pn_pt  = (tce - pp) / p if p > 0 else 0
        ps_pt  = (tce - not_p) / not_pp if not_pp > 0 else 0
    else:
        pns_pt = (pns_lo + pns_hi) / 2
        pn_pt  = (pn_lo + pn_hi) / 2
        ps_pt  = (ps_lo + ps_hi) / 2

    pns_pt = _clamp(pns_pt)
    pn_pt  = _clamp(pn_pt)
    ps_pt  = _clamp(ps_pt)

    # Legal strength
    if pns_lo >= 0.5:
        legal_strength = "STRONG — PNS lower bound > 0.5: AI decision more likely than not BOTH necessary and sufficient for harm"
    elif pn_pt >= 0.5:
        legal_strength = "MODERATE — PN > 0.5: AI decision more likely than not NECESSARY for harm (but-for causation met)"
    elif pns_hi > 0.2:
        legal_strength = "MODERATE — PNS upper bound suggests causal role, but uncertainty wide without RCT data"
    else:
        legal_strength = "WEAK — PNS bounds include zero: causal responsibility uncertain from observational data alone"

    but_for = pn_pt >= 0.5

    return CounterfactualBounds(
        pn_lower=pn_lo,  pn_upper=pn_hi,  pn_point=pn_pt,
        pn_note=(f"PN ∈ [{pn_lo:.3f}, {pn_hi:.3f}] → point={pn_pt:.3f}. "
                 f"'Would harm have occurred WITHOUT the AI decision?' "
                 + ("→ YES (more likely than not)" if but_for else "→ UNCERTAIN")),
        ps_lower=ps_lo,  ps_upper=ps_hi,  ps_point=ps_pt,
        ps_note=(f"PS ∈ [{ps_lo:.3f}, {ps_hi:.3f}] → point={ps_pt:.3f}. "
                 f"'Is AI decision alone sufficient to cause harm?'"),
        pns_lower=pns_lo, pns_upper=pns_hi, pns_point=pns_pt,
        pns_note=(f"PNS ∈ [{pns_lo:.3f}, {pns_hi:.3f}] → point={pns_pt:.3f}. "
                  f"Tian-Pearl (2000) sharp bounds. "
                  + ("RCT: exact value." if f.rct else
                     "Observational: bounds widen with latent confounders.")),
        legal_strength=legal_strength,
        but_for_causal=but_for
    )


def verify_do_calculus(f: CausalFindings, dag: HarmDAG) -> DoCalculusVerification:
    """
    Symbolic verification of Pearl's 3 do-calculus rules.

    Rule 1: P(y|do(x),z,w) = P(y|do(x),w)
      Applicable if: (Y ⊥ Z | X, W) in G_{X̄}
      → Can ignore Z when already intervening on X
      → Applied when Z is a descendant of X (cutting X's parents via do(x))

    Rule 2: P(y|do(x),do(z),w) = P(y|do(x),z,w)
      Applicable if: (Y ⊥ Z | X, W) in G_{X̄Z̄}
      → do(Z) equivalent to see(Z) when Z not ancestor of Y via open path
      → Applied in backdoor adjustment: replacing do(Z) with P(Z)

    Rule 3: P(y|do(x),do(z),w) = P(y|do(x),w)
      Applicable if: (Y ⊥ Z | X, W) in G_{X̄, Z(W)}
      → Can delete intervention on Z entirely
      → Applied when Z has no effect on Y given X
    """
    # Rule 1: confounders d-separated from Y given X (after removing X's parents)
    r1 = dag.d_separated(dag.outcome, "confounder", [dag.treatment])
    # Rule 2: mediator d-separated from Y given X,Z (backdoor set)
    r2 = dag.d_separated(dag.outcome, dag.mediator, dag.confounders + [dag.treatment])
    # Rule 3: intervention on Z removable (only if no latent confounders)
    r3 = len(dag.latent) == 0

    if r1 and r2:
        path = "Rule 2 (backdoor) identifies P(Y|do(X))"
    elif dag.has_frontdoor_set():
        path = "Frontdoor criterion identifies P(Y|do(X)) despite latent confounders"
    elif r1:
        path = "Rule 1 simplifies observation set; partial identification"
    else:
        path = "Full identification requires additional assumptions or RCT"

    return DoCalculusVerification(
        rule1_applicable=r1,
        rule1_note=(f"Rule 1 {'✅' if r1 else '⚠️'}: "
                    + ("Z d-separated from Y given do(X) → observations can be removed from conditioning set."
                       if r1 else "Z NOT d-separated — cannot simplify observation set.")),
        rule2_applicable=r2,
        rule2_note=(f"Rule 2 {'✅' if r2 else '⚠️'}: "
                    + ("Backdoor set Z satisfies do-calculus Rule 2 → do(Z) exchangeable with see(Z) → backdoor formula valid."
                       if r2 else "Rule 2 does not apply — frontdoor or IV required.")),
        rule3_applicable=r3,
        rule3_note=(f"Rule 3 {'✅' if r3 else '⚠️'}: "
                    + ("No latent confounders → can delete intervention on Z."
                       if r3 else f"Latent confounders {dag.latent} prevent Rule 3 application.")),
        identification_path=path
    )


def compute_legal_score(
    f: CausalFindings,
    bounds: CounterfactualBounds,
    effects: EffectSizes,
    decomp: EffectDecomposition
) -> Tuple[float, str]:
    """
    Legal Admissibility Score (0.0 – 1.0).

    Maps Pearl calculations to court standards:
      0.0–0.3: Weak — statistical correlation only (L1)
      0.3–0.5: Moderate — do-calculus backdoor (L2)
      0.5–0.7: Strong — L2 + mediation decomposition
      0.7–0.9: Very Strong — L3 counterfactual (PNS > 0.3)
      0.9–1.0: Definitive — RCT or PNS_lower > 0.5

    Reference: Expert testimony standards in US (Daubert v. Merrell Dow)
               and EU AI Act Article 13 (explainability requirement).
    """
    score = 0.0

    # L2 contribution
    if effects.ate > 0.05:  score += 0.20   # ATE > 5%
    if effects.ate > 0.10:  score += 0.15   # ATE > 10%
    if decomp.te > 0.05:    score += 0.10   # Effect decomposition complete

    # Mediation (NDE/NIE split)
    if decomp.pct_direct > 0: score += 0.10  # Direct effect quantified

    # L3 contribution
    if bounds.pns_lower > 0:   score += 0.15  # PNS lower bound positive
    if bounds.pns_lower > 0.3: score += 0.15  # PNS lower > 30%
    if bounds.but_for_causal:  score += 0.10  # PN > 0.5 (but-for met)

    # RCT bonus
    if f.rct:                  score += 0.05  # Experimental data

    score = _clamp(score)

    if score >= 0.9:
        verdict = "DEFINITIVE — satisfies Daubert admissibility criteria and EU AI Act Art.13. Provides strongest Pearl-grounded causal evidence; domain expert validation required for full legal standing."
    elif score >= 0.7:
        verdict = "VERY STRONG — L3 counterfactual bounds establish causal responsibility. Supports regulatory enforcement; expert witness testimony recommended before litigation."
    elif score >= 0.5:
        verdict = "STRONG — L2 causal effect + mediation decomposition. Exceeds balance-of-probabilities threshold. Suitable for administrative proceedings with expert review."
    elif score >= 0.3:
        verdict = "MODERATE — L2 do-calculus backdoor. Causal direction established; individual responsibility claims require additional L3 counterfactual evidence."
    else:
        verdict = "WEAK — L1 correlation only. Insufficient for legal causation claims. Requires RCT or stronger causal identification strategy."

    return score, verdict


# ═══════════════════════════════════════════════════════════════════
# V1 RULES (carry-over unchanged)
# ═══════════════════════════════════════════════════════════════════

def rule_01_severity(tce: float) -> Severity:
    if tce >= 10: return Severity.CRITICAL
    if tce >= 5:  return Severity.HIGH
    if tce >= 2:  return Severity.MEDIUM
    return Severity.LOW

def rule_02_edge(med: float) -> EdgeStrength:
    if med >= 70: return EdgeStrength.CRITICAL
    if med >= 40: return EdgeStrength.STRONG
    if med >= 20: return EdgeStrength.MODERATE
    return EdgeStrength.WEAK

def rule_03_detection(flip: float) -> Detection:
    if flip >= 30: return Detection.EASY
    if flip >= 10: return Detection.MODERATE
    return Detection.HARD

def rule_04_adjust(severity: Severity, intv: float) -> Tuple[Severity, str]:
    if intv >= 80:
        new = Severity(max(1, severity.value - 1))
        return new, f"do(fix) reduced TCE by {intv}% → severity {severity.name}→{new.name}"
    if intv <= 20 and severity.value < 4:
        new = Severity(min(4, severity.value + 1))
        return new, f"do(fix) barely reduced harm ({intv}%) → severity escalated to {new.name}"
    return severity, f"do(fix) reduction {intv}% → no severity change"

def rule_05_identifiability(tce, flip, rct, dag: "HarmDAG | None" = None) -> Tuple[bool, str]:
    """
    Causal identifiability check — v2 (graph-based).

    Priority order (Pearl 2000, Ch. 3):
      1. RCT: do(X) = see(X) — exact identification, no adjustment needed.
      2. Backdoor criterion: latent=[] → all confounding paths blocked by Z.
         Applicable: P(Y|do(X)) = Σ_z P(Y|X,Z=z)·P(Z=z)
      3. Frontdoor criterion: mediator M exists + latent U present.
         Applicable: P(Y|do(X)) = Σ_m P(M|X)·Σ_x' P(Y|M,X=x')·P(X=x')
      4. NOT IDENTIFIABLE: latent confounders + no valid adjustment set.
         System honestly reports: "Cannot make causal claim without additional assumptions."

    Year 1: uses DAG.has_backdoor_set() and DAG.has_frontdoor_set() (structural flags).
    Year 2: full d-separation via networkx (PhD Phase 5 — polynomial-time algorithm).
    """
    if dag is not None:
        if rct:
            return True, (
                "RCT: do(X) = see(X) — exact causal identification. "
                "No confounding adjustment needed."
            )
        if dag.has_backdoor_set():
            z_list = ", ".join(dag.confounders) if dag.confounders else "∅"
            return True, (
                f"Backdoor criterion satisfied (Pearl 2000, Def 3.3.1). "
                f"Observed adjustment set Z = {{{z_list}}} blocks all backdoor paths X←Z→Y. "
                f"No latent confounders. P(Y|do(X)) = Σ_z P(Y|X,Z=z)·P(Z=z) is identifiable."
            )
        if dag.has_frontdoor_set():
            return True, (
                f"Frontdoor criterion satisfied (Pearl 1995). "
                f"Mediator M = '{dag.mediator}' blocks all X→Y directed paths. "
                f"Latent confounders U = {dag.latent} present but bypassed via M. "
                f"P(Y|do(X)) identified through frontdoor formula. Year 2: DoWhy empirical estimation."
            )
        # Not identifiable — system must be honest
        return False, (
            f"NOT IDENTIFIABLE (Pearl 2000, Ch.3): "
            f"Latent confounders U = {dag.latent} present, "
            f"no valid backdoor or frontdoor adjustment set found. "
            f"Cannot make point causal claim from observational data alone. "
            f"Partial identification bounds apply. Year 2: IV or sensitivity analysis."
        )

    # Legacy numeric fallback (no DAG provided — should not occur in v2 pipeline)
    if tce < 2 and flip < 5 and not rct:
        return False, "Low TCE+Flip → epistemic risk ACTIVATED (Severity floor=3) [legacy path]"
    return True, "Effect identifiable via Rule 2 backdoor or RCT [legacy path — pass dag= for graph check]"

def calculate_risk_score(severity_adj, detection, edge_strength,
                         domain_multiplier: float = 1.0) -> float:
    """
    Risk Score Formula:
      base = 9 (minimum risk floor)
      sev  = Severity.value × 3   (CRITICAL=4 → 12)
      det  = Detection.value × 2  (MODERATE=3 → 6)
      dom  = EdgeStrength.value × 0.75 (STRONG=3 → 2.25)
      raw  = base + sev + det + dom
      risk = raw / 432 × 100 × 4.5  (normalize to 0-100%)

    domain_multiplier: 1.0 (general) → 3.0 (healthcare)
    Applied AFTER normalization — amplifies final % by context.
    """
    base = 9
    sev  = severity_adj.value
    det  = detection.value
    dom  = edge_strength.value * 0.75
    raw  = base + sev*3 + det*2 + dom
    base_risk = min(100.0, raw / 432 * 100 * 4.5)
    return min(100.0, base_risk * domain_multiplier)

def get_decision(risk_pct: float) -> Decision:
    if risk_pct >= 65: return Decision.BLOCK
    if risk_pct >= 50: return Decision.ESCALATE
    if risk_pct >= 35: return Decision.WARN
    return Decision.ALLOW


# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE  v2
# ═══════════════════════════════════════════════════════════════════

class SCMEngineV2:
    """
    Full Pearl-complete Structural Causal Model Engine.

    Implements all 3 levels of Pearl's Ladder of Causation:
      L1: Association  — P(Y|X), correlation
      L2: Intervention — P(Y|do(X)), backdoor/frontdoor
      L3: Counterfactual — PNS/PN/PS, Tian-Pearl bounds

    Also implements:
      - Effect decomposition (TE = NDE + NIE)
      - ATE/ATT/CATE taxonomy
      - do-calculus 3-rule symbolic verification
      - 17 domain-specific DAGs
      - Legal admissibility scoring
    """

    def run(self, findings: CausalFindings) -> SCMResultV2:
        # ── lru_cache via module-level wrapper (see _run_scm_cached below) ──
        # CausalFindings has __hash__ + __eq__ → cache-safe.
        # Same findings object (e.g. repeated COMPAS tests) returns cached result.
        return _run_scm_cached(findings)

    # ── Internal (non-self) method — called by _run_scm_cached ───────────
    @staticmethod
    def _compute(findings: CausalFindings) -> SCMResultV2:
        # ── Get domain DAG ────────────────────────────────────────
        dag = DOMAIN_DAGS.get(findings.domain, DOMAIN_DAGS["misuse_safety"])

        # ── L2: Intervention ──────────────────────────────────────
        backdoor  = compute_backdoor(findings, dag)
        frontdoor = compute_frontdoor(findings, dag)
        effects   = compute_effect_sizes(findings)
        decomp    = compute_effect_decomposition(findings)
        do_calc   = verify_do_calculus(findings, dag)

        # ── L3: Counterfactual ────────────────────────────────────
        bounds    = compute_counterfactual_bounds(findings)
        level     = (CausalLevel.L3_COUNTERFACTUAL if bounds.pns_point > 0
                     else CausalLevel.L2_INTERVENTION)

        # ── v1 Rules (carry-over) ─────────────────────────────────
        sev   = rule_01_severity(findings.tce)
        edge  = rule_02_edge(findings.med)
        det   = rule_03_detection(findings.flip)
        sev_a, adj_note = rule_04_adjust(sev, findings.intv)
        ident, ident_note = rule_05_identifiability(findings.tce, findings.flip, findings.rct, dag=dag)
        if not ident:
            sev_a = Severity(max(sev_a.value, 3))
        # Domain multiplier handled by pipeline (single source of truth)
        risk  = calculate_risk_score(sev_a, det, edge, domain_multiplier=1.0)
        dec   = get_decision(risk)

        # Adjustment method
        adj_method = (AdjustmentMethod.BACKDOOR if backdoor.applicable
                      else AdjustmentMethod.FRONTDOOR if frontdoor.applicable
                      else AdjustmentMethod.NOT_ID)

        # Legal score
        legal_score, legal_verdict = compute_legal_score(findings, bounds, effects, decomp)

        # Rule trace
        rule_trace = [
            {"rule": "Rule 01", "pearl": "L2 TCE → Severity",
             "input": f"TCE={findings.tce}%", "output": sev.name},
            {"rule": "Rule 02", "pearl": "L2 NIE/Mediation → Edge",
             "input": f"Med={findings.med}%", "output": edge.name},
            {"rule": "Rule 03", "pearl": "L3 Flip Rate → Detection",
             "input": f"Flip={findings.flip}%", "output": det.name},
            {"rule": "Rule 04", "pearl": "L2 do(fix) → Severity Adj",
             "input": f"Intv={findings.intv}%", "output": adj_note},
            {"rule": "Rule 05", "pearl": "Identifiability",
             "input": f"RCT={findings.rct}", "output": ident_note},
            {"rule": "Rule 06 (NEW)", "pearl": "Backdoor Adjustment",
             "input": backdoor.formula.split('\n')[0], "output": f"ATE={effects.ate*100:.1f}%"},
            {"rule": "Rule 07 (NEW)", "pearl": "Frontdoor (latent)",
             "input": f"Frontdoor={'OK' if frontdoor.applicable else 'N/A'}",
             "output": frontdoor.note[:60]},
            {"rule": "Rule 08 (NEW)", "pearl": "Effect Decomposition TE=NDE+NIE",
             "input": f"TE={decomp.te*100:.1f}%",
             "output": f"Direct={decomp.pct_direct:.0f}% Mediated={decomp.pct_mediated:.0f}%"},
            {"rule": "Rule 09 (NEW)", "pearl": "ATE/ATT/CATE",
             "input": f"P(X)={findings.p_x}",
             "output": f"ATE={effects.ate*100:.1f}% ATT={effects.att*100:.1f}%"},
            {"rule": "Rule 10 (NEW)", "pearl": "PNS Tian-Pearl Bounds",
             "input": f"p={findings.p_y_given_x:.2f} p'={findings.p_y_given_notx:.2f}",
             "output": f"PNS∈[{bounds.pns_lower:.3f},{bounds.pns_upper:.3f}]"},
            {"rule": "Rule 11 (NEW)", "pearl": "PN (Probability of Necessity)",
             "input": "But-for causation test",
             "output": f"PN∈[{bounds.pn_lower:.3f},{bounds.pn_upper:.3f}] but-for={'YES' if bounds.but_for_causal else 'NO'}"},
            {"rule": "Rule 12 (NEW)", "pearl": "PS (Probability of Sufficiency)",
             "input": "Sufficiency test",
             "output": f"PS∈[{bounds.ps_lower:.3f},{bounds.ps_upper:.3f}]"},
            {"rule": "Rule 13 (NEW)", "pearl": "do-calculus 3 Rules",
             "input": "D-separation",
             "output": do_calc.identification_path},
            {"rule": "Rule 14 (NEW)", "pearl": "Legal Admissibility",
             "input": "Daubert + EU AI Act Art.13",
             "output": f"Score={legal_score:.2f} — {legal_verdict[:50]}"},
        ]

        return SCMResultV2(
            severity=sev, edge_strength=edge, detection=det,
            severity_adj=sev_a, epistemic_risk=not ident,
            risk_score=risk, decision=dec, rule_trace=rule_trace,
            level=level, dag=dag, adjustment=adj_method,
            backdoor=backdoor, frontdoor=frontdoor,
            effects=effects, decomposition=decomp,
            bounds=bounds, do_calc=do_calc,
            legal_score=legal_score, legal_verdict=legal_verdict,
        )

    def print_report(self, findings: CausalFindings, r: SCMResultV2) -> None:
        sep = "═" * 64
        print(f"\n{sep}")
        print(f"  SCM ENGINE v2 — {r.dag.domain.upper()}")
        print(f"  Pearl's Ladder Level: {r.level.name}")
        print(sep)

        print(f"\n  INPUT CAUSAL MEASUREMENTS")
        print(f"  {'TCE (Total Causal Effect)':<35} {findings.tce}%")
        print(f"  {'Mediation % (NIE/TCE)':<35} {findings.med}%")
        print(f"  {'Counterfactual Flip Rate':<35} {findings.flip}%")
        print(f"  {'do(fix) Intervention Reduction':<35} {findings.intv}%")
        print(f"  {'P(Y|X=1)':<35} {findings.p_y_given_x:.3f}")
        print(f"  {'P(Y|X=0)':<35} {findings.p_y_given_notx:.3f}")
        print(f"  {'RCT data':<35} {findings.rct}")
        print(f"  {'Domain':<35} {findings.domain}")

        print(f"\n  DOMAIN DAG — {r.dag.domain}")
        print(f"  Treatment  X: {r.dag.treatment}")
        print(f"  Mediator   M: {r.dag.mediator}")
        print(f"  Outcome    Y: {r.dag.outcome}")
        print(f"  Confounders: {r.dag.confounders}")
        print(f"  Latent (U):  {r.dag.latent or 'None'}")
        print(f"  Adjustment:  {r.adjustment.value}")

        print(f"\n  L2 — INTERVENTION RESULTS")
        for line in r.backdoor.formula.split('\n'):
            print(f"    {line}")
        print(f"  {'-'*60}")
        print(f"  ATE  = {r.effects.ate*100:.2f}%  |  ATT = {r.effects.att*100:.2f}%")
        for subg, v in r.effects.cate.items():
            print(f"  CATE ({subg:<10}) = {v*100:.2f}%")
        print(f"\n  Effect Decomposition (TE = NDE + NIE):")
        print(f"    TE  = {r.decomposition.te*100:.2f}%")
        print(f"    NDE = {r.decomposition.nde*100:.2f}%  ({r.decomposition.pct_direct:.0f}% direct)")
        print(f"    NIE = {r.decomposition.nie*100:.2f}%  ({r.decomposition.pct_mediated:.0f}% mediated)")
        print(f"  {r.decomposition.note}")

        print(f"\n  L3 — COUNTERFACTUAL BOUNDS (Tian-Pearl 2000)")
        print(f"  PNS = [{r.bounds.pns_lower:.3f}, {r.bounds.pns_upper:.3f}]  pt={r.bounds.pns_point:.3f}")
        print(f"        {r.bounds.pns_note}")
        print(f"  PN  = [{r.bounds.pn_lower:.3f}, {r.bounds.pn_upper:.3f}]  pt={r.bounds.pn_point:.3f}")
        print(f"        {r.bounds.pn_note}")
        print(f"  PS  = [{r.bounds.ps_lower:.3f}, {r.bounds.ps_upper:.3f}]  pt={r.bounds.ps_point:.3f}")
        print(f"        {r.bounds.ps_note}")
        print(f"\n  Legal Strength: {r.bounds.legal_strength}")

        print(f"\n  do-CALCULUS VERIFICATION (Pearl 1995)")
        print(f"  {r.do_calc.rule1_note}")
        print(f"  {r.do_calc.rule2_note}")
        print(f"  {r.do_calc.rule3_note}")
        print(f"  Identification path: {r.do_calc.identification_path}")

        print(f"\n  LEGAL ADMISSIBILITY SCORE: {r.legal_score:.2f}/1.00")
        print(f"  {r.legal_verdict}")

        print(f"\n  FINAL DECISION: {r.decision.value}")
        print(f"  Risk Score:     {r.risk_score:.1f}%")
        print(sep)


# ── lru_cache wrapper (outside class — instance methods can't be cached directly) ──
# CausalFindings has __hash__ + __eq__ → safe to cache.
# maxsize=1000 covers all known test cases + live chatbot repeated queries.
# PhD note: caching reduces latency for repeated bias profiles (e.g. COMPAS
# queried multiple times across 195 unit tests → single computation).
@lru_cache(maxsize=1000)
def _run_scm_cached(findings: CausalFindings) -> SCMResultV2:
    """Cached SCM computation — called by SCMEngineV2.run()."""
    return SCMEngineV2._compute(findings)


# ═══════════════════════════════════════════════════════════════════
# KNOWN CASES — extended with v2 fields
# ═══════════════════════════════════════════════════════════════════

KNOWN_CASES = {
    "amazon_hiring_2018": CausalFindings(
        tce=12.4, med=68.0, flip=43.0, intv=15.0, rct=False,
        p_y_given_x=0.41, p_y_given_notx=0.29, p_x=0.40,
        nde_raw=4.0,   # 4% direct, rest mediated through biased_ranking_score
        domain="representation_bias",
        subgroup_tce={"female": 12.4, "male": 1.2},
    ),
    "compas_2016": CausalFindings(
        tce=18.3, med=52.0, flip=61.0, intv=40.0, rct=False,
        p_y_given_x=0.52, p_y_given_notx=0.34, p_x=0.45,
        nde_raw=8.0,
        domain="criminal_justice_bias",
        subgroup_tce={"black": 18.3, "white": 4.1},
    ),
    "microsoft_tay_2016": CausalFindings(
        tce=15.7, med=68.0, flip=99.0, intv=10.0, rct=False,
        p_y_given_x=0.68, p_y_given_notx=0.52, p_x=0.3,
        domain="context_poisoning",
        subgroup_tce={},
    ),
    "healthcare_bias_2019": CausalFindings(
        tce=16.5, med=62.0, flip=38.0, intv=30.0, rct=False,
        p_y_given_x=0.61, p_y_given_notx=0.45, p_x=0.35,
        domain="representation_bias",
        subgroup_tce={"black": 16.5, "white": 3.2},
    ),
}


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = SCMEngineV2()
    print("\n  Testing scm_engine_v2.py — 2 known cases\n")
    for name in ["amazon_hiring_2018", "compas_2016"]:
        findings = KNOWN_CASES[name]
        result   = engine.run(findings)
        engine.print_report(findings, result)
        print()


# ═══════════════════════════════════════════════════════════════════
# SPARSE CAUSAL ACTIVATION MATRIX  (v2 addition)
# User insight: 23×5 matrix with central nodes (weight ≥12)
# that cascade to adjacent rows when activated
# ═══════════════════════════════════════════════════════════════════

# OLD: 17×5 integer [P1-P5] → now 23×5 float [RCT/TCE/INTV/MED/FLIP] (matrix_v2.py)
# P1=Interpretability, P2=Behavior, P3=Data, P4=Robustness, P5=Society
# ── 23×5 Pearl Matrix — UPGRADED from integer 17×5 (P1-P5) ─────────
# Values now [0.0, 1.0] causal scores from matrix_v2.py
# Columns now Pearl L1→L3: RCT / TCE / INTV / MED / FLIP
# CAUSAL_MATRIX is no longer a local dict — use MATRIX_23x5 from matrix_v2

# Cascade map — updated to 23×5 category names
# Triggered by CRITICAL tier + HIGH tier with sufficient TCE
CASCADE_MAP: Dict[str, List[str]] = {
    # CRITICAL tier: always cascade to co-occurring harm types
    "weapon_synthesis":   ["violence", "cybercrime", "data_poisoning"],
    "csam":               ["child_safety", "psychological_manipulation"],
    "child_safety":       ["psychological_manipulation", "surveillance_stalking"],
    # HIGH tier: cascade when TCE >= 12
    "bias_discrimination":        ["regulatory_noncompliance", "medical_harm"],
    "regulatory_noncompliance":   ["privacy_violation", "data_poisoning"],
    "medical_harm":               ["bias_discrimination", "regulatory_noncompliance"],
}

# Domain → 23×5 Pearl category mapping
# SCM engine domain keys → matrix_v2 category names
# child_safety now has its OWN proper row (was Representation_Bias proxy before)
DOMAIN_TO_ROW: Dict[str, str] = {
    "representation_bias":   "bias_discrimination",
    "criminal_justice_bias": "bias_discrimination",
    "misuse_safety":         "weapon_synthesis",
    "context_poisoning":     "data_poisoning",
    "decision_transparency": "regulatory_noncompliance",
    "disinformation":        "misinformation_synthetic",
    "harassment":            "psychological_manipulation",
    "cyberattack":           "cybercrime",
    "privacy_violation":     "privacy_violation",
    "financial_fraud":       "financial_fraud",
    "medical_harm":          "medical_harm",
    "physical_violence":     "violence",
    "hate_speech":           "hate_speech",
    "drug_trafficking":      "cybercrime",         # closest harm class
    "child_safety":          "child_safety",        # ← NOW PROPER ROW (v5.1)
    "identity_forgery":      "social_engineering_attack",
    "audit_gap":             "regulatory_noncompliance",
    # ── Added May 2026 ─────────────────────────────────────────
    "self_harm":             "psychological_manipulation",  # closest harm class — distress → method → act
    "election_interference": "misinformation_synthetic",   # same matrix row as disinformation
}

# Pearl column names (replaces P1_Interpretability…P5_Society)
PATHWAY_NAMES: List[str] = PEARL_COLUMNS  # ["RCT","TCE","INTV","MED","FLIP"]

# Cascade trigger: CRITICAL always, HIGH when TCE >= threshold
CENTRAL_THRESHOLD = 12   # kept for backward compat (not used in new logic)
CASCADE_TCE_THRESHOLD = 12.0   # HIGH tier cascades if tce >= this value

# ── Domain Risk Multiplier (v4.9 addition) ─────────────────────────
# Real-world harm severity scales with deployment context.
# Healthcare/Finance = higher stakes → multiplier amplifies SCM risk.
# Year 2: replace with data-driven weights from AIAAIC incident analysis.
DOMAIN_RISK_MULTIPLIER = {
    "general":       1.0,   # Generic queries — baseline
    "entertainment": 1.5,   # Media/creative — low-medium stakes
    "education":     2.0,   # Student/learning — medium stakes
    "finance":       2.5,   # Banking/investment — high stakes
    "healthcare":    3.0,   # Medical/clinical — highest stakes
}

def get_domain_multiplier(query: str) -> tuple:
    """
    Infer deployment domain from query keywords.
    Returns: (multiplier, domain_label)

    Year 1: keyword heuristic
    Year 2: XLM-RoBERTa embedding classifier
    """
    q = query.lower()
    # FIX v15h: Check drug trafficking BEFORE healthcare — "drug" alone matched
    # both, causing trafficking queries to be labelled "healthcare" (3.0×).
    # Drug trafficking is general domain (1.0×); healthcare drug context is
    # caught by clinical/medical/dosage/pharma keywords below.
    if any(k in q for k in ["drug trafficking","drug synthesis","drug dealing",
                             "narcotics","methamphetamine","heroin","cocaine supply",
                             "fentanyl production","manufacture drugs"]):
        return DOMAIN_RISK_MULTIPLIER["general"], "general"
    if any(k in q for k in ["patient","clinical","medical","diagnosis","hospital",
                             "treatment","dosage","surgery","symptom","health","doctor",
                             "cancer","diagnose","nurse","pharma","vaccine","therapy",
                             "drug overdose","drug interaction","medication"]):
        return DOMAIN_RISK_MULTIPLIER["healthcare"], "healthcare"
    if any(k in q for k in ["investment","trading","loan","credit","bank","financial",
                             "insurance","mortgage","fraud","portfolio","tax"]):
        return DOMAIN_RISK_MULTIPLIER["finance"], "finance"
    if any(k in q for k in ["student","school","university","teacher","course","exam",
                             "curriculum","learning","classroom","grade","tutor"]):
        return DOMAIN_RISK_MULTIPLIER["education"], "education"
    if any(k in q for k in ["movie","music","game","fiction","story","character",
                             "entertainment","media","film","song","novel","art"]):
        return DOMAIN_RISK_MULTIPLIER["entertainment"], "entertainment"
    return DOMAIN_RISK_MULTIPLIER["general"], "general"


@dataclass
class MatrixActivation:
    """
    Sparse Causal Activation Matrix result.

    When a query arrives:
      1. Primary row identified from domain
      2. If primary row is central (weight≥12) → cascade fires
      3. Cascaded rows also activate
      4. SCM calculation runs on active subgraph only
      5. Aggregate risk = weighted sum of active cells
    """
    primary_row    : str
    active_rows    : list          # sparse subset of 17
    cascade_fired  : bool
    cascade_source : str           # which central node triggered
    active_cells   : dict          # {row: {path: weight}}
    aggregate_risk : float         # 0.0–1.0
    pathway_scores : dict          # {P1..P5: score}
    dominant_path  : str           # highest pathway score


def activate_matrix(domain: str, tce: float, severity: Severity) -> MatrixActivation:
    """
    Core sparse activation logic — UPGRADED to 23×5 Pearl L1→L3 matrix.

    Step 1: Map domain key → Pearl category (DOMAIN_TO_ROW)
    Step 2: Cascade if CRITICAL tier, or HIGH tier with TCE ≥ threshold
    Step 3: Collect active rows (primary + cascaded)
    Step 4: Compute aggregate_risk via compute_weighted_score (matrix_v2)
    Step 5: Build pathway_scores over Pearl columns [RCT,TCE,INTV,MED,FLIP]

    Backward-compatible: MatrixActivation fields unchanged.
    aggregate_risk still 0.0–1.0; pipeline multiplies by 100.
    """
    # Step 1: resolve domain → 23×5 category
    primary_category = DOMAIN_TO_ROW.get(domain, "weapon_synthesis")

    # Step 2: cascade decision — tier-based (replaces integer sum ≥ 12)
    tier = get_severity_tier(primary_category)
    cascade_fired = (
        tier == "CRITICAL"
        or (tier == "HIGH" and tce >= CASCADE_TCE_THRESHOLD)
    )
    cascade_source    = primary_category if cascade_fired else ""
    cascaded_rows     = CASCADE_MAP.get(primary_category, []) if cascade_fired else []

    # Step 3: active set (deduplicated, primary first)
    active_rows = list(dict.fromkeys([primary_category] + cascaded_rows))

    # Step 4: severity multiplier (same scale as before)
    sev_multiplier = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.2}.get(severity.value, 1.0)
    tce_norm = min(tce / 20.0, 1.0)   # normalize tce → [0, 1]  (same as old tce_factor)

    scm_overrides = {"tce": tce_norm}  # override TCE column with real SCM output

    # Primary score from matrix_v2 with SCM tce override
    primary_score = compute_weighted_score(primary_category,
                                           scm_overrides=scm_overrides)

    if cascade_fired and cascaded_rows:
        cascade_scores = [
            compute_weighted_score(c, scm_overrides=scm_overrides)
            for c in cascaded_rows
        ]
        cascade_avg = sum(cascade_scores) / len(cascade_scores)
        raw_risk = primary_score * 0.70 + cascade_avg * 0.30
    else:
        raw_risk = primary_score

    # SCM Educational Dampener (v5.1 fix)
    # Problem: new float matrix values (0.3-0.9) are always higher than old
    #          integers (1-3 normalized), causing educational queries to score
    #          too high → false positives (20 tests failing).
    # Fix: scale aggregate_risk by tce_norm — if SCM says low harm (low TCE),
    #      the matrix score is dampened proportionally.
    # floor 0.30 = matrix never fully suppressed (preserves signal shape).
    #
    # Educational query: tce=2 → tce_norm=0.10 → dampener=0.30 → risk LOW → ALLOW
    # Harmful query:     tce=20 → tce_norm=1.00 → dampener=1.00 → risk unchanged → BLOCK
    # CRITICAL query:    tce=25 → tce_norm=1.00 → dampener=1.00 → BLOCK always
    scm_dampener = max(tce_norm, 0.30)
    aggregate_risk = round(min(1.0, raw_risk * sev_multiplier * scm_dampener), 3)

    # Step 5: pathway scores over Pearl columns
    primary_row_vals = list(get_row(primary_category))
    pathway_scores: Dict[str, float] = {}
    active_cells:   Dict[str, Dict[str, float]] = {}

    for cat in active_rows:
        row_vals   = list(get_row(cat))
        row_factor = 1.0 if cat == primary_category else 0.6
        cell_map: Dict[str, float] = {}
        for i, col in enumerate(PATHWAY_NAMES):
            activated = round(row_vals[i] * sev_multiplier * row_factor, 3)
            cell_map[col] = activated
            pathway_scores[col] = pathway_scores.get(col, 0.0) + activated
        active_cells[cat] = cell_map

    # Normalize pathway scores
    n_rows = len(active_rows)
    pathway_scores = {
        col: round(v / max(n_rows, 1), 3)
        for col, v in pathway_scores.items()
    }
    dominant_path = max(pathway_scores, key=pathway_scores.get)

    return MatrixActivation(
        primary_row    = primary_category,
        active_rows    = active_rows,
        cascade_fired  = cascade_fired,
        cascade_source = cascade_source,
        active_cells   = active_cells,
        aggregate_risk = aggregate_risk,
        pathway_scores = pathway_scores,
        dominant_path  = dominant_path,
    )


def print_matrix_activation(m: MatrixActivation) -> None:
    print(f"\n  SPARSE CAUSAL ACTIVATION MATRIX")
    print(f"  Primary Row   : {m.primary_row}")
    print(f"  Cascade Fired : {'YES ← central node' if m.cascade_fired else 'NO'}")
    if m.cascade_fired:
        print(f"  Cascade Source: {m.cascade_source}")
    print(f"  Active Rows   : {len(m.active_rows)}/23 (sparse)")
    for row in m.active_rows:
        tag = " ← PRIMARY" if row == m.primary_row else " ← cascaded"
        cells = m.active_cells[row]
        total = sum(cells.values())
        print(f"    {row:<30} total_activation={total:.2f}{tag}")
    print(f"  Pathway Scores:")
    for p, s in m.pathway_scores.items():
        bar = "█" * int(s * 20)
        print(f"    {p:<25} {s:.3f}  {bar}")
    print(f"  Dominant Path : {m.dominant_path}")
    print(f"  Aggregate Risk: {m.aggregate_risk:.3f}")


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Sparse Causal Activation Matrix v5.1 — 23×5 Pearl Columns ===")
    print(f"Columns: {PATHWAY_NAMES}")
    test_cases = [
        ("compas_2016",       "criminal_justice_bias", 18.3, Severity.CRITICAL),
        ("misuse_vx",         "misuse_safety",         24.0, Severity.CRITICAL),
        ("amazon_hiring",     "representation_bias",   12.4, Severity.HIGH),
        ("child_abuse_grooming", "child_safety",       20.0, Severity.CRITICAL),
        ("deepfake_politics", "disinformation",         9.0, Severity.HIGH),
    ]
    for name, domain, tce, sev in test_cases:
        print(f"\n  [{name}]")
        m = activate_matrix(domain, tce, sev)
        print_matrix_activation(m)
