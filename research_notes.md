# Research Notes — Nirmalan
# RAI Framework — Ideas & Future Plans

---

## 🎯 CORE THESIS FRAMING (Updated March 2026)

### The Correct Claim:
"This framework is the FIRST to combine Safety + Responsible AI + Legal proof
 in a single unified middleware."

### Three Layers:

LAYER 1 — SAFETY (Steps 01, 04b, 07, 09):
  What: Harmful content detection, adversarial attack defense
  Who does this: Claude, GPT (built-in)
  Our addition: Adversarial engine (4 attack types + academic/fictional/symbolic wrappers)
  Evidence: 173/174 tests, 0/10 harmful output

LAYER 2 — RESPONSIBLE AI (Steps 05, 06, 08, 11):
  What: Causal bias detection, protected group discrimination proof
  Who does this: NOBODY at deployment stage
  Our contribution: Pearl SCM v2 + Sparse Matrix + Legal Score
  Evidence: Amazon hiring, COMPAS, Healthcare, Insurance cases

LAYER 3 — LEGAL PROOF (Step 05 SCM output):
  What: Daubert-admissible causal evidence, audit trail
  Who does this: NOBODY (VirnyFlow = training only, no legal layer)
  Our contribution: PNS/PN/PS bounds + Legal Admissibility Score
  Evidence: "VERY STRONG — L3 counterfactual bounds establish causal responsibility"

### Why Existing Systems Are Incomplete:
  Claude/GPT = Layer 1 only (safety)
  VirnyFlow = Training fairness (not deployment, not legal)
  RAISE = Evaluation metrics (not real-time, not causal)
  COMPAS = Biased system (no audit)

  Our framework = All 3 layers, real-time, deployed system

### PhD Defense Answer to "Why not just use Claude?":
  "Claude addresses harmful content (Layer 1).
   Our framework additionally proves causal bias in AI DECISIONS (Layer 2)
   and generates legally-admissible evidence (Layer 3).
   These are fundamentally different problems.
   A Claude-powered hiring system can be perfectly safe (no harmful content)
   while systematically discriminating by race — our framework catches this."

---

---

## 💡 IDEA: Bayesian Optimization for Matrix Weight Calibration
Date: March 2026
Source: VirnyFlow paper (Stoyanovich et al., 2025) → independently thought of same idea

### The Problem
Current matrix weights are manually set:
  "Representation_Bias": [3, 2, 3, 2, 3]
  
  P1=3 (Interpretability)
  P2=2 (Behavior)  
  P3=3 (Data)
  P4=2 (Robustness)
  P5=3 (Society)

GPT examiner said: "Why 3? Why not 4?" → no empirical justification
= WEAK point in PhD defense

### The Solution
Bayesian Optimization (BO) on AIAAIC 2,223 real incidents:

  Step 1: Label each incident
    "COMPAS" → domain=criminal_justice, weights=[?, ?, ?, ?, ?]
    
  Step 2: BO tries weight combinations:
    Try 1: [3,2,3,2,3] → F1=0.72
    Try 2: [3,3,2,2,3] → F1=0.75  ← learned: P2 more important
    Try 3: [4,2,3,1,3] → F1=0.78  ← P1 even more important?
    Try N: [3,2,4,2,3] → F1=0.89  ← OPTIMAL
    
  Step 3: Each try learns from previous (Gaussian Process)
    → Smarter next try
    → 100 smart tries > 10,000 random tries

### Why This Matters for PhD
  Year 1: "Logical estimates" → weak
  Year 2: "BO-calibrated on 2223 real incidents" → strong
  
  GPT revision addressed:
    "Metric calibration (heuristic → data-driven)" ✅

### Connection to VirnyFlow
  VirnyFlow: BO for ML model hyperparameters
  Our idea: BO for harm pathway weights
  
  Different domain, same technique
  → Shows research awareness of field
  → Cite in thesis: Stoyanovich et al. 2025

### Implementation Plan (Year 2)
  1. Annotate 1000 AIAAIC incidents with:
     - Primary harm domain
     - Which pathways (P1-P5) were involved
     - Severity
  
  2. Use scikit-optimize or OpenBox (VirnyFlow uses OpenBox)
     pip install scikit-optimize
  
  3. Objective function:
     def objective(weights):
       update CAUSAL_MATRIX with weights
       run pipeline on labeled test set
       return -F1_score  # minimize negative = maximize F1
  
  4. BO finds optimal weights in ~100 iterations
  
  5. Update HTML v5.0 + scm_engine_v2.py

### Code Sketch (Year 2)
  from skopt import gp_minimize
  from skopt.space import Integer
  
  space = [Integer(1, 4, name=f'w{i}') for i in range(85)]  # 17×5
  
  result = gp_minimize(
      objective,
      space,
      n_calls=100,
      random_state=42
  )
  
  optimal_weights = result.x

---

## 💡 IDEA: VirnyFlow Integration (Future)
Date: March 2026

### Concept
  VirnyFlow → bias-free model training (before deployment)
  Our framework → bias detection (during deployment)
  
  Combined pipeline:
    VirnyFlow training → Our framework governance → Complete RAI

### NYU Application Statement
  "I build on Stoyanovich et al.'s VirnyFlow (2025) by extending
   responsible AI governance from model development to the
   deployment layer. While VirnyFlow optimizes fairness during
   training, our framework provides real-time causal proof when
   bias emerges in deployed systems — using Pearl's do-calculus
   rather than correlation-based metrics."

---

## 📌 TODO — Year 2 Priorities
1. BO weight calibration (this idea)
2. DoWhy integration (auto-compute TCE)
3. AIAAIC 1000 annotations
4. XLM-RoBERTa for OOD detection
5. Real SHAP values (replace proxy)

---

## 📚 Key Papers to Cite
- VirnyFlow: Herasymuk, Protsiv, Stoyanovich (2025) — BO for fair ML
- Pearl (2009) — Causality
- Tian & Pearl (2000) — PNS bounds
- Obermeyer et al. (2019, Science) — Healthcare AI racial bias
- ProPublica COMPAS (2016) — Criminal justice AI bias

---

## 🎓 PhD Thesis Framing (Updated)
Chapter 1: Introduction
  "Rules can be hacked. Values cannot — if truly internalized."
  
Chapter 2: Literature Review
  VirnyFlow (training layer) → gap → our framework (deployment layer)
  
Chapter 3: Architecture
  12-step pipeline + SCM v2 + Sparse Matrix
  
Chapter 4: Case Studies
  10 real-world cases — 0 harmful output
  
Chapter 5: Empirical Evaluation
  Ablation study: 4/5 cases affected without SCM
  
Chapter 6: Year 2 Plan
  BO weight calibration + DoWhy + AIAAIC
  
Chapter 7: Limitations
  HarmBench 14.5%, manual weights, EU GDPR gap

---
Last updated: March 2026

---

## 📅 UPDATED YEAR 2/3 PLAN (Synthesized from Gemini + DeepSeek + Kimi reviews)

---

### YEAR 2 — EMPIRICAL FOUNDATION (Months 1-12)

#### Phase 1: Simulation Mode Deployment (Months 1-3)
Source: DeepSeek + Kimi (both agree this is FIRST priority)

  Week 1-2:
    Deploy Simulation Mode on real corpus
    WildChat 500 + AdvBench 520 + AIAAIC 2223
    Log all decisions (no enforcement yet)
    
  Week 3-8:
    Collect ground truth:
      Manual review sample of logged decisions
      Label: TP / FP / TN / FN
      Target: 500 reviewed cases minimum
      
  Week 9-12:
    Compute:
      Precision = TP / (TP + FP)
      Recall = TP / (TP + FN)
      F1 score per tier
      False Positive Rate per domain
    
  Output: "Empirical baseline report"
  = First real precision/recall numbers
  = Replaces "173/174 synthetic tests"

#### Phase 2: Tier Router Upgrade (Months 2-5)
Source: Gemini (SBERT + Confidence), backed by Kimi (pattern maintenance)

  Step 1: SBERT Hybrid Tiering
    Current: keyword threshold (brittle)
    Upgrade: sentence-transformers/all-MiniLM-L6-v2
    
    Implementation:
      Query → SBERT embedding vector
      Cosine similarity vs AdvBench failure library
      similarity > 0.90 → Tier 3 immediately
      
    Latency: <10ms (Tier 1 maintained)
    Expected improvement: novel attack detection +40%
    
  Step 2: Confidence-Based Escalation
    Current: binary tier decision
    Upgrade: BERT confidence score → tier decision
    
    Logic:
      BERT > 95% confident Tier 1 → pass
      BERT < 70% confidence → auto-escalate Tier 3
      
    PhD term: "Aleatoric Uncertainty Quantification"
    = System knows when it doesn't know
    
    Connection: Extends Step04b already built ✅
    
  Step 3: Fine-tune BERT Classifier
    Training data:
      WildChat 500 (labeled)
      + AdvBench 520 (labeled)
      + AIAAIC 1000 annotated
      = 2020 real labeled cases (NO synthetic augmentation)
      
    Note on Gemini's augmentation idea:
      LLM paraphrase = synthetic data
      Real > Synthetic for research claim
      "Data-driven not synthetic" = stronger PhD claim
      
    Expected: keyword accuracy ~72% → BERT ~89-91%

#### Phase 3: Bayesian Optimization (Months 4-7)
Source: DeepSeek (BO calibration) + research_notes (our own idea)

  Input: AIAAIC 2223 labeled incidents
  Method: scikit-optimize gp_minimize
  
  Optimize:
    Matrix weights [3,2,3,2,3] → data-driven
    Tier thresholds (12 → validate)
    Domain multipliers (3.0/2.5/2.0 → calibrate)
    
  Output:
    "Empirically calibrated weights"
    = GPT examiner's major revision ✅ DONE

#### Phase 4: DoWhy Integration (Months 5-9)
Source: DeepSeek (highest priority technical upgrade)

  Current: Manual TCE input
  Upgrade: DoWhy auto-compute from AIAAIC data
  
  Implementation:
    pip install dowhy
    For each AIAAIC incident:
      Build causal graph
      DoWhy estimates TCE, NIE, NDE
      Auto-feed to SCMEngineV2
      
  DAG validation:
    Sensitivity analysis on each domain DAG
    "Is representation_bias DAG correct?"
    Expert domain review (hiring expert, healthcare expert)
    
  Output: "Auto-computed causal inputs"
  = Manual dependency removed ✅

#### Phase 5: Causal-Neural Feedback Loop (Months 6-10)
Source: Gemini (unique idea — not in DeepSeek/Kimi)

  Concept:
    BERT says: "ALLOW" (80% confident)
    SCM runs counterfactual: PNS > 0.30
    SCM overrides: "BLOCK"
    
  Logic: Pearl > Neural always
    "Causal proof supersedes statistical confidence"
    
  Implementation:
    If BERT_confidence < 0.95 AND scm_risk > 25%:
      → SCM decision overrides BERT
    
  PhD value:
    "Causal-Neural Feedback Loop"
    = Novel architecture contribution
    = No existing system does this

#### Phase 6: Publication (Months 9-12)
Source: DeepSeek (strongly recommended)

  Target conferences:
    FAccT 2027 (Fairness, Accountability, Transparency)
    AIES 2027 (AI, Ethics, Society)
    
  Paper title candidates:
    "Causal-Neural Governance Middleware for
     Real-time Responsible AI Enforcement"
    
    "An Uncertainty-Aware Tiered Framework
     for Causal Bias Detection in Deployed AI"
    
  Required for paper:
    Empirical precision/recall (Phase 1) ✅
    BO calibration (Phase 3) ✅
    DoWhy validation (Phase 4) ✅

---

### YEAR 3 — PRODUCTION VALIDATION (Months 13-24)

#### Phase 1: Gradual Enforcement Rollout (Months 13-15)
Source: Kimi (critical — "don't enforce immediately")

  Month 13: Simulation Mode → Warn Mode
    BLOCK → WARN (softer)
    Collect user feedback
    
  Month 14-15: Warn → Enforce
    Tune FP rate below 2%
    Then full enforcement
    
  Computational cost plan (Kimi catch):
    1M queries/day × 5% Tier3 × 600ms = 8.3 GPU-hours/day
    Solution: Redis cache + p95 < 200ms target
    Pre-compute common patterns

#### Phase 2: Expert Red Team (Months 13-18)
Source: Our existing plan (Year 3)

  10 domains × 6000 queries:
    Hiring expert: 600 queries
    Healthcare expert: 600 queries
    Criminal justice expert: 600 queries
    (etc.)
    
  Monthly cycle: attack → detect → patch → verify

#### Phase 3: Infrastructure (Months 15-20)
Source: DeepSeek + Kimi

  Redis caching:
    Societal Monitor (Step 11) activation
    Population-level drift detection
    
  REST API:
    p95 < 200ms target
    Kubernetes deployment
    
  XLM-RoBERTa:
    Replace Google Translate
    Language-agnostic embeddings
    Reduce patterns by 60% (Kimi estimate)

#### Phase 4: Uncertainty Meter — HTML v6.0 (Month 16)
Source: Gemini (HTML dashboard suggestion)

  New tab: "🎯 Uncertainty Meter"
    Live confidence score per query
    BERT confidence: 87% | SBERT similarity: 0.23
    SCM override: YES/NO
    Tier decision trace
    
  = Visual proof of Causal-Neural Feedback Loop
  = Professor demo-ready

---

### COMPARISON: What Each AI Uniquely Contributed

  Gemini:
    ✅ SBERT hybrid tiering (technical upgrade)
    ✅ Confidence-Based Escalation (uncertainty)
    ✅ Causal-Neural Feedback Loop (most novel)
    ✅ Uncertainty Meter HTML widget
    ⚠️ Augmentation idea (good but synthetic data weaker)

  DeepSeek:
    ✅ Simulation Mode first (highest priority)
    ✅ DoWhy integration (critical Year 2)
    ✅ FAccT publication target (career impact)
    ✅ Ground truth collection methodology
    = Most practical, implementation-focused

  Kimi:
    ✅ Gradual rollout warning (production reality)
    ✅ Computational cost analysis (8.3 GPU-hours)
    ✅ Pattern maintenance problem (XLM-RoBERTa fix)
    ✅ Jurisdiction conflict flagged
    = Most honest about production challenges

---

### UPDATED THESIS CONTRIBUTION (Year 2 added)

  Year 1 (done):
    Safety + RAI + Legal proof
    Pearl SCM v2 + Sparse Matrix
    173/174 tests, 10/10 cases

  Year 2 (planned):
    Uncertainty-Aware Tiered Causal-Neural Governance
    SBERT + BERT + BO + DoWhy
    Empirical precision/recall
    FAccT paper

  Year 3 (planned):
    Production-validated system
    Expert red team (10 domains)
    REST API p95 < 200ms
    "Validated System" label earned

---
Last updated: March 2026
Synthesized from: Gemini + DeepSeek + Kimi reviews
