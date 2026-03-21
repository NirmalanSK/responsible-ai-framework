# Research Notes — Nirmalan
# RAI Framework — Ideas & Future Plans

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
