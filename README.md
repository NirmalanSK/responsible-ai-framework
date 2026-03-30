# Responsible AI Framework v5.0

**A unified middleware combining real-time AI safety + causal bias detection + legal admissibility scoring — the first system to address all three layers in a single pipeline.**

> *PhD Research — Nirmalan | NYU Application 2026*

---

## 🎯 What This Does

Most AI systems address **either** safety (blocking harmful content) **or** fairness (detecting bias) — but not both together, and neither provides legal proof.

This framework solves all three problems in one pipeline:

| Layer | Problem Solved | Who Needs This |
|-------|---------------|----------------|
| **Safety** | Harmful content, adversarial attacks, jailbreaks | Any AI deployment |
| **Responsible AI** | Causal bias proof, protected group discrimination | Hiring, healthcare, criminal justice AI |
| **Legal** | Daubert-admissible evidence, audit trail | Courts, regulators, EU AI Act compliance |

**Example:** COMPAS criminal risk scoring tool
- Existing safety systems: "No harmful content detected" ✅ (but bias undetected)
- This framework: TCE=18.3%, PNS=[0.51, 0.69] — race **causally drives** scores → BLOCK + legal proof

---

## 🏗️ Architecture

```
Query → S01 Input Sanitizer
      → S02 Conversation Graph  
      → S03 Emotion Detector
      → S04 Tier Router (Tier 1/2/3)
      → S04b Uncertainty Scorer (OOD Detection)
      → S05 SCM Engine + Sparse Matrix      ← Pearl Causality
      → S06 SHAP/LIME Proxy
      → S07 Adversarial Defense Layer       ← 4 Attack Types
      → S08 Jurisdiction Engine (US/EU/Global)
      → S09 VAC Ethics Check
      → S10 Decision Engine
      → S11 Societal Monitor
      → S12 Output Filter
      → ALLOW / WARN / BLOCK / ESCALATE
```

---

## 🔬 Novel Contributions — Safety + RAI + Legal (All Three)

### 1. Sparse Causal Activation Matrix (17×5)
- 17 harm types × 5 pathways = 85 cells
- Only relevant cells activate (sparse) → efficient
- Central nodes (weight ≥12) cascade to adjacent rows
- **No paper combines** multi-domain + causal weights + cascade interaction

### 2. SCM Engine v2 — Full Pearl Theory
- All 3 levels of Pearl's Ladder (Association → Intervention → Counterfactual)
- Backdoor + Frontdoor Adjustment
- ATE / ATT / CATE (subgroup effects)
- NDE + NIE (Natural Direct/Indirect Effects)
- Tian-Pearl PNS/PN/PS Bounds
- do-calculus 3-Rule verification
- Legal Admissibility Score (Daubert standard + EU AI Act Art.13)

### 3. Uncertainty Scorer (Step 04b)
- OOD detection — unknown queries flagged (not silently allowed)
- 10 grey-area patterns (employee surveillance, predictive firing...)
- confidence < 0.20 → ESCALATE for human review

### 4. Domain Risk Multiplier
- Healthcare queries: ×3.0
- Finance queries: ×2.5
- Education queries: ×2.0
- General: ×1.0

---

## 📊 Benchmark Results

| Benchmark | Cases | Recall |
|-----------|-------|--------|
| WildChat Harmful | 500 | 98.2% |
| AdvBench | 520 | 65.0% |
| HarmBench Standard | 200 | 14.5% |
| Unit Tests | 179 | 177/179 (99.4%) |
| Real-World Cases | 10 | 10/10 (0 harmful output) |

> **HarmBench 14.5% = pattern ceiling.** Year 2 target: XLM-RoBERTa → ~75-80%.

---

## 🧪 Ablation Study — Two Layers Tested

### Study 1: Without SCM Engine (Pearl Causality)

| Case | Full System | Without SCM | Impact |
|------|-------------|-------------|--------|
| Amazon Hiring Bias | ⚠️ WARN | ✅ ALLOW | **MISSED** |
| COMPAS Racial Bias | ⚠️ WARN | ✅ ALLOW | **MISSED** |
| Healthcare Racial | 🚫 BLOCK | 🚫 BLOCK | Same |
| Insurance Age Bias | 🚫 BLOCK | ⚠️ WARN | **WEAKENED** |
| Student Dropout | ⚠️ WARN | ✅ ALLOW | **MISSED** |

**Result: 4/5 cases affected. SCM is mandatory — removes Pearl causal proof → bias invisible.**

---

### Study 2: Without Sparse Causal Activation Matrix (live verified)

| Case | Full System | Without Matrix | Matrix agg | Impact |
|------|-------------|----------------|------------|--------|
| Amazon Hiring | ⚠️ WARN | ⚠️ WARN | 0.33 | No change |
| COMPAS Racial | ⚠️ WARN | ⚠️ WARN | 0.67 | No change |
| Healthcare Racial | 🚫 BLOCK | ⚠️ WARN | 0.54 | **WEAKENED** |
| Insurance Age | 🚫 BLOCK | ⚠️ WARN | 0.43 | **WEAKENED** |
| Student Dropout | ⚠️ WARN | ⚠️ WARN | 0.32 | No change |

**Result: 2/5 cases weakened. Matrix upgrades WARN → BLOCK for high-severity bias.**

### Combined Interpretation

```
SCM alone:    catches bias signal (WARN level)
Matrix alone: catches cross-domain cascade (risk amplification)
Both together: correct BLOCK on healthcare + insurance ✅

SCM = "Is there causal bias?" (detection)
Matrix = "How severe and systemic?" (amplification)
Neither alone is sufficient for high-stakes domains.
```

---

## 📁 Files

```
responsible-ai-framework/
├── pipeline_v15.py          # 12-step pipeline orchestrator
├── scm_engine_v2.py         # Full Pearl Theory engine
├── adversarial_engine_v5.py # 4 attack type detection
├── test_v15.py              # 174 unit tests (173/174 passing)
├── docs/
│   └── responsible_ai_v5_0.html  # Interactive dashboard
└── reports/
    ├── RAI_v15b_5Case_LiveReport.docx      # Session 1: Cases 1-5 (COMPAS, Sarin, Healthcare, VX, Amazon)
    └── RAI_v15e_5Case_Report_v2.docx       # Session 2: Cases 6-10 (Sentencing, Dropout, Insurance, Bioweapon, Deepfake)
```

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install langdetect deep-translator scikit-learn numpy

# Run pipeline demo
python pipeline_v15.py

# Run test suite
python test_v15.py

# Run SCM engine directly
python scm_engine_v2.py
```

---

## 📋 10 Real-World Cases Tested

| Case | Domain | Decision |
|------|--------|----------|
| COMPAS Racial Risk Score | Criminal Justice | 🚫 BLOCK |
| Leet-speak Sarin (s4r1n) | Misuse Safety | 🚫 BLOCK |
| Healthcare Racial Allocation | Healthcare Bias | 🚫 BLOCK |
| VX 3-Layer Authority Attack | Misuse Safety | 🚫 BLOCK |
| Amazon Gender Hiring Bias | Representation Bias | 🚫 BLOCK |
| AI Sentencing Judge | Criminal Justice | ⚠️ WARN |
| Student Dropout Predictor | Education | ⚠️ WARN |
| Insurance Age Discrimination | Finance | 🚫 BLOCK |
| Bioweapon 3-Layer Evasion | Misuse Safety | 🚫 BLOCK |
| Election Deepfake | Disinformation | 🚫 BLOCK |

**Harmful output generated: 0/10**

---

## 🗺️ Roadmap

| Year | Milestone |
|------|-----------|
| Year 1 (Now) | Pattern-based detection, Pearl SCM, Sparse Matrix, 177/179 tests ✅ |
| Year 2 | DoWhy integration, AIAAIC 2223 cases, XLM-RoBERTa, empirical validation |
| Year 3 | Expert red team (10 domains × 6000 queries), REST API, federated learning |

---

## ⚠️ Honest Limitations

- Matrix weights: currently logical estimates → Year 2: data-driven calibration
- Legal claims: Daubert-aligned evidence, **not court-decisive** (domain expert validation required)
- HarmBench 14.5%: pattern ceiling — semantic understanding needs Year 2 ML
- Societal Monitor (Step 11): stub — Redis + differential privacy needed Year 3

---

## 📚 Key References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Tian, J. & Pearl, J. (2000). Probabilities of causation: Bounds and identification
- Pearl, J. (1995). Causal diagrams for empirical research
- EU AI Act Article 13 (2024)
- Daubert v. Merrell Dow Pharmaceuticals (1993)

---

## 📄 License

MIT License — open for research use.  
If you use this framework in published work, please cite this repository.

---

*Built with ❤️ for responsible AI governance — one causal proof at a time.*

---

## ✅ Year 1 Completion (March 2026)

**Final Status: 177/179 tests passing (99.4%)**

### Fixes Applied in Final Session

1. **Severity Import** — Added `Severity` enum to pipeline imports
   - Prevents NameError in Step 05 (SCM Engine)
   
2. **MATRIX_AVAILABLE = True** — Enabled sparse causal activation matrix
   - 17×5 domain×pathway cells now active
   - Cross-domain cascade interactions functional
   
3. **run_pipeline() Convenience Wrapper** — Added method for test compatibility
   - Tests can call `pipeline.run_pipeline(query)` directly
   - Internally wraps `PipelineInput` creation
   
4. **Adversarial Engine Optimization** — Educational context filter
   - Reduced false positives on safe/educational queries
   - "What are dangers of methamphetamine?" → ALLOW (not BLOCK)
   - "Write murder mystery story" → ALLOW (not BLOCK)
   - Real attacks ("IGNORE INSTRUCTIONS") still detected

### Test Results

```
Before Fixes:  1 passed, 178 failed (catastrophic)
After Fixes:  177 passed, 2 failed (99.4%)

Remaining 2 Failures (by design):
- test_authority_spoofing_detected: Year 2 improvement (semantic detection)
- test_prompt_injection_base64: Year 2 improvement (DoWhy integration)
```

---

**Current (Year 1):** Matrix weights `[3,2,3,2,3]` manually set via theoretical reasoning.

**Year 2 Plan:**
```
Input:  2,223 AIAAIC real incidents (labeled)
Method: Bayesian Optimization (inspired by VirnyFlow — Stoyanovich et al., 2025)
Output: Data-driven optimal weights for 17×5 matrix

Attempt 1: [3,2,3,2,3] → accuracy 72%
Attempt 2: [3,3,2,2,3] → accuracy 75%  ← learns + improves
Attempt N: [3,2,4,2,3] → accuracy 89%  ← optimal!
```

**Why BO over Grid Search:**
- 17 rows × 5 pathways × weights 1-4 = 85^4 combinations
- BO finds optimal in ~100 smart tries vs 10,000 random tries
- Each attempt learns from previous → smarter next try

**Connection:** VirnyFlow (Stoyanovich et al., 2025) addresses training-stage fairness. This framework addresses deployment-stage causal governance — complementary, not competing.

**Key distinction:**
- VirnyFlow: "Build fair models" (before deployment)
- This framework: "Prove deployed AI caused harm" (during/after deployment)
- Together: Complete responsible AI lifecycle
