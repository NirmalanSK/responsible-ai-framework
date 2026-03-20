# Responsible AI Framework v5.0

**A unified middleware combining Judea Pearl's causal inference, sparse activation architecture, and multi-layer governance for real-time AI decision auditing.**

> *PhD Research — Nirmalan | NYU Application 2026*

---

## 🎯 What This Does

Most AI safety systems detect harm using **correlation-based pattern matching**.  
This framework uses **causal proof** — mathematically establishing *why* an AI decision is harmful, not just *that* it might be.

**Example:** COMPAS criminal risk scoring tool  
- Pattern matching: "Black defendants score higher" (correlation)  
- This framework: TCE=18.3%, PNS=[0.51, 0.69] — race **causally drives** scores (causal proof)

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

## 🔬 Novel Contributions

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
| Unit Tests | 174 | 173/174 (99%) |
| Real-World Cases | 10 | 10/10 (0 harmful output) |

> **HarmBench 14.5% = pattern ceiling.** Year 2 target: XLM-RoBERTa → ~75-80%.

---

## 🧪 Ablation Study (GPT PhD Examiner Required)

| Case | Full System | Without SCM | Impact |
|------|-------------|-------------|--------|
| Amazon Hiring Bias | ⚠️ WARN | ✅ ALLOW | **MISSED** |
| COMPAS Racial Bias | ⚠️ WARN | ✅ ALLOW | **MISSED** |
| Healthcare Racial | 🚫 BLOCK | 🚫 BLOCK | Same |
| Insurance Age Bias | 🚫 BLOCK | ⚠️ WARN | **WEAKENED** |
| Student Dropout | ⚠️ WARN | ✅ ALLOW | **MISSED** |

**Result: 4/5 bias cases affected without SCM. SCM is mandatory, not decorative.**

---

## 📁 Files

```
responsible-ai-framework/
├── pipeline_v15.py          # 12-step pipeline orchestrator
├── scm_engine.py            # SCM v1 (carry-over)
├── scm_engine_v2.py         # Full Pearl Theory engine
├── adversarial_engine_v5.py # 4 attack type detection
├── test_v15.py              # 174 unit tests (173/174 passing)
├── docs/
│   └── responsible_ai_v5_0.html  # Interactive dashboard
└── reports/
    └── RAI_v15e_5Case_Report_v2.docx
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
| Year 1 (Now) | Pattern-based detection, Pearl SCM, Sparse Matrix, 173/174 tests |
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
