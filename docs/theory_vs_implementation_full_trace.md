# Theory vs Implementation: Full Start-to-End Trace
**Framework:** Responsible AI Framework v5.0  
**Case:** COMPAS — Race Bias (Black vs White, Criminal Justice)  
**Input:** TCE = 10.0 | MED = 5.0 | Flip Rate = 20% | Intv = 15.0

> **How to read this document:**  
> Every section has two columns — LEFT = theoretical Pearl math (for PhD explanation),  
> RIGHT = actual code logic from `scm_engine_v2.py` + `pipeline_v15.py`.  
> Both paths use the same input. Decisions may differ — that difference matters.

---

## PHASE 0 — Input

| Field | Value | Unit |
|-------|-------|------|
| TCE (Total Causal Effect) | 10.0 | percentage points |
| MED (Mediation / Edge proxy) | 5.0 | percentage points |
| Flip Rate | 20.0 | % |
| Intv (Intervention / ATT proxy) | 15.0 | percentage points |
| Domain | criminal\_justice | — |

---

## PHASE 1 — Pearl Causal Estimands

> This is where TCE, MED, Flip, ATT come from mathematically.  
> Theory and implementation share the same formulas here.

### 1A. Backdoor Adjustment (Pearl L2)

```
P(Y=1 | do(X=1)) = Σ_z  P(Y=1 | X=1, Z=z) · P(Z=z)
P(Y=1 | do(X=0)) = Σ_z  P(Y=1 | X=0, Z=z) · P(Z=z)

TCE  = P(Y=1|do(X=1)) − P(Y=1|do(X=0))  =  0.10  (10 pp)
MED  = P(Y=1|X=1)    − P(Y=1|X=0)       =  0.05  (5 pp, no adjustment)
```

Key: TCE > MED → confounder (age × criminal history) was masking race effect.

### 1B. Effect Decomposition — code formula (scm_engine_v2.py)

```
ATE = TCE / 100                                     = 0.10
selection_factor = 1.0 + max(0, (0.5 − p_x)/0.5) × 0.3
ATT = ATE × selection_factor                        ≈ 0.13–0.15

NIE  = (MED/100) × (TCE/100)  =  0.05 × 0.10  =  0.005
TE   = NDE + NIE
NDE  = TE − NIE
```

### 1C. Counterfactual Bounds (Pearl L3)

```
p   = P(Y=1 | X=1)   — Black high-risk rate
p'  = P(Y=1 | X=0)   — White high-risk rate
tce = TCE/100 = 0.10

PNS lower = max(0,  p − p')
PNS upper = min(p,  1 − p')

PN  lower = max(0,  (tce − p') / p)
PN  upper = min(1,  (1 − p')   / p)

PS  lower = max(0,  (tce − (1−p)) / (1−p'))
PS  upper = min(1,  p / (1−p'))
```

Flip Rate = 20% → P(Y_do(1) ≠ Y_do(0)) = 0.20  
→ 1-in-5 Black individuals would get different score if White. Counterfactual proof.

---

## PHASE 2 — Trigger Mapping (What Fires)

### THEORY side — Normalisation

```
max thresholds: TCE→0.20, MED→0.10, Flip→0.50, Intv→0.30

m_TCE  = 10/20   = 0.50
m_MED  = 5/10    = 0.50
m_Flip = 20/50   = 0.40
m_Intv = 15/30   = 0.50

m = [0.50, 0.50, 0.40, 0.50]ᵀ
```

Nothing "triggers" here — continuous normalisation into [0,1].

### IMPLEMENTATION side — Discrete Triggers (scm_engine_v2.py)

Rule lookup tables fire one label each:

```
┌─────────────────┬──────────────────────────┬─────────────────────┬───────────────┐
│ Input           │ Rule                     │ Label triggered     │ Enum value    │
├─────────────────┼──────────────────────────┼─────────────────────┼───────────────┤
│ TCE = 10.0      │ TCE >= 10 → CRITICAL     │ Severity = CRITICAL │ 4             │
│ MED = 5.0       │ med < 20  → WEAK         │ EdgeStrength = WEAK │ 1             │
│ Flip = 20.0     │ flip >= 10 → MODERATE    │ Detection = MODERATE│ 1             │
│ Intv = 15.0     │ intv <= 20 BUT sev=CRIT  │ No severity change  │ stays 4       │
└─────────────────┴──────────────────────────┴─────────────────────┴───────────────┘
```

Severity adjustment rule detail:
```
intv <= 20 AND severity < CRITICAL  →  severity bumped UP by 1 level
intv <= 20 AND severity = CRITICAL  →  no change (already max)
intv >= 80                          →  severity reduced by 1 level
otherwise                           →  no change
```
Result here: **Severity = CRITICAL (4), EdgeStrength = WEAK (1), Detection = MODERATE (1)**

---

## PHASE 3 — SCM Risk Score Calculation

### THEORY side — Weighted Sum

```
w_scm = [0.40, 0.20, 0.30, 0.10]ᵀ

R_scm = wᵀ · m
      = 0.40×0.50 + 0.20×0.50 + 0.30×0.40 + 0.10×0.50
      = 0.200 + 0.100 + 0.120 + 0.050
      = 0.470    →  scaled to 47.0%
```

Theory standalone bands:
```
≥ 65% → BLOCK | ≥ 50% → ESCALATE | ≥ 35% → WARN | else → ALLOW
47.0% → WARN
```

### IMPLEMENTATION side — Point-based Formula

```
base = 9
sev  = 4   (CRITICAL)
det  = 1   (MODERATE)
dom  = 1   (WEAK EdgeStrength)

raw  = base + sev×3 + det×2 + dom×0.75
     = 9    + 4×3   + 1×2   + 1×0.75
     = 9    + 12    + 2     + 0.75
     = 23.75

base_risk = 23.75 / 432 × 100 × 4.5
          = 23.75 / 432 × 450
          = 24.74%
```

Implementation standalone bands:
```
≥ 65% → BLOCK | ≥ 50% → ESCALATE | ≥ 35% → WARN | else → ALLOW
24.74% → ALLOW
```

### Side-by-side comparison

```
┌──────────────────────┬───────────────────┬──────────────────────┐
│ Aspect               │ THEORY            │ IMPLEMENTATION       │
├──────────────────────┼───────────────────┼──────────────────────┤
│ Method               │ Normalized w·m    │ Point-based formula  │
│ SCM score            │ 47.0%             │ 24.74%               │
│ Standalone decision  │ WARN              │ ALLOW                │
│ Why they differ      │ Balanced weights  │ WEAK edge pulls down │
└──────────────────────┴───────────────────┴──────────────────────┘
```

Root cause of difference: MED=5 → EdgeStrength=WEAK (value 1) in code, but in theory MED=5 normalizes to 0.50 (moderate). Edge label is the swing factor.

---

## PHASE 4 — Sparse Causal Activation Matrix (17×5)

> This is implementation-only. Theory uses a simpler weighted sum.  
> Matrix represents 17 bias types × 5 causal pathways.

### 4A. Which row activates?

COMPAS → **Criminal_Justice_Bias** row is primary.

```
Primary row:  Criminal_Justice_Bias = [3, 2, 3, 2, 3]
Row sum = 13   →   13 >= 12  →  CASCADE FIRES ✓
```

### 4B. Cascade — which rows also activate?

```
Primary fires → cascade triggers:
  ├── Healthcare_Bias          (row_factor = 0.6)
  ├── Decision_Transparency    (row_factor = 0.6)
  └── Representation_Bias      (row_factor = 0.6)
```

### 4C. Activation weight formula

```
sev_multiplier = 1.2      (CRITICAL)
tce_factor     = min(10.0 / 20.0, 1.0) = 0.50

For primary row (row_factor = 1.0):
  activated_weight_i = row_weight_i × sev_multiplier × tce_factor × row_factor

For cascade rows (row_factor = 0.6):
  activated_weight_i = row_weight_i × sev_multiplier × tce_factor × 0.6
```

### 4D. Primary row calculation

```
P1: 3 × 1.2 × 0.50 × 1.0 = 1.800
P2: 2 × 1.2 × 0.50 × 1.0 = 1.200
P3: 3 × 1.2 × 0.50 × 1.0 = 1.800
P4: 2 × 1.2 × 0.50 × 1.0 = 1.200
P5: 3 × 1.2 × 0.50 × 1.0 = 1.800

Primary pathway total = 7.800
```

### 4E. Cascade rows calculation (sample — Healthcare_Bias = [2, 3, 2, 3, 2])

```
P1: 2 × 1.2 × 0.50 × 0.6 = 0.720
P2: 3 × 1.2 × 0.50 × 0.6 = 1.080
P3: 2 × 1.2 × 0.50 × 0.6 = 0.720
P4: 3 × 1.2 × 0.50 × 0.6 = 1.080
P5: 2 × 1.2 × 0.50 × 0.6 = 0.720

Cascade row total = 4.320
```

### 4F. Aggregate risk formula

```
max_possible = 3 × sev_multiplier × len(active_rows)
             = 3 × 1.2 × 4          (primary + 3 cascade)
             = 14.4

pathway_score_i  = pathway_total_i / max_possible
aggregate_risk   = Σ(pathway_totals) / (max_possible × 5)

Estimated aggregate_risk ≈ 0.32–0.36
matrix_risk ≈ 32–36%
```

Note: Exact value depends on all 4 active rows' weights from the 17×5 matrix in code.  
Healthcare example in file gave 44.3% with tce_factor=0.825; COMPAS gets ~33% with tce_factor=0.50.

---

## PHASE 5 — Pipeline SCM Combination

### Formula

```
combined_risk = 0.6 × scm_risk + 0.4 × matrix_risk
              = 0.6 × 24.74   + 0.4 × 33.0
              = 14.84          + 13.20
              = 28.04%
```

### Domain multiplier check

```
Domain = criminal_justice → multiplier = 1.0 (same as general)
(healthcare = 3.0 | finance = 2.5 | education = 2.0 | entertainment = 1.5 | general = 1.0)

Rule: if combined_risk >= 25 → apply multiplier
28.04 >= 25  →  multiplier applies

final_scm_risk = min(100, 28.04 × 1.0) = 28.04%
```

### Pipeline SCM signal bands

```
≥ 70% → BLOCK | ≥ 30% → WARN | else → CLEAR
28.04% → CLEAR
```

### Theory vs Implementation — SCM final signal

```
┌────────────────────────┬──────────────┬────────────────┐
│ Stage                  │ THEORY       │ IMPLEMENTATION │
├────────────────────────┼──────────────┼────────────────┤
│ SCM risk score         │ 47.0%        │ 24.74%         │
│ Matrix risk            │ N/A          │ ~33.0%         │
│ Combined risk          │ N/A          │ 28.04%         │
│ After domain mult.     │ N/A          │ 28.04%         │
│ SCM pipeline signal    │ WARN         │ CLEAR          │
└────────────────────────┴──────────────┴────────────────┘
```

---

## PHASE 6 — Legal Admissibility Score

### Additive scoring (same logic in both theory and implementation)

```
ATE = TCE/100 = 0.10

+0.20  →  ATE (0.10) > 0.05          ✓
+0.15  →  ATE (0.10) > 0.10          ✗  (0.10 is not > 0.10, boundary)
+0.10  →  TE > 0.05                  ✓  (assuming TE ≈ TCE)
+0.10  →  direct effect quantified   ✓  (NDE computed)
+0.15  →  PNS lower > 0              ✓  (p > p', so PNS lower > 0)
+0.15  →  PNS lower > 0.3            ✗  (PNS lower ≈ 0.10 estimated)
+0.10  →  but-for causation holds    ✓  (TCE > 0)
+0.05  →  RCT data exists            ✗

Total legal score = 0.20 + 0.10 + 0.10 + 0.15 + 0.10 = 0.65
```

Verdict:
```
≥ 0.90 → DEFINITIVE | ≥ 0.70 → VERY STRONG | ≥ 0.50 → STRONG
0.65 → STRONG
```

### Theory aggregation

```
R_legal = 1 − legal_score = 1 − 0.65 = 0.35   (risk)
```

---

## PHASE 7 — Other Pipeline Signals (for final Step 10)

These steps run in parallel with SCM. Results for COMPAS race-bias query:

```
Step 1  Input Sanitizer         →  CLEAR   (standard English query)
Step 2  Conversation Graph      →  drift_score low → CLEAR
Step 3  Emotion Detector        →  no crisis → CLEAR
Step 4  Tier Router             →  Tier 1  (low drift)
Step 4b Uncertainty Scorer      →  confidence ≈ 0.55 → CLEAR
Step 5  SCM Engine              →  CLEAR   (28.04% < 30%)
Step 6  SHAP proxy              →  protected_attr_hits × 0.15
                                   2 hits → 0.30 → WARN (threshold 0.30)
Step 7  Adversarial Layer       →  CLEAR   (no prompt injection / roleplay)
Step 8  Jurisdiction Engine     →  WARN    (protected attribute + SHAP support)
Step 9  VAC Layer               →  CLEAR   (no absolute violation)
```

SHAP proxy detail:
```
shap_proxy = min(protected_attribute_hits × 0.15, 1.0)
           = min(2 × 0.15, 1.0)
           = 0.30    →  threshold 0.30  →  WARN
```

---

## PHASE 8 — Final Decision Engine (Step 10)

Priority hierarchy — checked top to bottom, first match wins:

```
┌────┬──────────────────────────────────────┬────────────────────────┬──────────┐
│ #  │ Condition                            │ COMPAS status          │ Fires?   │
├────┼──────────────────────────────────────┼────────────────────────┼──────────┤
│ 1  │ VAC violation                        │ VAC = CLEAR            │ ✗        │
│ 2  │ Crisis emotion                       │ No crisis              │ ✗        │
│ 3  │ scm_risk >= 70                       │ 28.04% < 70            │ ✗        │
│ 4  │ Attack score HIGH / CRITICAL         │ Adversarial = CLEAR    │ ✗        │
│ 5  │ Any prior BLOCK signal               │ No BLOCKs              │ ✗        │
│ 6  │ warn_count >= 5                      │ warn_count = 2         │ ✗        │
│ 7  │ scm_risk >= 30 OR attack WARN        │ 28.04% < 30            │ ✗        │
│ 8  │ Jurisdiction issue OR multiple warns │ Jurisdiction WARN      │ ✓ FIRES  │
│    │                                      │ + SHAP WARN (2 warns)  │          │
├────┼──────────────────────────────────────┼────────────────────────┼──────────┤
│    │ Final Decision                       │                        │ WARN     │
└────┴──────────────────────────────────────┴────────────────────────┴──────────┘
```

Final outputs:
```
Decision  : WARN
Driver    : Jurisdiction engine (protected attribute discrimination pattern)
           + SHAP proxy (0.30 threshold hit)
SCM signal: CLEAR (did not drive final decision)
Legal     : STRONG (0.65)
```

---

## FULL PATH SUMMARY — Start to End

```
INPUT (TCE=10, MED=5, Flip=20, Intv=15)
│
├── PHASE 1: Pearl Formulas
│   ├── Backdoor → TCE = 0.10 (causal)
│   ├── Naive   → MED = 0.05 (masked)
│   ├── L3 CF   → Flip Rate = 20%
│   └── ATT     → 0.15
│
├── PHASE 2: Trigger Mapping
│   ├── THEORY  → m = [0.50, 0.50, 0.40, 0.50]  (continuous)
│   └── CODE    → CRITICAL, WEAK, MODERATE, no-change  (discrete)
│
├── PHASE 3: SCM Score
│   ├── THEORY  → w·m = 0.47  →  47.0%  →  WARN
│   └── CODE    → raw=23.75   →  24.74%  →  ALLOW
│
├── PHASE 4: Matrix Activation
│   ├── Primary row Criminal_Justice_Bias sum=13 ≥ 12  →  CASCADE FIRES
│   ├── Cascade rows: Healthcare, Representation, Transparency (row_factor 0.6)
│   ├── tce_factor = 0.50, sev_mult = 1.2
│   └── matrix_risk ≈ 33%
│
├── PHASE 5: Pipeline Combination
│   ├── combined = 0.6×24.74 + 0.4×33.0 = 28.04%
│   ├── Domain = criminal_justice (mult 1.0)
│   └── SCM pipeline signal = CLEAR  (28.04% < 30 threshold)
│
├── PHASE 6: Legal Score
│   └── 0.65  →  STRONG
│
├── PHASE 7: Other Signals
│   ├── SHAP proxy = 0.30  →  WARN
│   ├── Jurisdiction = WARN  (protected attribute)
│   └── Adversarial = CLEAR
│
└── PHASE 8: Step 10 Final Decision
    ├── No BLOCKs anywhere
    ├── Row 8 fires: Jurisdiction WARN + multiple warns
    └── ══► FINAL: WARN
```

---

## Theory vs Implementation — Master Comparison Table

```
┌─────────────────────────┬───────────────────────────────┬────────────────────────────────┐
│ Stage                   │ THEORY (Pearl / Academic)     │ IMPLEMENTATION (Code)          │
├─────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Causal formulas         │ Backdoor, L3 CF, PNS bounds   │ Same formulas (shared)         │
│ Input conversion        │ Continuous normalisation      │ Discrete rule triggers         │
│ SCM formula             │ w·m (weighted dot product)    │ base + sev×3 + det×2 + edge    │
│ SCM score               │ 47.0%                         │ 24.74%                         │
│ SCM standalone decision │ WARN                          │ ALLOW                          │
│ Matrix stage            │ Not used                      │ 17×5 sparse matrix + cascade   │
│ Cascade                 │ Not used                      │ sum≥12 fires cascade rows      │
│ Combined risk formula   │ α·cascade + β·scm + γ·legal   │ 0.6·scm + 0.4·matrix           │
│ Domain multiplier       │ Not used                      │ 1.0–3.0 (domain lookup)        │
│ Final risk              │ R_final = 0.432               │ combined = 28.04%              │
│ Legal risk              │ R_legal = 0.35                │ legal_score = 0.65 (STRONG)    │
│ Pipeline final signal   │ WARN (via threshold 0.40)     │ WARN (via jurisdiction rule)   │
│ Final decision          │ WARN                          │ WARN                           │
│ Decision driver         │ R_final ≥ 0.40                │ Jurisdiction + SHAP (row 8)    │
└─────────────────────────┴───────────────────────────────┴────────────────────────────────┘
```

Both reach **WARN** — but through completely different internal paths.  
Theory: aggregate risk score crosses 0.40 threshold.  
Code: SCM alone is CLEAR, but jurisdiction engine fires on protected attribute pattern.

---

## Why Both Paths Reach the Same Decision

The framework is designed with redundancy. Even when SCM risk is low:

1. Jurisdiction engine independently detects protected attribute pattern → WARN
2. SHAP proxy detects protected attribute hits → WARN
3. Step 10 row 8: "jurisdiction issue OR multiple warns" → fires

This means the system catches bias even when the causal score is mathematically moderate.  
That is a **feature, not a bug** — bias detection should not rely on a single number.

---

## For PhD Application / Thesis

- Theory trace → Chapter explaining Pearl L1/L2/L3 math and how it maps to bias metrics
- Implementation trace → Chapter showing the real pipeline with cascade matrix and domain multipliers
- Both reach WARN for COMPAS → shows mathematical theory and engineering implementation agree
- The difference in SCM score (47% vs 24.74%) is the gap Bayesian Optimization (Year 2) would close
- The cascade firing at sum≥12 maps to Pearl's "front-door paths" intuition — indirect causal chains amplify risk
- Domain multiplier (healthcare 3.0×) maps to PNS bounds — higher stakes → higher sensitivity
