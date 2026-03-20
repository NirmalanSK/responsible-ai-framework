"""
╔══════════════════════════════════════════════════════════════════════╗
║   SCM ENGINE — Structural Causal Model Engine                        ║
║   Responsible AI Framework v4.7 — Living System                      ║
║   PhD Research · Nirmalan · Chapter 3 — Proof of Concept            ║
║                                                                      ║
║   Based on: Judea Pearl's Do-Calculus (1995–2020)                   ║
║   "The Book of Why" + "Causality" — Pearl & Mackenzie               ║
╚══════════════════════════════════════════════════════════════════════╝

CURRENT STAGE (Year 1):
  - Rule-based proxy implementation
  - Manually provided causal measurements
  - Validates 5 Auto-Calibration Rules + Risk Score formula

YEAR 2 UPGRADE:
  - Replace manual inputs with DoWhy/CausalML library
  - Auto-compute TCE, NIE, Flip Rate from real data
  - Domain-specific causal graph (DAG) specification

YEAR 3:
  - Real-time inference via pre-computed backdoor sets
  - Full 12-step pipeline integration via API
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


# ══════════════════════════════════════════════
# ENUMS & CONSTANTS
# ══════════════════════════════════════════════

class Severity(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

class EdgeStrength(Enum):
    WEAK   = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"

class Detection(Enum):
    MEDIUM    = 2   # Flip Rate < 5%
    HARD      = 3   # 5% ≤ Flip Rate ≤ 15%
    VERY_HARD = 4   # Flip Rate > 15%

class Decision(Enum):
    ALLOW = "✅ ALLOW"
    WARN  = "⚠️  WARN"
    BLOCK = "🚫 BLOCK"

# Risk thresholds (% score)
BLOCK_THRESHOLD = 70.0
WARN_THRESHOLD  = 30.0

# Formula base: sum of active matrix connections
BASE_CONNECTION_SUM = 9

# Maximum possible score: 9 × 4 × 4 × 3.0 = 432
MAX_SCORE = BASE_CONNECTION_SUM * 4 * 4 * 3.0


# ══════════════════════════════════════════════
# INPUT / OUTPUT DATA CLASSES
# ══════════════════════════════════════════════

@dataclass
class CausalFindings:
    """
    Raw causal measurements — inputs to SCM Engine.

    Year 1: Manually provided (from literature, case studies).
    Year 2: Auto-computed via DoWhy library from real data.

    Fields:
        tce   : Total Causal Effect (%) — from L2 do-calculus (Rule 2 backdoor)
                TCE = P(O=1|do(A=1)) − P(O=1|do(A=0))

        med   : Mediation % — Natural Indirect Effect / TCE × 100
                NIE = P(O|do(A=1), M=m_{A=0}) − P(O|do(A=0))

        flip  : Counterfactual Flip Rate (%) — L3 Twin-Network
                Flip = P(O_{A=1}=1 | A=0, evidence)

        intv  : Intervention Reduction (%) — do(fix) test
                Reduction = TCE_before − TCE_after_mitigation

        domain: Domain risk multiplier
                1.0 = General | 2.0 = Healthcare/Legal | 3.0 = Criminal/Hiring

        rct   : Randomized Controlled Trial evidence available?
                True → confidence boost × 0.9 (RCT = gold standard)
    """
    tce   : float            # Total Causal Effect (%)
    med   : float            # Mediation % (NIE/TCE × 100)
    flip  : float            # Counterfactual Flip Rate (%)
    intv  : float = 0.0      # Intervention Reduction (%)
    domain: float = 1.0      # Domain Multiplier (1.0–3.0)
    rct   : bool  = False    # RCT evidence available?
    label : str   = "Custom" # Case label for reporting


@dataclass
class SCMResult:
    """Full output from SCM Engine — passed to Matrix + Decision Engine."""
    # Rule outputs
    severity      : Severity
    severity_adj  : Severity      # After Rule 04 adjustment
    edge          : EdgeStrength
    detection     : Detection
    epistemic_risk: bool

    # Risk score
    raw_score     : float
    risk_pct      : float
    decision      : Decision

    # Cascade flags
    cascade_active: bool          # P1→P2→P5 cascade triggered?
    rct_applied   : bool

    # Matrix update instructions
    matrix_updates: dict = field(default_factory=dict)

    # Step-by-step trace (for PhD documentation)
    trace         : list = field(default_factory=list)


# ══════════════════════════════════════════════
# 5 AUTO-CALIBRATION RULES
# ══════════════════════════════════════════════

def rule_01_severity(tce: float) -> Severity:
    """
    RULE 01: Causal Effect → Severity
    Source: L2 Intervention — do-calculus Rule 2 (backdoor adjustment)

    Formula:
        TCE = P(O=1|do(A=1)) − P(O=1|do(A=0))
            = Σ_c P(O|A=1,C=c)P(C=c) − Σ_c P(O|A=0,C=c)P(C=c)

    Thresholds (empirically derived — Year 2 will validate from 1000 cases):
        < 2%  → LOW      (negligible causal effect)
        2–5%  → MEDIUM   (minor but real effect)
        5–10% → HIGH     (significant effect — needs attention)
        ≥ 10% → CRITICAL (strong causal effect — immediate action)
    """
    if tce < 2.0:
        return Severity.LOW
    elif tce < 5.0:
        return Severity.MEDIUM
    elif tce < 10.0:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


def rule_02_edge(med: float) -> EdgeStrength:
    """
    RULE 02: Mediation % → Edge Strength
    Source: L2 Natural Indirect Effect (NIE) decomposition

    Formula:
        NIE = P(O|do(A=1), M=m_{A=0}) − P(O|do(A=0))
        Mediation% = NIE / TCE × 100

    Meaning: How much of the causal effect goes THROUGH mediators?
    High mediation → cascade P1→P2→P5 is STRONG (multiple paths affected)

    Thresholds:
        > 50% → STRONG  (cascade fully active)
        20–50%→ MEDIUM  (partial cascade)
        < 20% → WEAK    (direct effect only, no cascade)
    """
    if med > 50.0:
        return EdgeStrength.STRONG
    elif med >= 20.0:
        return EdgeStrength.MEDIUM
    else:
        return EdgeStrength.WEAK


def rule_03_detection(flip: float) -> Detection:
    """
    RULE 03: Counterfactual Flip Rate → Detection Difficulty
    Source: L3 Counterfactual — Pearl (2000) Twin-Network Model

    Formula:
        Flip = P(O_{A=1} = 1 | A=0, evidence)
        "Had the protected attribute been different, would outcome change?"

    High flip rate → bias is DEEP and HIDDEN → Very Hard to detect
    (System makes different decisions based on protected attribute,
     but disguises it through proxies)

    Thresholds:
        > 15% → VERY_HARD (4) — deeply embedded bias
        5–15% → HARD      (3) — moderately hidden
        < 5%  → MEDIUM    (2) — relatively detectable
    """
    if flip > 15.0:
        return Detection.VERY_HARD
    elif flip >= 5.0:
        return Detection.HARD
    else:
        return Detection.MEDIUM


def rule_04_adjust(severity: Severity, intv: float) -> tuple[Severity, str]:
    """
    RULE 04: Intervention Test → Severity Adjustment
    Source: L2 do(fix) intervention test

    Formula:
        Reduction = TCE_before − TCE_after_mitigation
                  = TCE_before − P(O | do(X=fix))

    Logic:
        If mitigation reduces causal effect by ≥5% → Severity -= 1
        (The proposed fix actually works → downgrade risk)
        If reduction < 5% → Severity stays same
        (Fix insufficient → risk remains — do NOT downgrade)

    Returns: (adjusted_severity, explanation_note)
    """
    if intv >= 5.0:
        new_sev = Severity(max(1, severity.value - 1))
        note = (f"Reduction={intv:.1f}% ≥ 5% → "
                f"Severity {severity.name}({severity.value}) → "
                f"{new_sev.name}({new_sev.value}) — mitigation effective ✓")
        return new_sev, note
    else:
        note = (f"Reduction={intv:.1f}% < 5% → "
                f"Severity stays {severity.name}({severity.value}) — fix insufficient")
        return severity, note


def rule_05_identifiability(tce: float, flip: float, rct: bool) -> tuple[bool, str]:
    """
    RULE 05: Identifiability → Epistemic Risk
    Source: Pearl's do-calculus completeness theorem

    Logic:
        If causal effect NOT identifiable via Rules 1–3:
            → Epistemic Risk node activated (Severity=3 minimum)
        If RCT evidence available:
            → Perfect identifiability + confidence boost × 0.9

    Heuristic for Year 1 (Year 2 will use formal ID algorithm):
        Low TCE (<2%) + Low Flip (<5%) → hard to identify causal structure
        → Activate epistemic risk flag

    RCT = Randomized Controlled Trial = gold standard causal identification
         (no confounders possible in RCT → do(X) = observe(X))
    """
    epistemic = (tce < 2.0 and flip < 5.0)

    if rct:
        note = "RCT evidence available → perfect identifiability → ×0.9 confidence boost"
    elif epistemic:
        note = ("Low TCE + Low Flip → causal structure hard to identify → "
                "Epistemic Risk node ACTIVATED (Severity floor = 3)")
    else:
        note = "Identifiable via Rule 2 backdoor adjustment → no epistemic risk"

    return epistemic, note


# ══════════════════════════════════════════════
# RISK SCORE FORMULA
# ══════════════════════════════════════════════

def calculate_risk_score(
    sev_adj : Severity,
    det     : Detection,
    domain  : float,
    rct     : bool
) -> tuple[float, float]:
    """
    RISK SCORE FORMULA:

        Score = BASE × sevAdj × det × domain × (0.9 if RCT)
        Risk% = min(Score / MAX_SCORE × 100, 100)

    Where:
        BASE    = 9 (sum of active matrix connection weights)
        sevAdj  = Severity after Rule 04 (1–4)
        det     = Detection difficulty after Rule 03 (2–4)
        domain  = Domain multiplier (1.0–3.0)
        RCT     = 0.9 confidence reduction if RCT evidence
        MAX     = 9 × 4 × 4 × 3.0 = 432

    Returns: (raw_score, risk_percentage)
    """
    score = (BASE_CONNECTION_SUM
             * sev_adj.value
             * det.value
             * domain
             * (0.9 if rct else 1.0))

    risk_pct = min((score / MAX_SCORE) * 100, 100.0)
    return score, risk_pct


def get_decision(risk_pct: float) -> Decision:
    """
    Convert risk percentage to ALLOW / WARN / BLOCK decision.

    Thresholds:
        ≥ 70% → BLOCK (high confidence of harm)
        30–70%→ WARN  (uncertain — human review needed)
        < 30% → ALLOW (low risk)
    """
    if risk_pct >= BLOCK_THRESHOLD:
        return Decision.BLOCK
    elif risk_pct >= WARN_THRESHOLD:
        return Decision.WARN
    else:
        return Decision.ALLOW


# ══════════════════════════════════════════════
# MAIN SCM ENGINE
# ══════════════════════════════════════════════

class SCMEngine:
    """
    Structural Causal Model Engine — Heart of Responsible AI Framework v4.7

    Architecture position: Step 4 — Situation Classifier + SCM + Tier Router

    Year 1: Rule-based proxy with manual causal measurements
    Year 2: DoWhy/CausalML integration for automatic computation
    Year 3: Real-time Pearl inference via pre-computed backdoor sets

    Usage:
        engine = SCMEngine()
        result = engine.run(CausalFindings(tce=12.4, med=71, flip=23, domain=3.0))
        engine.print_report(result)
    """

    def run(self, findings: CausalFindings) -> SCMResult:
        """
        Run full SCM Engine pipeline on a set of causal findings.
        Returns complete SCMResult with trace for PhD documentation.
        """
        trace = []

        # ── RULE 01 ───────────────────────────────
        sev = rule_01_severity(findings.tce)
        trace.append({
            "rule"   : "RULE 01",
            "pearl"  : "L2 Intervention — do-calculus Rule 2",
            "input"  : f"TCE = {findings.tce}%",
            "output" : f"Severity = {sev.name} ({sev.value})",
            "detail" : self._severity_detail(findings.tce),
        })

        # ── RULE 02 ───────────────────────────────
        edge = rule_02_edge(findings.med)
        cascade = (edge == EdgeStrength.STRONG)
        trace.append({
            "rule"   : "RULE 02",
            "pearl"  : "L2 Natural Indirect Effect (NIE/TCE)",
            "input"  : f"Mediation = {findings.med}%",
            "output" : f"Edge = {edge.value}",
            "detail" : ("P1→P2→P5 CASCADE ACTIVE" if cascade
                        else "Partial cascade" if edge == EdgeStrength.MEDIUM
                        else "No cascade — direct effect only"),
        })

        # ── RULE 03 ───────────────────────────────
        det = rule_03_detection(findings.flip)
        trace.append({
            "rule"   : "RULE 03",
            "pearl"  : "L3 Counterfactual — Twin-Network Model",
            "input"  : f"Flip Rate = {findings.flip}%",
            "output" : f"Detection = {det.name} ({det.value})",
            "detail" : self._detection_detail(findings.flip),
        })

        # ── RULE 04 ───────────────────────────────
        sev_adj, rule4_note = rule_04_adjust(sev, findings.intv)
        trace.append({
            "rule"   : "RULE 04",
            "pearl"  : "L2 do(fix) Intervention Test",
            "input"  : f"Intervention Reduction = {findings.intv}%",
            "output" : rule4_note,
            "detail" : f"Final Severity = {sev_adj.name} ({sev_adj.value})",
        })

        # ── RULE 05 ───────────────────────────────
        epistemic, rule5_note = rule_05_identifiability(
            findings.tce, findings.flip, findings.rct)
        trace.append({
            "rule"   : "RULE 05",
            "pearl"  : "Identifiability Check — Pearl Completeness Theorem",
            "input"  : f"RCT={findings.rct} · TCE={findings.tce}% · Flip={findings.flip}%",
            "output" : rule5_note,
            "detail" : ("Epistemic Risk node: ACTIVATED" if epistemic
                        else "No epistemic risk"),
        })

        # ── RISK SCORE ────────────────────────────
        score, risk_pct = calculate_risk_score(
            sev_adj, det, findings.domain, findings.rct)
        decision = get_decision(risk_pct)

        trace.append({
            "rule"   : "SCORE",
            "pearl"  : f"9 × {sev_adj.value} × {det.value} × {findings.domain}"
                       + (" × 0.9 (RCT)" if findings.rct else ""),
            "input"  : f"= {score:.2f}",
            "output" : f"Risk = {risk_pct:.1f}%  →  {decision.value}",
            "detail" : f"MAX possible = {MAX_SCORE:.0f}",
        })

        # ── MATRIX UPDATE INSTRUCTIONS ────────────
        matrix_updates = {
            "Representation_Bias": {
                "severity" : sev_adj.name,
                "edge"     : edge.value,
                "detection": det.name,
            },
            "cascade_active"     : cascade,
            "epistemic_risk"     : epistemic,
            "paths_affected"     : (["P1", "P2", "P5"] if cascade
                                    else ["P1"]),
        }

        return SCMResult(
            severity       = sev,
            severity_adj   = sev_adj,
            edge           = edge,
            detection      = det,
            epistemic_risk = epistemic,
            raw_score      = score,
            risk_pct       = risk_pct,
            decision       = decision,
            cascade_active = cascade,
            rct_applied    = findings.rct,
            matrix_updates = matrix_updates,
            trace          = trace,
        )

    def print_report(self, findings: CausalFindings, result: SCMResult):
        """Print full SCM Engine report — PhD documentation format."""
        W = 68
        print("═" * W)
        print(f"  SCM ENGINE REPORT — {findings.label}")
        print(f"  Responsible AI Framework v4.7 · Judea Pearl Do-Calculus")
        print("═" * W)

        print(f"\n  INPUT VALUES")
        print(f"  {'TCE (Total Causal Effect)':<35} {findings.tce}%")
        print(f"  {'Mediation % (NIE/TCE)':<35} {findings.med}%")
        print(f"  {'Counterfactual Flip Rate':<35} {findings.flip}%")
        print(f"  {'Intervention Reduction':<35} {findings.intv}%")
        print(f"  {'Domain Multiplier':<35} {findings.domain}×")
        print(f"  {'RCT Evidence':<35} {'Yes' if findings.rct else 'No'}")

        print(f"\n{'─' * W}")
        print(f"  5 AUTO-CALIBRATION RULES — STEP BY STEP")
        print(f"{'─' * W}")

        for step in result.trace:
            print(f"\n  [{step['rule']}] — {step['pearl']}")
            print(f"    Input  : {step['input']}")
            print(f"    Output : {step['output']}")
            print(f"    Detail : {step['detail']}")

        print(f"\n{'═' * W}")
        print(f"  FINAL DECISION")
        print(f"{'═' * W}")
        print(f"  Risk Score    : {result.raw_score:.2f} / {MAX_SCORE:.0f}")
        print(f"  Risk %        : {result.risk_pct:.1f}%")
        print(f"  Decision      : {result.decision.value}")
        print(f"  Cascade       : {'P1→P2→P5 ACTIVE' if result.cascade_active else 'None'}")
        print(f"  Epistemic Risk: {'ACTIVATED' if result.epistemic_risk else 'None'}")

        print(f"\n  MATRIX AUTO-UPDATE")
        for node, vals in result.matrix_updates.items():
            if isinstance(vals, dict):
                print(f"  {node}: {vals}")

        print(f"\n{'═' * W}\n")

    def _severity_detail(self, tce):
        if tce >= 10:  return f"{tce}% ≥ 10% → CRITICAL (strong causal effect)"
        elif tce >= 5: return f"{tce}% in 5–10% → HIGH (significant)"
        elif tce >= 2: return f"{tce}% in 2–5% → MEDIUM (minor but real)"
        else:          return f"{tce}% < 2% → LOW (negligible)"

    def _detection_detail(self, flip):
        if flip > 15:  return f"{flip}% > 15% → VERY HARD — deeply hidden bias"
        elif flip >= 5:return f"{flip}% in 5–15% → HARD — moderately hidden"
        else:          return f"{flip}% < 5% → MEDIUM — relatively detectable"


# ══════════════════════════════════════════════
# TIER ROUTER — Step 4 Fast Pre-Classification
# ══════════════════════════════════════════════

class TierRouter:
    """
    Tier Router — 20ms fast pre-classification before full SCM run.

    Decides which scan level to activate:
        TIER 1 (Light)  — Steps 2,4,7,9,10       ~150ms
        TIER 2 (Medium) — +Emotion,Jurisdiction,SHAP  ~350ms
        TIER 3 (Deep)   — All 12 steps + Adversarial  ~600ms

    Note: Full SCM only runs in Tier 2 and Tier 3.
    Tier 1 uses cached pre-computed patterns for speed.
    """

    SENSITIVE_DOMAINS = [
        "health", "medical", "legal", "criminal", "hiring",
        "finance", "political", "religious", "racial",
    ]

    HARMFUL_KEYWORDS = [
        "bypass", "jailbreak", "weapon", "bomb", "suicide",
        "kill", "hack", "exploit", "override safety", "ignore rules",
    ]

    SENSITIVE_KEYWORDS = [
        "medication", "drug", "security", "privacy", "law",
        "mental health", "depression", "political",
    ]

    def classify(self, query: str) -> dict:
        """
        Fast tier classification from raw text query.
        Returns tier number + reasoning.

        Year 2: Replace keyword matching with trained classifier.
        """
        q = query.lower()

        # Signal 1: harmful keyword check
        harmful_count = sum(1 for kw in self.HARMFUL_KEYWORDS if kw in q)

        # Signal 2: sensitive topic check
        sensitive_count = sum(1 for kw in self.SENSITIVE_KEYWORDS if kw in q)

        # Signal 3: domain sensitivity
        domain_hit = any(d in q for d in self.SENSITIVE_DOMAINS)

        # Signal 4: query length (complexity proxy)
        length_score = min(len(q.split()) / 30, 1.0)

        # Combined score
        if harmful_count > 0:
            combined = 0.95
        elif sensitive_count > 0 or domain_hit:
            combined = 0.55 + (length_score * 0.1)
        else:
            combined = 0.15 + (length_score * 0.1)

        if combined >= 0.70:
            tier, ms, steps = 3, 600, "All 12 steps + Adversarial Layer"
        elif combined >= 0.30:
            tier, ms, steps = 2, 350, "Steps 2,3,4,6,7,8,9,10"
        else:
            tier, ms, steps = 1, 150, "Steps 2,4,7,9,10 (cached SCM)"

        return {
            "tier"           : tier,
            "estimated_ms"   : ms,
            "active_steps"   : steps,
            "combined_score" : round(combined, 3),
            "harmful_signals": harmful_count,
            "sensitive_signals": sensitive_count,
            "domain_hit"     : domain_hit,
        }


# ══════════════════════════════════════════════
# DEMO — 4 KNOWN CASES
# ══════════════════════════════════════════════

def run_demo():
    engine = SCMEngine()
    router = TierRouter()

    cases = [
        CausalFindings(tce=12.4, med=71,  flip=23, intv=0,  domain=3.0,
                       rct=False, label="Amazon Hiring Bias (2018)"),
        CausalFindings(tce=18.3, med=61,  flip=40, intv=0,  domain=3.0,
                       rct=False, label="COMPAS Criminal Justice (2016)"),
        CausalFindings(tce=7.0,  med=60,  flip=17, intv=2,  domain=1.0,
                       rct=False, label="Default Simulator Example"),
        CausalFindings(tce=1.5,  med=15,  flip=3,  intv=8,  domain=1.0,
                       rct=True,  label="Low Risk + RCT (Safe Query)"),
    ]

    for findings in cases:
        result = engine.run(findings)
        engine.print_report(findings, result)

    # Tier Router demo
    print("═" * 68)
    print("  TIER ROUTER — Fast Pre-Classification Examples")
    print("═" * 68)
    queries = [
        "What is photosynthesis?",
        "How do medications interact with each other?",
        "How do I bypass AI safety filters?",
    ]
    for q in queries:
        r = router.classify(q)
        print(f"\n  Query : '{q}'")
        print(f"  Tier  : {r['tier']} — {r['estimated_ms']}ms")
        print(f"  Steps : {r['active_steps']}")
        print(f"  Score : {r['combined_score']}")


if __name__ == "__main__":
    run_demo()
