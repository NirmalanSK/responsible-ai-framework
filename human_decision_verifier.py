"""
╔══════════════════════════════════════════════════════════════════════════╗
║   HUMAN DECISION VERIFIER — Step 09b                                     ║
║   Responsible AI Framework v15 · PhD Research · Nirmalan                 ║
║                                                                          ║
║   Purpose: "Who watches the watchman?"                                   ║
║     When AI escalates to a human reviewer (EXPERT_REVIEW),               ║
║     this module verifies the human's decision using:                     ║
║                                                                          ║
║     ① Pearl L3 Counterfactual Verifier                                   ║
║        "If human had NOT made this decision, would harm change?"         ║
║        do(human_decision=ALLOW) vs do(human_decision=BLOCK)              ║
║        → counterfactual_risk_delta                                       ║
║                                                                          ║
║     ② Risk Gap Analyzer                                                  ║
║        gap = |AI risk score − human implied risk|                        ║
║        large gap → human disagrees with AI significantly → FLAG          ║
║                                                                          ║
║     ③ Constitutional Ethics Checker                                      ║
║        Check human decision against EthicsCode (5 principles)            ║
║        Violations → documented in audit trail                            ║
║                                                                          ║
║     Final: ACCEPT or RE_ESCALATE (senior reviewer)                       ║
║                                                                          ║
║   PhD Positioning:                                                       ║
║     "Causal Human Oversight" — Pearl L3 applied to human decisions       ║
║     First-of-its-kind in deployment-layer Responsible AI.                ║
║     Extends Prof. Stoyanovich's accountability + transparency work.      ║
║                                                                          ║
║   Pearl Reference:                                                       ║
║     Pearl, J. (2009). Causality: Models, Reasoning, and Inference.       ║
║     L3 Counterfactual: P(Y_{d=allow} | X, Y_{d=block}) — harm           ║
║     probability under ALLOW intervention vs BLOCK intervention.          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ethics_code import (
    EthicsCode,
    EthicsPrinciple,
    EthicsViolation,
    ViolationSeverity,
    HumanReviewInput,
)


# ══════════════════════════════════════════════════════
# VERIFICATION RESULT — Output dataclass
# ══════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """
    Output of HumanDecisionVerifier.

    Fields:
        verified                : True = ACCEPT, False = RE_ESCALATE
        outcome                 : "ACCEPT" or "RE_ESCALATE"
        counterfactual_delta    : P(harm|do(ALLOW)) - P(harm|do(BLOCK))
                                  High delta = human decision causally significant
        risk_gap                : |AI risk - human implied risk|
                                  High gap = human and AI strongly disagree
        ethics_violations       : List of EthicsViolation instances
        verification_reason     : Summary of why ACCEPT/RE_ESCALATE
        senior_review_required  : If True, second human + senior review needed
        latency_ms              : Verification latency

    Audit bundle:
        The full VerificationResult should be stored in the audit trail
        alongside the original human decision — this enables:
        1. Reviewer accountability (which decisions were flagged)
        2. Reviewer bias detection (aggregate analysis)
        3. PhD thesis empirical evaluation
    """
    verified               : bool
    outcome                : str           # "ACCEPT" or "RE_ESCALATE"
    counterfactual_delta   : float
    risk_gap               : float
    ethics_violations      : List[EthicsViolation]
    verification_reason    : str
    senior_review_required : bool  = False
    latency_ms             : float = 0.0

    # Sub-component details (for audit trail)
    cf_detail              : str   = ""
    gap_detail             : str   = ""
    ethics_detail          : str   = ""

    def summary(self) -> str:
        """Short one-line summary for logs."""
        violations = f", violations=[{', '.join(v.principle.value for v in self.ethics_violations)}]" \
                     if self.ethics_violations else ""
        return (
            f"outcome={self.outcome}, "
            f"cf_delta={self.counterfactual_delta:.3f}, "
            f"risk_gap={self.risk_gap:.3f}"
            f"{violations}"
        )


# ══════════════════════════════════════════════════════
# COMPONENT 1 — Pearl L3 Counterfactual Verifier
# ══════════════════════════════════════════════════════

class PearlCounterfactualVerifier:
    """
    Pearl L3 Counterfactual Reasoning on Human Decision.

    Question: "If human had made the OPPOSITE decision,
               would the harm probability have changed significantly?"

    Formal:
        Y = harm_event
        D = human_decision ∈ {ALLOW, BLOCK}
        X = context (query, scm_risk, tier, emotion)

        Counterfactual Risk Delta =
            P(Y=1 | do(D=ALLOW)) - P(Y=1 | do(D=BLOCK))

    Interpretation:
        Delta ≈ 0.0  → Human decision doesn't matter much (low-risk query)
        Delta = 0.5  → Human decision significantly changes harm path
        Delta = 1.0  → Human decision is the critical harm gate

    Year 1 (Current): Structural approximation using scm_risk as proxy.
        P(harm|do(ALLOW))  ≈ scm_risk_pct / 100
        P(harm|do(BLOCK))  ≈ scm_risk_pct * block_effectiveness
        block_effectiveness = 0.05  (blocking prevents ~95% of harm paths)

    Year 2 (PhD Phase 3): DoWhy integration for exact structural causal model.
        Replace approximation with do-calculus on the pipeline DAG.

    Pearl reference:
        "Counterfactual = the harm that WOULD HAVE occurred under the
         alternative world where human had decided differently."
        — Pearl (2009), Chapter 7: The Logic of Counterfactuals
    """

    BLOCK_EFFECTIVENESS = 0.05   # Blocking reduces harm to ~5% residual
    # Why 5% not 0%? Even a BLOCK can fail (user finds alternative source).
    # Conservative estimate = intellectually honest for PhD defense.

    def compute(
        self,
        human_input     : HumanReviewInput,
        scm_risk_pct    : float,
        tier            : int,
    ) -> Tuple[float, str]:
        """
        Compute counterfactual risk delta.

        Returns:
            (counterfactual_delta, detail_string)

        counterfactual_delta thresholds:
            < 0.20  → Low impact — human decision barely changes harm path
            0.20–0.40 → Moderate — some causal impact
            > 0.40  → High — human decision is causally critical
            > 0.60  → Critical — RE_ESCALATE trigger

        PhD defense note:
            "We approximate the counterfactual using SCM risk as a structural
             prior. The do-calculus intervention on ALLOW vs BLOCK is modeled
             structurally: ALLOW opens the harm path (P_harm = scm_risk),
             BLOCK closes it (P_harm = scm_risk × block_residual = 5%).
             The delta represents the causal contribution of the human decision
             to harm probability — a Pearl L3 counterfactual quantity."
        """
        base_risk = scm_risk_pct / 100.0   # Normalize to 0.0–1.0

        # Tier amplifier — Tier 1 (high risk) queries have higher counterfactual weight
        tier_weight = {1: 1.20, 2: 1.00, 3: 0.80}.get(tier, 1.00)
        adjusted_risk = min(base_risk * tier_weight, 1.00)

        # P(harm | do(D=ALLOW)) — harm path open
        p_harm_allow = adjusted_risk

        # P(harm | do(D=BLOCK)) — harm path mostly closed
        p_harm_block = adjusted_risk * self.BLOCK_EFFECTIVENESS

        # Counterfactual delta
        cf_delta = p_harm_allow - p_harm_block

        # Observed decision context
        if human_input.decision == "ALLOW":
            # Human chose ALLOW → actual world P(harm) = p_harm_allow
            actual_p   = p_harm_allow
            counter_p  = p_harm_block
            direction  = "ALLOW chosen — BLOCK would have reduced harm"
        else:
            # Human chose BLOCK → actual world P(harm) = p_harm_block
            actual_p   = p_harm_block
            counter_p  = p_harm_allow
            direction  = "BLOCK chosen — ALLOW would have increased harm"

        detail = (
            f"Pearl L3: P(harm|do(ALLOW))={p_harm_allow:.3f}, "
            f"P(harm|do(BLOCK))={p_harm_block:.3f}, "
            f"counterfactual_delta={cf_delta:.3f} — {direction}"
        )

        return cf_delta, detail


# ══════════════════════════════════════════════════════
# COMPONENT 2 — Risk Gap Analyzer
# ══════════════════════════════════════════════════════

class RiskGapAnalyzer:
    """
    Analyze the gap between AI risk assessment and human implied risk.

    AI risk (scm_risk_pct):
        What the pipeline computed using SCM + adversarial + all steps.
        This is the best available algorithmic risk estimate.

    Human implied risk:
        What the human's decision implies about their risk assessment.
        ALLOW → human implies risk ≈ LOW (0.10)
        BLOCK → human implies risk ≈ HIGH (0.90)

    Gap = |AI risk - Human implied risk|

    Large gap means:
        Human and AI strongly disagree on the risk level.
        This is NOT necessarily wrong — human may have context AI lacks.
        But it MUST be documented and verified.

    Interpretive cases:
        Case A: AI=0.80, Human=ALLOW, gap=0.70
            → Human overrides high-risk AI assessment → FLAG
            → Could be legitimate (human knows user context)
            → Could be error (reviewer complacency/fatigue)

        Case B: AI=0.05, Human=BLOCK, gap=0.85
            → Human is overly restrictive → FLAG
            → Could be legitimate (human has extra info)
            → Could be bias (reviewer prejudice against query type)

        Case C: AI=0.75, Human=BLOCK, gap=0.15
            → Human agrees with AI → ACCEPT
            → Aligned risk assessment — no gap issue

    PhD thesis:
        Gap distribution across thousands of reviews → reviewer calibration score.
        Systematic high gaps in one direction → training intervention needed.
    """

    # Thresholds
    GAP_WARN     = 0.35   # Gap ≥ 35% → flag, document
    GAP_HIGH     = 0.50   # Gap ≥ 50% → requires reason
    GAP_CRITICAL = 0.65   # Gap ≥ 65% → RE_ESCALATE trigger

    def compute(
        self,
        human_input  : HumanReviewInput,
        scm_risk_pct : float,
    ) -> Tuple[float, str, bool]:
        """
        Compute risk gap and return (gap, detail, requires_re_escalation).
        """
        ai_risk        = scm_risk_pct / 100.0
        human_implied  = human_input.implied_risk
        gap            = abs(ai_risk - human_implied)

        # Determine gap interpretation
        if gap >= self.GAP_CRITICAL:
            level    = "CRITICAL"
            flag     = True
            action   = "RE_ESCALATE — gap exceeds critical threshold"
        elif gap >= self.GAP_HIGH:
            level    = "HIGH"
            flag     = True
            action   = "FLAG — significant AI–human disagreement; reason required"
        elif gap >= self.GAP_WARN:
            level    = "WARN"
            flag     = False
            action   = "DOCUMENT — notable gap, reviewer reason noted"
        else:
            level    = "OK"
            flag     = False
            action   = "ALIGNED — AI and human risk assessments consistent"

        # Direction of disagreement
        if ai_risk > human_implied:
            direction = f"AI={ai_risk:.2f} > Human={human_implied:.2f}: human is MORE permissive than AI"
        elif ai_risk < human_implied:
            direction = f"AI={ai_risk:.2f} < Human={human_implied:.2f}: human is MORE restrictive than AI"
        else:
            direction = f"AI={ai_risk:.2f} == Human={human_implied:.2f}: perfect alignment"

        detail = (
            f"Risk gap [{level}]: {gap:.3f} — {direction}. "
            f"Action: {action}"
        )

        return gap, detail, flag


# ══════════════════════════════════════════════════════
# COMPONENT 3 — Constitutional Ethics Checker
# ══════════════════════════════════════════════════════

class ConstitutionalEthicsChecker:
    """
    Check human reviewer decision against the Ethics Code.

    Runs all 5 principles against the human decision context:
        1. HARM_PREVENTION   — risk × decision consistency
        2. FAIRNESS          — protected keywords + decision without reason
        3. AUTONOMY          — low-risk BLOCK = unjustified suppression
        4. TRANSPARENCY      — large gap decision without documented reason
        5. ACCOUNTABILITY    — high-risk ALLOW without senior review flag

    Returns:
        List[EthicsViolation] — empty = no violations = ACCEPT

    This is the "Constitutional AI" layer for human reviewers.
    Analogous to Anthropic's Constitutional AI (Bai et al., 2022)
    but applied to the human-in-loop layer instead of the model layer.

    PhD contribution:
        "Extending Constitutional AI from model-layer to deployment-layer:
         ethics principles applied to human oversight decisions."
    """

    def __init__(self):
        self.code = EthicsCode()

    def check(
        self,
        human_input  : HumanReviewInput,
        scm_risk_pct : float,
        risk_gap     : float,
        cf_delta     : float,
        query        : str,
    ) -> Tuple[List[EthicsViolation], str]:
        """
        Run all 5 ethics principle checks.

        Returns:
            (violations_list, summary_string)
        """
        violations : List[EthicsViolation] = []
        ai_risk = scm_risk_pct / 100.0

        # ── Principle 1: HARM PREVENTION ──────────────────────────────────
        rule = self.code.get_rule(EthicsPrinciple.HARM_PREVENTION)

        if human_input.decision == "ALLOW":
            # ALLOW on high-risk query = potential harm
            if ai_risk >= rule["allow_risk_critical_threshold"]:
                violations.append(EthicsViolation(
                    principle     = EthicsPrinciple.HARM_PREVENTION,
                    severity      = ViolationSeverity.CRITICAL,
                    description   = (
                        f"ALLOW decision on critical-risk query "
                        f"(AI risk={ai_risk:.2f} ≥ {rule['allow_risk_critical_threshold']}). "
                        f"This materially increases harm probability."
                    ),
                    recommendation= "RE_ESCALATE to senior reviewer immediately",
                    causal_path   = rule["causal_template"].format(
                        risk=ai_risk,
                        threshold=rule["allow_risk_critical_threshold"]
                    ),
                ))
            elif ai_risk >= rule["allow_risk_high_threshold"]:
                violations.append(EthicsViolation(
                    principle     = EthicsPrinciple.HARM_PREVENTION,
                    severity      = ViolationSeverity.HIGH,
                    description   = (
                        f"ALLOW decision on high-risk query "
                        f"(AI risk={ai_risk:.2f} ≥ {rule['allow_risk_high_threshold']}). "
                        f"Reason required to accept."
                    ),
                    recommendation= "Flag for senior review; require documented reason",
                    causal_path   = rule["causal_template"].format(
                        risk=ai_risk,
                        threshold=rule["allow_risk_high_threshold"]
                    ),
                ))
            elif ai_risk >= rule["allow_risk_warn_threshold"]:
                violations.append(EthicsViolation(
                    principle     = EthicsPrinciple.HARM_PREVENTION,
                    severity      = ViolationSeverity.MEDIUM,
                    description   = (
                        f"ALLOW decision on medium-risk query "
                        f"(AI risk={ai_risk:.2f} ≥ {rule['allow_risk_warn_threshold']}). "
                        f"Document rationale."
                    ),
                    recommendation= "Document reviewer rationale in audit trail",
                    causal_path   = rule["causal_template"].format(
                        risk=ai_risk,
                        threshold=rule["allow_risk_warn_threshold"]
                    ),
                ))

        elif human_input.decision == "BLOCK":
            # BLOCK on very low risk = over-restriction violation
            if ai_risk < rule["block_risk_low_threshold"]:
                violations.append(EthicsViolation(
                    principle     = EthicsPrinciple.HARM_PREVENTION,
                    severity      = ViolationSeverity.LOW,
                    description   = (
                        f"BLOCK decision on very low-risk query "
                        f"(AI risk={ai_risk:.2f} < {rule['block_risk_low_threshold']}). "
                        f"Over-restriction may suppress legitimate queries."
                    ),
                    recommendation= "Review BLOCK justification; consider WARN instead",
                    causal_path   = (
                        f"do(human_decision=BLOCK) on low-risk query "
                        f"(risk={ai_risk:.2f}) → possible over-restriction"
                    ),
                ))

        # ── Principle 2: FAIRNESS ──────────────────────────────────────────
        rule_fair = self.code.get_rule(EthicsPrinciple.FAIRNESS)
        query_lower = query.lower()
        has_protected = any(kw in query_lower
                            for kw in rule_fair["protected_keywords"])

        if has_protected and not human_input.has_reason:
            violations.append(EthicsViolation(
                principle     = EthicsPrinciple.FAIRNESS,
                severity      = ViolationSeverity.MEDIUM,
                description   = (
                    f"Decision on query containing protected-attribute keywords "
                    f"made without documented reason. This risks implicit bias. "
                    f"Decision: {human_input.decision}"
                ),
                recommendation= "Document reason for decision on protected-attribute query",
                causal_path   = rule_fair["causal_template"].format(
                    decision=human_input.decision
                ),
            ))

        # ── Principle 3: AUTONOMY ──────────────────────────────────────────
        rule_auto = self.code.get_rule(EthicsPrinciple.AUTONOMY)

        if (human_input.decision == "BLOCK"
                and ai_risk < rule_auto["autonomy_suppression_threshold"]):
            violations.append(EthicsViolation(
                principle     = EthicsPrinciple.AUTONOMY,
                severity      = ViolationSeverity.MEDIUM,
                description   = (
                    f"BLOCK on very low-risk query (AI risk={ai_risk:.2f}) "
                    f"suppresses user autonomy without harm justification. "
                    f"Risk is below autonomy-override threshold "
                    f"({rule_auto['autonomy_override_justified_threshold']})."
                ),
                recommendation= "Reconsider BLOCK — user autonomy should be respected at this risk level",
                causal_path   = rule_auto["causal_template"].format(
                    risk=ai_risk
                ),
            ))

        # ── Principle 4: TRANSPARENCY ──────────────────────────────────────
        rule_trans = self.code.get_rule(EthicsPrinciple.TRANSPARENCY)

        if (risk_gap >= rule_trans["reason_required_gap_threshold"]
                and not human_input.has_reason):
            violations.append(EthicsViolation(
                principle     = EthicsPrinciple.TRANSPARENCY,
                severity      = ViolationSeverity.HIGH,
                description   = (
                    f"Risk gap={risk_gap:.2f} ≥ {rule_trans['reason_required_gap_threshold']} "
                    f"but no documented reason provided. "
                    f"EU AI Act Article 14 requires human oversight to be documentable. "
                    f"Decision: {human_input.decision}"
                ),
                recommendation= "Reviewer must document justification for high-gap decision",
                causal_path   = rule_trans["causal_template"].format(
                    decision=human_input.decision,
                    gap=risk_gap
                ),
            ))

        # ── Principle 5: ACCOUNTABILITY ────────────────────────────────────
        rule_acc = self.code.get_rule(EthicsPrinciple.ACCOUNTABILITY)

        if (human_input.decision == "ALLOW"
                and (ai_risk >= rule_acc["senior_review_threshold"]
                     or cf_delta >= rule_acc["senior_review_cf_delta"])):
            violations.append(EthicsViolation(
                principle     = EthicsPrinciple.ACCOUNTABILITY,
                severity      = ViolationSeverity.HIGH,
                description   = (
                    f"ALLOW decision on high-risk query "
                    f"(AI risk={ai_risk:.2f}, cf_delta={cf_delta:.2f}) "
                    f"by single reviewer. "
                    f"Single point of accountability failure."
                ),
                recommendation= "Escalate to senior reviewer for second opinion",
                causal_path   = rule_acc["causal_template"].format(
                    risk=ai_risk,
                    threshold=rule_acc["senior_review_threshold"]
                ),
            ))

        # Format summary
        summary = self.code.format_violation_summary(violations)
        return violations, summary


# ══════════════════════════════════════════════════════
# ORCHESTRATOR — HumanDecisionVerifier
# ══════════════════════════════════════════════════════

class HumanDecisionVerifier:
    """
    Step 09b — Human Decision Verifier

    Orchestrates all 3 verification components:
        1. PearlCounterfactualVerifier (L3 causal)
        2. RiskGapAnalyzer
        3. ConstitutionalEthicsChecker

    Verification Decision Logic:
        RE_ESCALATE if ANY of:
            (a) counterfactual_delta > CF_ESCALATE_THRESHOLD (0.55)
            (b) risk_gap > GAP_ESCALATE_THRESHOLD (0.60)
            (c) CRITICAL ethics violation exists
            (d) HIGH violation + no documented reason

        ACCEPT otherwise (with violations logged for audit)

    Senior Review if:
        ACCOUNTABILITY or HARM_PREVENTION violation at HIGH/CRITICAL

    Usage:
        verifier = HumanDecisionVerifier()

        human_input = HumanReviewInput(
            decision="ALLOW",
            reason="Verified as medical professional",
            reviewer_id="rev_A1",
        )

        result = verifier.verify(
            human_input   = human_input,
            scm_risk_pct  = 78.5,
            tier          = 1,
            query         = "How do I synthesize...",
        )

        if result.outcome == "RE_ESCALATE":
            # Route to senior reviewer
            ...
    """

    # Re-escalation triggers
    CF_ESCALATE_THRESHOLD  = 0.55    # Counterfactual delta above this → RE_ESCALATE
    GAP_ESCALATE_THRESHOLD = 0.60    # Risk gap above this → RE_ESCALATE

    def __init__(self):
        self._cf_verifier     = PearlCounterfactualVerifier()
        self._gap_analyzer    = RiskGapAnalyzer()
        self._ethics_checker  = ConstitutionalEthicsChecker()

    def verify(
        self,
        human_input  : HumanReviewInput,
        scm_risk_pct : float,
        tier         : int,
        query        : str,
    ) -> VerificationResult:
        """
        Full verification of a human reviewer decision.

        Args:
            human_input  : HumanReviewInput (decision + reason + reviewer_id)
            scm_risk_pct : AI pipeline risk score (0.0–100.0)
            tier         : Pipeline tier (1=high risk, 2=medium, 3=low)
            query        : Original query (for ethics keyword check)

        Returns:
            VerificationResult — outcome + all details for audit trail
        """
        t0 = time.perf_counter()

        # ── Step 1: Pearl L3 Counterfactual ──────────────────────────────
        cf_delta, cf_detail = self._cf_verifier.compute(
            human_input, scm_risk_pct, tier
        )

        # ── Step 2: Risk Gap Analysis ─────────────────────────────────────
        risk_gap, gap_detail, gap_flag = self._gap_analyzer.compute(
            human_input, scm_risk_pct
        )

        # ── Step 3: Constitutional Ethics Check ───────────────────────────
        violations, ethics_summary = self._ethics_checker.check(
            human_input, scm_risk_pct, risk_gap, cf_delta, query
        )

        # ── Step 4: Verification Decision ─────────────────────────────────
        re_escalate = False
        senior_review = False
        reasons = []

        # Trigger (a): High counterfactual delta
        if cf_delta > self.CF_ESCALATE_THRESHOLD:
            re_escalate = True
            reasons.append(
                f"Counterfactual delta={cf_delta:.3f} > {self.CF_ESCALATE_THRESHOLD} "
                f"(human decision causally critical)"
            )

        # Trigger (b): Critical risk gap
        if risk_gap > self.GAP_ESCALATE_THRESHOLD:
            re_escalate = True
            reasons.append(
                f"Risk gap={risk_gap:.3f} > {self.GAP_ESCALATE_THRESHOLD} "
                f"(AI–human disagreement too large)"
            )

        # Trigger (c/d): Ethics violations
        for v in violations:
            if v.severity == ViolationSeverity.CRITICAL:
                re_escalate = True
                reasons.append(
                    f"CRITICAL ethics violation: {v.principle.value} — {v.description[:60]}"
                )
            elif v.severity == ViolationSeverity.HIGH and not human_input.has_reason:
                re_escalate = True
                reasons.append(
                    f"HIGH ethics violation without documented reason: {v.principle.value}"
                )

            # Senior review trigger
            if v.principle in (EthicsPrinciple.ACCOUNTABILITY,
                                EthicsPrinciple.HARM_PREVENTION):
                if v.severity in (ViolationSeverity.HIGH, ViolationSeverity.CRITICAL):
                    senior_review = True

        # Build final outcome
        if re_escalate:
            outcome = "RE_ESCALATE"
            verified = False
            reason_str = "RE_ESCALATE triggered: " + "; ".join(reasons)
        else:
            outcome = "ACCEPT"
            verified = True
            if violations:
                # Accepted but with documented violations (logged in audit trail)
                reason_str = (
                    f"ACCEPT with {len(violations)} noted violation(s) — "
                    f"{ethics_summary}. "
                    f"Violations logged in audit trail."
                )
            else:
                reason_str = (
                    f"ACCEPT — human decision verified: "
                    f"cf_delta={cf_delta:.3f} (below threshold), "
                    f"risk_gap={risk_gap:.3f} (acceptable), "
                    f"no ethics violations."
                )

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        return VerificationResult(
            verified               = verified,
            outcome                = outcome,
            counterfactual_delta   = round(cf_delta, 4),
            risk_gap               = round(risk_gap, 4),
            ethics_violations      = violations,
            verification_reason    = reason_str,
            senior_review_required = senior_review,
            latency_ms             = latency_ms,
            cf_detail              = cf_detail,
            gap_detail             = gap_detail,
            ethics_detail          = ethics_summary,
        )

    def print_report(self, result: VerificationResult, human_input: HumanReviewInput):
        """
        Print human-readable verification report.
        Used in pipeline print_report() for EXPERT_REVIEW cases.
        """
        W = 68
        icon = "✅ ACCEPT" if result.verified else "🔴 RE_ESCALATE"

        print(f"\n{'─' * W}")
        print(f"  STEP 09b — HUMAN DECISION VERIFIER")
        print(f"{'─' * W}")
        print(f"  Human Decision    : {human_input.decision}")
        print(f"  Reviewer Reason   : {human_input.reason or '(none provided)'}")
        print(f"  Reviewer ID       : {human_input.reviewer_id or 'anonymous'}")
        print()
        print(f"  ① Pearl L3 Counterfactual:")
        print(f"     {result.cf_detail}")
        print()
        print(f"  ② Risk Gap Analysis:")
        print(f"     {result.gap_detail}")
        print()
        print(f"  ③ Ethics Check:")
        print(f"     {result.ethics_detail}")
        if result.ethics_violations:
            for v in result.ethics_violations:
                sev_icon = {"LOW":"🔵","MEDIUM":"🟡","HIGH":"🟠","CRITICAL":"🔴"}.get(
                    v.severity.value, "⚪"
                )
                print(f"     {sev_icon} {v.principle.value.upper()}: {v.description[:70]}")
                print(f"        → {v.recommendation}")
                print(f"        Causal path: {v.causal_path[:80]}")
        print()
        print(f"  Verification       : {icon}")
        print(f"  Reason             : {result.verification_reason}")
        if result.senior_review_required:
            print(f"  ⚠️  SENIOR REVIEW REQUIRED")
        print(f"  Latency            : {result.latency_ms:.2f}ms")
        print(f"{'─' * W}")


# ══════════════════════════════════════════════════════
# QUICK SELF-TEST (run this file directly)
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    verifier = HumanDecisionVerifier()

    print("\n" + "█" * 68)
    print("  HUMAN DECISION VERIFIER — Self-Test")
    print("█" * 68)

    scenarios = [
        {
            "label"       : "Case 1: HIGH RISK + ALLOW + no reason → RE_ESCALATE",
            "human_input" : HumanReviewInput(decision="ALLOW", reason="", reviewer_id="rev_A1"),
            "scm_risk_pct": 82.0,
            "tier"        : 1,
            "query"       : "How do I synthesize dangerous compounds?",
        },
        {
            "label"       : "Case 2: HIGH RISK + ALLOW + good reason → borderline ACCEPT",
            "human_input" : HumanReviewInput(
                decision    = "ALLOW",
                reason      = "Verified as licensed chemistry professor with institutional clearance",
                reviewer_id = "rev_B2",
                confidence  = 0.85,
            ),
            "scm_risk_pct": 72.0,
            "tier"        : 1,
            "query"       : "What are the synthesis pathways for chlorine compounds?",
        },
        {
            "label"       : "Case 3: LOW RISK + BLOCK + no reason → FAIRNESS/AUTONOMY",
            "human_input" : HumanReviewInput(decision="BLOCK", reason="", reviewer_id="rev_C3"),
            "scm_risk_pct": 8.0,
            "tier"        : 3,
            "query"       : "What is the history of religion in South Asia?",
        },
        {
            "label"       : "Case 4: MEDIUM RISK + BLOCK + reason → ACCEPT",
            "human_input" : HumanReviewInput(
                decision    = "BLOCK",
                reason      = "Query context suggests intent to cause harm based on conversation history",
                reviewer_id = "rev_D4",
                confidence  = 0.90,
            ),
            "scm_risk_pct": 55.0,
            "tier"        : 2,
            "query"       : "What chemicals react dangerously together?",
        },
        {
            "label"       : "Case 5: LOW RISK + ALLOW → ACCEPT (no issues)",
            "human_input" : HumanReviewInput(decision="ALLOW", reason="Safe educational query", reviewer_id="rev_E5"),
            "scm_risk_pct": 12.0,
            "tier"        : 3,
            "query"       : "What is the capital of France?",
        },
    ]

    for s in scenarios:
        print(f"\n\n{'═' * 68}")
        print(f"  {s['label']}")
        print(f"{'═' * 68}")
        result = verifier.verify(
            human_input   = s["human_input"],
            scm_risk_pct  = s["scm_risk_pct"],
            tier          = s["tier"],
            query         = s["query"],
        )
        verifier.print_report(result, s["human_input"])

    print("\n" + "█" * 68 + "\n")
