"""
╔══════════════════════════════════════════════════════════════════════════╗
║   ETHICS CODE — Human Decision Verification Layer                        ║
║   Responsible AI Framework v15 · PhD Research · Nirmalan                 ║
║                                                                          ║
║   Purpose: Constitutional ethics principles used by                      ║
║            HumanDecisionVerifier to evaluate human reviewer decisions.   ║
║                                                                          ║
║   Design philosophy:                                                     ║
║     "Who watches the watchman?" — Even human reviewers are subject       ║
║      to ethical consistency checks. This is the ethics constitution      ║
║      that all decisions (human AND AI) must satisfy.                     ║
║                                                                          ║
║   Pearl L3 framing:                                                      ║
║     Each principle encodes a causal rule:                                ║
║     do(human_decision=X) → must not violate principle Y                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ══════════════════════════════════════════════════════
# ETHICS PRINCIPLE ENUM
# ══════════════════════════════════════════════════════

class EthicsPrinciple(Enum):
    """
    Five foundational ethics principles — derived from:
    - EU AI Act (Articles 9, 13, 14)
    - IEEE Ethically Aligned Design
    - Anthropic Constitutional AI
    - Stoyanovich et al. Responsible AI frameworks

    These apply universally — jurisdiction-independent.
    A human reviewer ALLOW decision must satisfy all five.
    """
    HARM_PREVENTION = "harm_prevention"
    FAIRNESS        = "fairness"
    AUTONOMY        = "autonomy"
    TRANSPARENCY    = "transparency"
    ACCOUNTABILITY  = "accountability"


class ViolationSeverity(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"


# ══════════════════════════════════════════════════════
# ETHICS VIOLATION — dataclass
# ══════════════════════════════════════════════════════

@dataclass
class EthicsViolation:
    """
    Represents a detected violation of an ethics principle
    by a human reviewer's decision.

    Fields:
        principle       : Which EthicsPrinciple was violated
        severity        : LOW / MEDIUM / HIGH / CRITICAL
        description     : What the violation is
        recommendation  : What should happen next
        causal_path     : Pearl L3 causal explanation
                          e.g. "do(ALLOW) → harm_risk=0.82 → HARM_PREVENTION violated"

    PhD thesis use:
        EthicsViolation instances form the audit trail.
        Aggregate across queries → systematic reviewer bias detection.
    """
    principle      : EthicsPrinciple
    severity       : ViolationSeverity
    description    : str
    recommendation : str
    causal_path    : str = ""


# ══════════════════════════════════════════════════════
# ETHICS CODE — Full rule set
# ══════════════════════════════════════════════════════

class EthicsCode:
    """
    Constitutional ethics code for human reviewer decisions.

    Each principle has:
        - description     : What it means
        - thresholds      : Numeric triggers
        - violation_check : Logic description (implemented in HumanDecisionVerifier)

    Usage:
        code = EthicsCode()
        rules = code.get_rule(EthicsPrinciple.HARM_PREVENTION)
    """

    RULES = {

        EthicsPrinciple.HARM_PREVENTION: {
            "description": (
                "Human decisions must not materially increase the probability "
                "of harm. An ALLOW decision on a high-risk query violates this "
                "principle if no strong mitigating reason is provided."
            ),
            # Risk thresholds for violation detection
            "allow_risk_warn_threshold"    : 0.50,   # ALLOW when AI risk ≥ 50% → MEDIUM violation
            "allow_risk_high_threshold"    : 0.65,   # ALLOW when AI risk ≥ 65% → HIGH violation
            "allow_risk_critical_threshold": 0.80,   # ALLOW when AI risk ≥ 80% → CRITICAL violation
            # BLOCK at low risk: over-restriction
            "block_risk_low_threshold"     : 0.15,   # BLOCK when AI risk < 15% → LOW violation
            "causal_template": (
                "do(human_decision=ALLOW) → harm_probability={risk:.2f} → "
                "HARM_PREVENTION violated (risk exceeds {threshold:.2f} threshold)"
            ),
        },

        EthicsPrinciple.FAIRNESS: {
            "description": (
                "Decisions must be consistent across similar risk profiles. "
                "A reviewer who ALLOWs high-risk content should not BLOCK "
                "comparable low-risk content based on non-risk factors. "
                "Prevents reviewer bias based on query framing, perceived "
                "authority, or identity-coded language."
            ),
            # Protected attribute keywords — BLOCK on these without risk justification
            # triggers a fairness audit flag
            "protected_keywords": [
                "race", "gender", "religion", "caste", "nationality",
                "ethnicity", "sexual orientation", "disability", "age",
                "political opinion",
            ],
            # If query contains protected keywords AND decision = BLOCK without reason
            "flag_block_without_reason_on_protected": True,
            "causal_template": (
                "do(human_decision={decision}) on protected-attribute query "
                "without documented reason → FAIRNESS violation"
            ),
        },

        EthicsPrinciple.AUTONOMY: {
            "description": (
                "User autonomy must be respected unless harm risk exceeds threshold. "
                "Blocking low-risk educational or research queries suppresses "
                "legitimate knowledge access — a rights violation. "
                "ALLOW decisions on high-risk queries override autonomy to protect "
                "third parties — this is ethically justified only at thresholds below."
            ),
            # Only BLOCK (override autonomy) when risk exceeds this
            "autonomy_override_justified_threshold": 0.55,
            # BLOCK below this threshold = unjustified autonomy suppression
            "autonomy_suppression_threshold"       : 0.20,
            "causal_template": (
                "do(human_decision=BLOCK) on low-risk query (risk={risk:.2f}) "
                "→ unjustified autonomy suppression → AUTONOMY violated"
            ),
        },

        EthicsPrinciple.TRANSPARENCY: {
            "description": (
                "Human reviewers must provide a documented reason for non-default "
                "decisions. Default = follow AI recommendation. "
                "Overriding AI recommendation (ALLOW on BLOCK-risk or BLOCK on "
                "ALLOW-risk) without reason → lacks auditability. "
                "EU AI Act Article 14: human oversight must be documentable."
            ),
            # Risk gap above which reason is REQUIRED
            "reason_required_gap_threshold": 0.35,   # Gap > 35% → must have reason
            # Minimum reason length (characters)
            "min_reason_length"            : 10,     # At least 10 chars = real reason
            "causal_template": (
                "do(human_decision={decision}) with risk_gap={gap:.2f} "
                "and no documented reason → TRANSPARENCY violated "
                "(EU AI Act Article 14)"
            ),
        },

        EthicsPrinciple.ACCOUNTABILITY: {
            "description": (
                "High-risk ALLOW decisions must trigger a second senior review. "
                "A single reviewer approving high-risk content creates a "
                "single point of accountability failure. "
                "Pearl L3: do(single_ALLOW, high_risk) → accountability_gap=True"
            ),
            # ALLOW at this risk level → senior review mandatory
            "senior_review_threshold": 0.70,
            # ALLOW with this counterfactual delta → senior review mandatory
            "senior_review_cf_delta" : 0.50,
            "causal_template": (
                "do(human_decision=ALLOW) with risk={risk:.2f} ≥ {threshold:.2f} "
                "→ single-reviewer approval of high-risk content → "
                "ACCOUNTABILITY violated (senior review required)"
            ),
        },
    }

    def get_rule(self, principle: EthicsPrinciple) -> dict:
        """Return the ethics rule dictionary for a given principle."""
        return self.RULES[principle]

    def get_all_principles(self) -> List[EthicsPrinciple]:
        """Return all principles in check order."""
        return list(EthicsPrinciple)

    def format_violation_summary(self, violations: List[EthicsViolation]) -> str:
        """
        Format ethics violations as a readable audit string.

        Example output:
            "2 violations: HARM_PREVENTION [CRITICAL], TRANSPARENCY [MEDIUM]"
        """
        if not violations:
            return "No ethics violations"
        parts = [f"{v.principle.value.upper()} [{v.severity.value}]"
                 for v in violations]
        return f"{len(violations)} violation(s): {', '.join(parts)}"


# ══════════════════════════════════════════════════════
# HUMAN DECISION — Input dataclass
# ══════════════════════════════════════════════════════

@dataclass
class HumanReviewInput:
    """
    Represents a human reviewer's decision on an EXPERT_REVIEW case.

    Fields:
        decision     : "ALLOW" or "BLOCK"
        reason       : Reviewer's documented justification (required for audit)
        reviewer_id  : Anonymized reviewer identifier (for accountability)
        timestamp    : When the decision was made (epoch seconds)
        confidence   : Reviewer's self-reported confidence 0.0–1.0 (optional)

    Usage:
        human_input = HumanReviewInput(
            decision="ALLOW",
            reason="Verified as legitimate medical education query",
            reviewer_id="reviewer_A1",
            confidence=0.85,
        )
    """
    decision    : str          # "ALLOW" or "BLOCK"
    reason      : str  = ""   # Reviewer's documented justification
    reviewer_id : str  = ""   # Anonymized ID
    timestamp   : float = 0.0
    confidence  : float = 0.0  # Self-reported 0.0–1.0

    def __post_init__(self):
        if self.decision not in ("ALLOW", "BLOCK"):
            raise ValueError(
                f"HumanReviewInput.decision must be 'ALLOW' or 'BLOCK', "
                f"got: '{self.decision}'"
            )
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

    @property
    def implied_risk(self) -> float:
        """
        Risk score implied by the human decision.
        ALLOW → reviewer believes risk is LOW (≈0.10)
        BLOCK → reviewer believes risk is HIGH (≈0.90)

        Used in RiskGapAnalyzer for gap computation.
        """
        return 0.10 if self.decision == "ALLOW" else 0.90

    @property
    def has_reason(self) -> bool:
        """True if reviewer provided a meaningful reason (>10 chars)."""
        return len(self.reason.strip()) >= 10
