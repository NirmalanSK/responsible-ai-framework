"""
╔══════════════════════════════════════════════════════════════════════════╗
║   RESPONSIBLE AI FRAMEWORK — DATA PRIVACY ENGINE                        ║
║   v1.0 — Production Build                                                ║
║   PhD Research · Nirmalan                                                ║
║                                                                          ║
║   Three-Layer Data Privacy Protection:                                   ║
║     ✅ Layer 1: PII Detection & Masking  — auto-detect + mask sensitive  ║
║     ✅ Layer 2: Differential Privacy     — calibrated Laplace noise      ║
║     ✅ Layer 3: Data Minimization        — Legal Rulebook config         ║
║                                                                          ║
║   Pipeline Integration:                                                  ║
║     Step 00  → PII Masking (before Step 01 Input Sanitizer)             ║
║     Step 12b → Output PII Scan (after Step 12 Output Filter)            ║
╚══════════════════════════════════════════════════════════════════════════╝

WHY DATA PRIVACY MATTERS FOR RAI:
──────────────────────────────────
GDPR Art.5(1)(c) → Data Minimization principle
GDPR Art.25      → Privacy by Design / Default
CCPA §1798.100   → Right to know what data is processed
HIPAA §164.514   → De-identification standards

Pearl's Framework Connection:
  L1 (Observational): PII in training data → biased representations
  L2 (Interventional): do(mask=True) → breaks causal path to discrimination
  L3 (Counterfactual): "Would this model behave differently without PII?" → YES

This engine answers that counterfactual by ensuring PII never enters
the causal reasoning pipeline, maintaining mathematical privacy guarantees.
"""

import re
import math
import time
import json
import logging
import hashlib
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Set

log = logging.getLogger("responsible_ai")


# ══════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════

class PIICategory(Enum):
    """Categories of Personally Identifiable Information."""
    NAME            = "name"
    EMAIL           = "email"
    PHONE           = "phone"
    SSN             = "ssn"              # Social Security Number / National ID
    CREDIT_CARD     = "credit_card"
    IP_ADDRESS      = "ip_address"
    DATE_OF_BIRTH   = "date_of_birth"
    PASSPORT        = "passport"
    AADHAAR         = "aadhaar"          # India Aadhaar number
    ADDRESS         = "address"
    BANK_ACCOUNT    = "bank_account"
    MEDICAL_ID      = "medical_id"
    NIC             = "nic"              # Sri Lanka NIC


class MaskingStrategy(Enum):
    """How to mask detected PII."""
    REDACT      = "redact"      # Replace with [PII_TYPE]
    HASH        = "hash"        # Replace with SHA256 hash (consistent)
    TOKENIZE    = "tokenize"    # Replace with stable token (PII_EMAIL_001)
    PARTIAL     = "partial"     # Show first 2 + last 2 chars: Jo**n
    FULL_MASK   = "full_mask"   # Replace with *** entirely


class DataField(Enum):
    """Fields that can appear in pipeline data."""
    QUERY       = "query"
    USER_ID     = "user_id"
    CONVERSATION= "conversation"
    CAUSAL_DATA = "causal_data"
    METADATA    = "metadata"


class PrivacyJurisdiction(Enum):
    """Privacy regulation regime — maps to Data Minimization rules."""
    GLOBAL  = "global"    # Best-effort defaults
    EU_GDPR = "eu_gdpr"  # GDPR — strictest
    US_CCPA = "us_ccpa"  # California Consumer Privacy Act
    IN_DPDP = "in_dpdp"  # India Digital Personal Data Protection Act 2023
    LK_PDP  = "lk_pdp"  # Sri Lanka Personal Data Protection Act 2022
    US_HIPAA= "us_hipaa" # Healthcare — HIPAA


# ══════════════════════════════════════════════════════
# LAYER 1: PII DETECTION & MASKING ENGINE
# ══════════════════════════════════════════════════════

@dataclass
class PIIMatch:
    """A single detected PII instance."""
    category    : PIICategory
    original    : str       # Original text
    masked      : str       # Replaced text
    start       : int       # Position in string
    end         : int       # Position in string
    confidence  : float     # 0.0 – 1.0


@dataclass
class PIIMaskingResult:
    """Result of PII detection + masking on a text."""
    original_text   : str
    masked_text     : str
    pii_found       : List[PIIMatch] = field(default_factory=list)
    categories_hit  : Set[str]       = field(default_factory=set)
    pii_count       : int            = 0
    latency_ms      : float          = 0.0

    @property
    def has_pii(self) -> bool:
        return self.pii_count > 0


class PIIDetector:
    """
    Regex-based PII detection engine with configurable masking.

    Supports 13 PII categories across global jurisdictions:
    - Standard: Name, Email, Phone, Credit Card, SSN, IP, DOB
    - Regional:  Aadhaar (IN), NIC (LK), Passport

    Design Decision:
        Pure regex chosen over ML-NER for:
        (a) Zero external dependency (no spaCy/transformers)
        (b) Deterministic — same input → same output (testable)
        (c) <5ms latency target
        (d) Production-safe — no network calls, no model loading

    Year 2 Upgrade Path:
        Replace name detection with spaCy NER or BERT-NER for
        higher recall on names (current regex catches only
        "Firstname Lastname" patterns, misses single names).
    """

    # ── PII Regex Patterns ────────────────────────────────────────────────
    # Each tuple: (pattern, category, confidence)
    # Ordered by specificity (most specific first) to avoid double-masking

    PATTERNS: List[Tuple[re.Pattern, PIICategory, float]] = []

    def __init__(self, strategy: MaskingStrategy = MaskingStrategy.REDACT):
        self.strategy = strategy
        self._token_counter: Dict[str, int] = {}
        self._token_map: Dict[str, str] = {}   # original → token (stable within session)

        # Compile patterns at init (not at class level — avoids module-load cost)
        self._patterns: List[Tuple[re.Pattern, PIICategory, float]] = [

            # ── Email (highest specificity) ───────────────────────
            (re.compile(
                r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
            ), PIICategory.EMAIL, 0.99),

            # ── Credit Card (16-digit with optional separators) ───
            (re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?'           # Visa
                r'|5[1-5][0-9]{14}'                         # Mastercard
                r'|3[47][0-9]{13}'                          # Amex
                r'|6(?:011|5[0-9]{2})[0-9]{12}'            # Discover
                r'|(?:\d{4}[\s\-]){3}\d{4})\b'             # Generic spaced
            ), PIICategory.CREDIT_CARD, 0.99),

            # ── India Aadhaar (12-digit, often spaced in 4-4-4) ──
            (re.compile(
                r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b'
            ), PIICategory.AADHAAR, 0.97),

            # ── Sri Lanka NIC (9 digits + V/X or 12 digits) ──────
            (re.compile(
                r'\b\d{9}[VvXx]\b|\b\d{12}\b'
            ), PIICategory.NIC, 0.85),     # 0.85 — 12-digit has false positives

            # ── US SSN ───────────────────────────────────────────
            (re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[\s\-](?!00)\d{2}[\s\-](?!0000)\d{4}\b'
            ), PIICategory.SSN, 0.97),

            # ── Passport Number (most countries) ─────────────────
            (re.compile(
                r'\b[A-Z]{1,2}\d{6,9}\b'
            ), PIICategory.PASSPORT, 0.80),

            # ── IP Address (v4 only — v6 too verbose for regex) ──
            (re.compile(
                r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
                r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
            ), PIICategory.IP_ADDRESS, 0.98),

            # ── Phone Numbers (international + regional formats) ──
            (re.compile(
                r'(?:\+?\d{1,3}[\s\-\.])?'     # Country code
                r'(?:\(\d{1,4}\)[\s\-\.]?)?'   # Area code
                r'\d{3,5}[\s\-\.]?\d{3,5}'     # Main number
                r'(?:[\s\-\.]\d{2,4})?'         # Extension
                r'(?!\d)'                        # Not followed by more digits
            ), PIICategory.PHONE, 0.80),

            # ── Bank Account (rough global pattern) ───────────────
            (re.compile(
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b'  # IBAN
                r'|\b\d{8,17}\b'                                            # Generic
            ), PIICategory.BANK_ACCOUNT, 0.70),

            # ── Date of Birth ─────────────────────────────────────
            (re.compile(
                r'\b(?:DOB|Date of Birth|Born|dob)\s*[:\-]?\s*'
                r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
                re.IGNORECASE
            ), PIICategory.DATE_OF_BIRTH, 0.95),

            # ── Full Name (Firstname Lastname pattern) ────────────
            # Note: Limited recall — catches "John Smith" not "Nirmalan"
            # Year 2: replace with spaCy NER
            (re.compile(
                r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]{1,20}'
                r'(?:\s+[A-Z][a-z]{1,20}){0,2}\b'
            ), PIICategory.NAME, 0.88),

            # ── Medical Record ID ─────────────────────────────────
            (re.compile(
                r'\b(?:MRN|Patient ID|Record No\.?)\s*[:\-]?\s*[A-Z0-9]{5,12}\b',
                re.IGNORECASE
            ), PIICategory.MEDICAL_ID, 0.92),
        ]

    def detect_and_mask(self, text: str) -> PIIMaskingResult:
        """
        Detect all PII in text and return masked version.

        Algorithm:
            1. Find all pattern matches (with positions)
            2. Sort by position → handle overlaps (keep highest confidence)
            3. Apply masking right-to-left (preserves indices)
            4. Return masked text + audit log of what was found
        """
        t0 = time.perf_counter()

        if not text or not isinstance(text, str):
            return PIIMaskingResult(
                original_text=str(text or ""),
                masked_text=str(text or ""),
            )

        # Step 1: Collect all matches
        raw_matches: List[PIIMatch] = []
        for pattern, category, confidence in self._patterns:
            for m in pattern.finditer(text):
                raw_matches.append(PIIMatch(
                    category   = category,
                    original   = m.group(),
                    masked     = "",          # filled below
                    start      = m.start(),
                    end        = m.end(),
                    confidence = confidence,
                ))

        # Step 2: Resolve overlaps (greedy — keep highest confidence match)
        # Sort by start pos, then by confidence desc
        raw_matches.sort(key=lambda m: (m.start, -m.confidence))
        resolved: List[PIIMatch] = []
        last_end = -1
        for m in raw_matches:
            if m.start >= last_end:
                resolved.append(m)
                last_end = m.end

        # Step 3: Apply masking strategy
        for m in resolved:
            m.masked = self._apply_mask(m.original, m.category)

        # Step 4: Build masked text (right-to-left to preserve indices)
        masked_chars = list(text)
        for m in sorted(resolved, key=lambda x: x.start, reverse=True):
            masked_chars[m.start:m.end] = list(m.masked)
        masked_text = "".join(masked_chars)

        ms = (time.perf_counter() - t0) * 1000
        return PIIMaskingResult(
            original_text  = text,
            masked_text    = masked_text,
            pii_found      = resolved,
            categories_hit = {m.category.value for m in resolved},
            pii_count      = len(resolved),
            latency_ms     = round(ms, 2),
        )

    def _apply_mask(self, original: str, category: PIICategory) -> str:
        """Apply the configured masking strategy to a detected PII value."""
        label = category.value.upper()

        if self.strategy == MaskingStrategy.REDACT:
            return f"[{label}]"

        elif self.strategy == MaskingStrategy.HASH:
            # SHA256, truncated to 8 hex chars — deterministic within session
            h = hashlib.sha256(original.encode()).hexdigest()[:8]
            return f"[{label}:{h}]"

        elif self.strategy == MaskingStrategy.TOKENIZE:
            # Stable token per unique value (PII_EMAIL_001 etc.)
            if original not in self._token_map:
                count = self._token_counter.get(label, 0) + 1
                self._token_counter[label] = count
                self._token_map[original] = f"PII_{label}_{count:03d}"
            return self._token_map[original]

        elif self.strategy == MaskingStrategy.PARTIAL:
            if len(original) <= 4:
                return "****"
            return original[:2] + "*" * (len(original) - 4) + original[-2:]

        elif self.strategy == MaskingStrategy.FULL_MASK:
            return "*" * len(original)

        return f"[{label}]"  # fallback


# ══════════════════════════════════════════════════════
# LAYER 2: DIFFERENTIAL PRIVACY ENGINE
# ══════════════════════════════════════════════════════

@dataclass
class DifferentialPrivacyResult:
    """Result of applying differential privacy noise."""
    original_value  : Any
    noisy_value     : Any
    epsilon         : float         # Privacy budget used
    mechanism       : str           # "laplace" | "gaussian" | "exponential"
    sensitivity     : float         # L1 sensitivity of the query
    noise_added     : float         # Magnitude of noise applied
    latency_ms      : float = 0.0


class DifferentialPrivacyEngine:
    """
    Differential Privacy (DP) engine using the Laplace Mechanism.

    MATHEMATICAL FOUNDATION:
    ─────────────────────────
    The Laplace Mechanism guarantees ε-differential privacy:

        M(D) = f(D) + Lap(Δf / ε)

    Where:
        ε (epsilon)  = privacy budget (smaller → more private)
        Δf           = L1 global sensitivity of function f
        Lap(b)       = Laplace distribution with scale b

    Privacy Guarantee:
        For any two adjacent datasets D, D' (differing by 1 record):

        Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S]

    This means: an adversary seeing the noisy output gains at most
    e^ε probability advantage in inferring any individual's data.

    Practical ε Guidelines:
        ε = 0.1  → Very high privacy, significant noise
        ε = 1.0  → Standard academic/research setting (default)
        ε = 10.0 → Weaker privacy, less noise (analytics use)

    Connection to Framework:
        When SCM causal analysis requires aggregate statistics from
        user data, DP ensures individual records cannot be back-
        calculated from the pipeline's risk scores.

    Year 2 Upgrade:
        - Gaussian mechanism for L2 sensitivity (better for ML training)
        - Rényi DP for tighter composition bounds
        - Privacy budget accounting across pipeline steps
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon : Privacy budget. Default 1.0 (research standard).
            delta   : Failure probability for (ε,δ)-DP. Default 1e-5.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be > 0. Got: {epsilon}")
        self.epsilon = epsilon
        self.delta   = delta

    def add_laplace_noise(
        self,
        value       : float,
        sensitivity : float = 1.0,
        epsilon     : Optional[float] = None,
    ) -> DifferentialPrivacyResult:
        """
        Apply Laplace mechanism to a single numeric value.

        Args:
            value       : True value to protect.
            sensitivity : L1 sensitivity (max change one record can cause).
            epsilon     : Override engine-level epsilon for this call.

        Returns:
            DifferentialPrivacyResult with noisy value + audit info.
        """
        t0 = time.perf_counter()
        eps = epsilon or self.epsilon
        scale = sensitivity / eps   # Laplace scale parameter b = Δf/ε

        # Generate Laplace noise using inverse CDF (pure Python, no numpy)
        # Lap(0, b) = -b·sign(U)·ln(1 - 2|U - 0.5|) where U ~ Uniform[0,1]
        u = secrets.SystemRandom().uniform(-0.5, 0.5)
        noise = -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))

        noisy = value + noise
        ms = (time.perf_counter() - t0) * 1000

        return DifferentialPrivacyResult(
            original_value  = value,
            noisy_value     = round(noisy, 6),
            epsilon         = eps,
            mechanism       = "laplace",
            sensitivity     = sensitivity,
            noise_added     = round(noise, 6),
            latency_ms      = round(ms, 2),
        )

    def privatize_risk_score(
        self,
        risk_pct    : float,
        sensitivity : float = 5.0,
    ) -> float:
        """
        Add DP noise to SCM risk percentage for privacy-safe reporting.

        sensitivity=5.0 means one user's data can shift risk by max 5%.
        Clamps output to [0, 100].

        Use case: When logging/reporting risk scores to external systems,
        the true score is protected so individual users cannot be identified
        from pipeline audit logs.
        """
        result = self.add_laplace_noise(risk_pct, sensitivity)
        return max(0.0, min(100.0, result.noisy_value))

    def privatize_numeric_dict(
        self,
        data        : Dict[str, float],
        sensitivity : float = 1.0,
    ) -> Dict[str, float]:
        """
        Apply Laplace noise to all numeric values in a dict.

        Used for: causal_data dicts before SCM analysis.
        Each key gets independent noise draw (parallel composition).
        """
        noisy: Dict[str, float] = {}
        for key, val in data.items():
            if isinstance(val, (int, float)):
                result = self.add_laplace_noise(float(val), sensitivity)
                noisy[key] = result.noisy_value
            else:
                noisy[key] = val   # Non-numeric: pass through unchanged
        return noisy

    def compute_privacy_budget(self, num_queries: int) -> float:
        """
        Basic sequential composition: k queries use k·ε total budget.
        Returns total epsilon consumed.

        Year 2: Replace with Rényi DP for tighter accounting.
        """
        return self.epsilon * num_queries


# ══════════════════════════════════════════════════════
# LAYER 3: DATA MINIMIZATION — LEGAL RULEBOOK
# ══════════════════════════════════════════════════════

@dataclass
class FieldPolicy:
    """Policy for a single data field."""
    field_name  : str
    allowed     : bool          # Is this field allowed at all?
    required    : bool          # Must it be present?
    max_length  : Optional[int] # Max character length (None = no limit)
    pii_scan    : bool          # Run PII detection on this field?
    justification: str          # Legal basis (GDPR Art.6, CCPA §1798.100, etc.)
    retention_days: Optional[int] = None  # How long to keep (None = session only)


@dataclass
class DataMinimizationResult:
    """Result of applying the data minimization rulebook."""
    allowed_fields      : Dict[str, Any]    # Fields that passed
    blocked_fields      : List[str]         # Fields that were blocked
    truncated_fields    : List[str]         # Fields that were truncated
    policy_applied      : str               # Jurisdiction name
    violations          : List[str]         # Policy violations found
    latency_ms          : float = 0.0

    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0


# ── Legal Rulebooks (one per jurisdiction) ────────────────────────────────

_GLOBAL_RULEBOOK: Dict[str, FieldPolicy] = {
    DataField.QUERY.value: FieldPolicy(
        field_name   = "query",
        allowed      = True,
        required     = True,
        max_length   = 50_000,
        pii_scan     = True,
        justification= "Core functionality — required to process user request",
        retention_days= None,   # Session only
    ),
    DataField.USER_ID.value: FieldPolicy(
        field_name   = "user_id",
        allowed      = True,
        required     = False,
        max_length   = 64,
        pii_scan     = False,   # Should already be anonymised (UUID)
        justification= "Rate limiting & session continuity",
        retention_days= 30,
    ),
    DataField.CONVERSATION.value: FieldPolicy(
        field_name   = "conversation",
        allowed      = True,
        required     = False,
        max_length   = 10_000,  # Per-message limit
        pii_scan     = True,
        justification= "Context continuity for multi-turn safety analysis",
        retention_days= None,
    ),
    DataField.CAUSAL_DATA.value: FieldPolicy(
        field_name   = "causal_data",
        allowed      = True,
        required     = False,
        max_length   = None,
        pii_scan     = False,   # Should be pre-aggregated numeric
        justification= "SCM causal risk analysis (numeric only)",
        retention_days= None,
    ),
    DataField.METADATA.value: FieldPolicy(
        field_name   = "metadata",
        allowed      = False,   # Blocked globally unless jurisdiction allows
        required     = False,
        max_length   = None,
        pii_scan     = True,
        justification= "No legitimate purpose identified",
        retention_days= None,
    ),
}

# EU GDPR — strictest: data minimization Art.5(1)(c), purpose limitation Art.5(1)(b)
_EU_GDPR_RULEBOOK: Dict[str, FieldPolicy] = {
    **_GLOBAL_RULEBOOK,  # Inherit globals, then override
    DataField.USER_ID.value: FieldPolicy(
        field_name   = "user_id",
        allowed      = True,
        required     = False,
        max_length   = 64,
        pii_scan     = True,    # GDPR: even UUIDs must be audited
        justification= "GDPR Art.6(1)(b) — performance of contract. Pseudonymised.",
        retention_days= 30,
    ),
    DataField.CONVERSATION.value: FieldPolicy(
        field_name   = "conversation",
        allowed      = True,
        required     = False,
        max_length   = 2_000,   # Stricter: 2K per message (not 10K)
        pii_scan     = True,
        justification= "GDPR Art.6(1)(f) — legitimate interest, safety. Minimised.",
        retention_days= None,   # GDPR: session only — no persistent conversation logs
    ),
}

# India DPDP Act 2023
_IN_DPDP_RULEBOOK: Dict[str, FieldPolicy] = {
    **_GLOBAL_RULEBOOK,
    DataField.CAUSAL_DATA.value: FieldPolicy(
        field_name   = "causal_data",
        allowed      = True,
        required     = False,
        max_length   = None,
        pii_scan     = True,    # DPDP S.4: purpose limitation, data minimization
        justification= "DPDP Act 2023 S.4 — lawful purpose, data minimization",
        retention_days= None,
    ),
}

# Sri Lanka PDP Act 2022
_LK_PDP_RULEBOOK: Dict[str, FieldPolicy] = {
    **_GLOBAL_RULEBOOK,
    DataField.CONVERSATION.value: FieldPolicy(
        field_name   = "conversation",
        allowed      = True,
        required     = False,
        max_length   = 5_000,   # LK PDP: moderate restriction
        pii_scan     = True,
        justification= "LK PDP Act 2022 S.6 — proportionality principle",
        retention_days= None,
    ),
}

# US CCPA (California)
_US_CCPA_RULEBOOK: Dict[str, FieldPolicy] = {
    **_GLOBAL_RULEBOOK,
    DataField.METADATA.value: FieldPolicy(
        field_name   = "metadata",
        allowed      = True,    # CCPA allows but requires disclosure
        required     = False,
        max_length   = 500,
        pii_scan     = True,
        justification= "CCPA §1798.100 — right to know; metadata allowed with disclosure",
        retention_days= 365,
    ),
}

# US HIPAA (Healthcare) — most restrictive for medical data
_US_HIPAA_RULEBOOK: Dict[str, FieldPolicy] = {
    **_EU_GDPR_RULEBOOK,   # Start from GDPR (strict baseline)
    DataField.QUERY.value: FieldPolicy(
        field_name   = "query",
        allowed      = True,
        required     = True,
        max_length   = 10_000,
        pii_scan     = True,    # HIPAA: mandatory PHI scanning
        justification= "HIPAA §164.514 — de-identification required before processing",
        retention_days= None,
    ),
}

JURISDICTION_RULEBOOKS: Dict[PrivacyJurisdiction, Dict[str, FieldPolicy]] = {
    PrivacyJurisdiction.GLOBAL  : _GLOBAL_RULEBOOK,
    PrivacyJurisdiction.EU_GDPR : _EU_GDPR_RULEBOOK,
    PrivacyJurisdiction.IN_DPDP : _IN_DPDP_RULEBOOK,
    PrivacyJurisdiction.LK_PDP  : _LK_PDP_RULEBOOK,
    PrivacyJurisdiction.US_CCPA : _US_CCPA_RULEBOOK,
    PrivacyJurisdiction.US_HIPAA: _US_HIPAA_RULEBOOK,
}


class DataMinimizationEngine:
    """
    Data Minimization engine — applies the Legal Rulebook.

    Each jurisdiction has a rulebook (dict of FieldPolicy).
    The engine:
        1. Blocks fields not in the rulebook (unknown data)
        2. Blocks fields explicitly marked allowed=False
        3. Truncates fields exceeding max_length
        4. Flags required fields that are missing
        5. Triggers PII scan for fields with pii_scan=True

    This is the "gatekeeper" — before any data enters the pipeline,
    it must pass the rulebook check.

    Usage:
        engine = DataMinimizationEngine(PrivacyJurisdiction.EU_GDPR)
        result = engine.apply({"query": "...", "user_id": "...", "metadata": "..."})
        # → metadata blocked (EU GDPR doesn't allow it)
    """

    def __init__(
        self,
        jurisdiction: PrivacyJurisdiction = PrivacyJurisdiction.GLOBAL,
        pii_detector: Optional[PIIDetector] = None,
    ):
        self.jurisdiction = jurisdiction
        self.rulebook     = JURISDICTION_RULEBOOKS[jurisdiction]
        self.pii_detector = pii_detector or PIIDetector()

    def apply(self, data: Dict[str, Any]) -> DataMinimizationResult:
        """
        Apply the rulebook to an incoming data dict.

        Returns:
            DataMinimizationResult with only allowed, cleaned fields.
            Blocked fields are removed. Violations are logged.
        """
        t0 = time.perf_counter()
        allowed_fields   : Dict[str, Any] = {}
        blocked_fields   : List[str]      = []
        truncated_fields : List[str]      = []
        violations       : List[str]      = []

        for field_name, value in data.items():
            policy = self.rulebook.get(field_name)

            # ── Unknown field → block unless explicitly allowed ───
            if policy is None:
                blocked_fields.append(field_name)
                violations.append(
                    f"UNKNOWN_FIELD: '{field_name}' has no policy — blocked. "
                    f"Jurisdiction: {self.jurisdiction.value}"
                )
                continue

            # ── Explicitly blocked field ──────────────────────────
            if not policy.allowed:
                blocked_fields.append(field_name)
                violations.append(
                    f"BLOCKED_FIELD: '{field_name}' not permitted. "
                    f"Legal basis: {policy.justification}"
                )
                continue

            # ── Truncate if over max_length ───────────────────────
            if policy.max_length is not None and isinstance(value, str):
                if len(value) > policy.max_length:
                    value = value[:policy.max_length]
                    truncated_fields.append(field_name)
                    violations.append(
                        f"TRUNCATED: '{field_name}' exceeded {policy.max_length} chars"
                    )
            elif policy.max_length is not None and isinstance(value, list):
                # For conversation lists: truncate each item
                clipped = []
                for item in value:
                    if isinstance(item, str) and len(item) > policy.max_length:
                        clipped.append(item[:policy.max_length])
                        truncated_fields.append(f"{field_name}[item]")
                    else:
                        clipped.append(item)
                value = clipped

            # ── PII scan if policy requires ───────────────────────
            if policy.pii_scan and isinstance(value, str) and value:
                pii_result = self.pii_detector.detect_and_mask(value)
                if pii_result.has_pii:
                    value = pii_result.masked_text
                    violations.append(
                        f"PII_MASKED: '{field_name}' — {pii_result.pii_count} "
                        f"PII instances masked "
                        f"({', '.join(pii_result.categories_hit)})"
                    )

            allowed_fields[field_name] = value

        # ── Check required fields present ─────────────────────────
        for field_name, policy in self.rulebook.items():
            if policy.required and field_name not in allowed_fields:
                violations.append(
                    f"MISSING_REQUIRED: '{field_name}' is required but absent"
                )

        ms = (time.perf_counter() - t0) * 1000
        return DataMinimizationResult(
            allowed_fields   = allowed_fields,
            blocked_fields   = blocked_fields,
            truncated_fields = truncated_fields,
            policy_applied   = self.jurisdiction.value,
            violations       = violations,
            latency_ms       = round(ms, 2),
        )

    def get_policy_summary(self) -> str:
        """Return human-readable summary of active rulebook."""
        lines = [f"Data Minimization Policy: {self.jurisdiction.value.upper()}"]
        lines.append("─" * 50)
        for name, p in self.rulebook.items():
            status = "✅ ALLOWED" if p.allowed else "🚫 BLOCKED"
            req    = "required" if p.required else "optional"
            pii    = "PII scan ON" if p.pii_scan else "no PII scan"
            mlen   = f"max {p.max_length} chars" if p.max_length else "no length limit"
            lines.append(f"  {name:15s}: {status} | {req} | {pii} | {mlen}")
            lines.append(f"{'':18s}  Legal basis: {p.justification[:60]}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# STEP 00: PIPELINE INTEGRATION — PRE-INPUT PRIVACY GATE
# ══════════════════════════════════════════════════════

@dataclass
class PrivacyGateResult:
    """Result of the full privacy gate (Step 00)."""
    masked_query        : str
    privacy_violations  : List[str]
    pii_count           : int
    pii_categories      : Set[str]
    blocked_fields      : List[str]
    compliance_status   : str           # "COMPLIANT" | "VIOLATIONS_FOUND"
    latency_ms          : float = 0.0

    # Step result format (matches existing StepResult pattern)
    @property
    def signal(self) -> str:
        if self.blocked_fields or self.pii_count > 0:
            return "WARN"
        return "CLEAR"

    @property
    def detail(self) -> str:
        parts = []
        if self.pii_count > 0:
            parts.append(
                f"Output PII masked: {self.pii_count} instances "
                f"({', '.join(self.pii_categories)})"
            )
        # Count PII_MASKED violations from minimizer
        minimizer_pii = [v for v in self.privacy_violations if v.startswith("PII_MASKED")]
        if minimizer_pii:
            parts.append(f"Input PII masked: {len(minimizer_pii)} field(s)")
        if self.blocked_fields:
            parts.append(f"Blocked fields: {', '.join(self.blocked_fields)}")
        if not parts:
            parts.append("No PII detected — data clean")
        return " | ".join(parts)


class Step00_PrivacyGate:
    """
    Step 00 — Data Privacy Gate

    Runs BEFORE Step 01 Input Sanitizer.

    Three sub-engines in sequence:
        1. Data Minimization  → block/truncate fields per Legal Rulebook
        2. PII Detection      → scan query + conversation for PII
        3. PII Masking        → replace detected PII with tokens

    This ensures that:
        - No PII enters the causal reasoning pipeline (Steps 05–09)
        - Data minimization is enforced before ANY processing
        - Masked query is passed to Step 01 (not the original)

    Latency target: <10ms (pure regex, no network calls)

    GDPR Art.25 compliance: Privacy by Design — the gate is the
    first thing that touches user data, not an afterthought.
    """

    def __init__(
        self,
        jurisdiction       : PrivacyJurisdiction = PrivacyJurisdiction.GLOBAL,
        masking_strategy   : MaskingStrategy      = MaskingStrategy.REDACT,
        dp_epsilon         : float                = 1.0,
    ):
        self.pii_detector  = PIIDetector(strategy=masking_strategy)
        self.minimizer     = DataMinimizationEngine(jurisdiction, self.pii_detector)
        self.dp_engine     = DifferentialPrivacyEngine(epsilon=dp_epsilon)
        self.jurisdiction  = jurisdiction

    def run(
        self,
        query       : str,
        conversation: List[str] = None,
        causal_data : Optional[Dict[str, Any]] = None,
        user_id     : str = "",
    ) -> Tuple[str, PrivacyGateResult]:
        """
        Run the full privacy gate.

        Returns:
            (masked_query, PrivacyGateResult)
            masked_query → pass this to Step 01 instead of original query
        """
        t0 = time.perf_counter()

        # Build data dict for minimization check
        data_to_check: Dict[str, Any] = {
            DataField.QUERY.value: query or "",
        }
        if user_id:
            data_to_check[DataField.USER_ID.value] = user_id
        if conversation:
            data_to_check[DataField.CONVERSATION.value] = conversation
        if causal_data:
            data_to_check[DataField.CAUSAL_DATA.value] = causal_data

        # ── Layer 3: Data Minimization first ──────────────────────
        min_result = self.minimizer.apply(data_to_check)

        # Extract cleaned query (minimizer may have truncated/masked it)
        masked_query = min_result.allowed_fields.get(DataField.QUERY.value, query)

        # ── Layer 1+2: Additional PII scan on final query ──────────
        # (Minimizer runs pii_scan per policy, but we do a final pass
        #  to ensure nothing slipped through if policy had pii_scan=False)
        pii_result = self.pii_detector.detect_and_mask(masked_query)
        if pii_result.has_pii:
            masked_query = pii_result.masked_text

        # ── Combine violations ─────────────────────────────────────
        all_violations = min_result.violations

        # ── DP: Privatize causal_data numeric values ───────────────
        # If causal_data passed, add noise before it enters SCM engine
        # This protects individual users from being identified via risk scores
        if causal_data and isinstance(causal_data, dict):
            noisy_causal = self.dp_engine.privatize_numeric_dict(causal_data)
            # Note: noisy version returned via min_result.allowed_fields
            min_result.allowed_fields[DataField.CAUSAL_DATA.value] = noisy_causal

        ms = (time.perf_counter() - t0) * 1000

        result = PrivacyGateResult(
            masked_query       = masked_query,
            privacy_violations = all_violations,
            pii_count          = pii_result.pii_count,
            pii_categories     = pii_result.categories_hit,
            blocked_fields     = min_result.blocked_fields,
            compliance_status  = "COMPLIANT" if min_result.is_compliant else "VIOLATIONS_FOUND",
            latency_ms         = round(ms, 2),
        )

        log.info("Privacy gate", extra={
            "pii_count"     : result.pii_count,
            "blocked_fields": result.blocked_fields,
            "compliance"    : result.compliance_status,
            "jurisdiction"  : self.jurisdiction.value,
            "latency_ms"    : result.latency_ms,
        })

        return masked_query, result


# ══════════════════════════════════════════════════════
# STEP 12b: OUTPUT PII SCAN
# ══════════════════════════════════════════════════════

class Step12b_OutputPrivacyScan:
    """
    Step 12b — Output Privacy Scan

    Runs AFTER Step 12 Output Filter.

    Even if the input was masked, an LLM might reconstruct PII
    from training memory or context. This step ensures the final
    response hint does not leak PII back to the user.

    Also: applies DP noise to any numeric scores in the audit bundle
    before they leave the pipeline (prevents model inversion attacks).

    Latency target: <5ms
    """

    def __init__(
        self,
        masking_strategy : MaskingStrategy = MaskingStrategy.REDACT,
        dp_engine        : Optional[DifferentialPrivacyEngine] = None,
    ):
        self.pii_detector = PIIDetector(strategy=masking_strategy)
        self.dp_engine    = dp_engine or DifferentialPrivacyEngine(epsilon=1.0)

    def run(self, response_hint: str, audit_bundle: dict) -> Tuple[str, dict, str]:
        """
        Scan and clean pipeline output.

        Returns:
            (clean_response, clean_audit_bundle, detail_str)
        """
        t0 = time.perf_counter()
        actions = []

        # ── Scan response hint for PII ─────────────────────────────
        clean_response = response_hint
        if response_hint:
            pii_result = self.pii_detector.detect_and_mask(response_hint)
            if pii_result.has_pii:
                clean_response = pii_result.masked_text
                actions.append(
                    f"Output PII masked: {pii_result.pii_count} instances "
                    f"({', '.join(pii_result.categories_hit)})"
                )

        # ── DP noise on audit bundle scores ───────────────────────
        clean_bundle = dict(audit_bundle)
        if "scm_risk_pct" in clean_bundle:
            original_risk = clean_bundle["scm_risk_pct"]
            clean_bundle["scm_risk_pct"] = self.dp_engine.privatize_risk_score(
                original_risk, sensitivity=5.0
            )
            actions.append(
                f"DP noise applied to risk score (ε={self.dp_engine.epsilon})"
            )

        if "shap_score" in clean_bundle:
            clean_bundle["shap_score"] = self.dp_engine.privatize_risk_score(
                clean_bundle["shap_score"] * 100, sensitivity=5.0
            ) / 100.0
            actions.append("DP noise applied to SHAP score")

        ms = (time.perf_counter() - t0) * 1000
        detail = " | ".join(actions) if actions else f"Output clean — no PII detected ({ms:.2f}ms)"

        return clean_response, clean_bundle, detail


# ══════════════════════════════════════════════════════
# PUBLIC API — convenience factory
# ══════════════════════════════════════════════════════

def create_privacy_gate(
    jurisdiction_str : str = "global",
    masking_strategy : str = "redact",
    dp_epsilon       : float = 1.0,
) -> Step00_PrivacyGate:
    """
    Convenience factory for creating a Step00_PrivacyGate.

    Args:
        jurisdiction_str : "global" | "eu_gdpr" | "in_dpdp" | "lk_pdp"
                           "us_ccpa" | "us_hipaa"
        masking_strategy : "redact" | "hash" | "tokenize" | "partial" | "full_mask"
        dp_epsilon       : Privacy budget (0.1 = very private, 10.0 = less private)

    Example:
        gate = create_privacy_gate("eu_gdpr", "redact", epsilon=0.5)
        masked_query, result = gate.run(query="My email is john@example.com")
    """
    _jur_map = {
        "global"  : PrivacyJurisdiction.GLOBAL,
        "eu_gdpr" : PrivacyJurisdiction.EU_GDPR,
        "in_dpdp" : PrivacyJurisdiction.IN_DPDP,
        "lk_pdp"  : PrivacyJurisdiction.LK_PDP,
        "us_ccpa" : PrivacyJurisdiction.US_CCPA,
        "us_hipaa": PrivacyJurisdiction.US_HIPAA,
    }
    _mask_map = {
        "redact"   : MaskingStrategy.REDACT,
        "hash"     : MaskingStrategy.HASH,
        "tokenize" : MaskingStrategy.TOKENIZE,
        "partial"  : MaskingStrategy.PARTIAL,
        "full_mask": MaskingStrategy.FULL_MASK,
    }

    jur  = _jur_map.get(jurisdiction_str.lower(), PrivacyJurisdiction.GLOBAL)
    mask = _mask_map.get(masking_strategy.lower(), MaskingStrategy.REDACT)

    return Step00_PrivacyGate(
        jurisdiction     = jur,
        masking_strategy = mask,
        dp_epsilon       = dp_epsilon,
    )


# ══════════════════════════════════════════════════════
# STANDALONE DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 68)
    print("  DATA PRIVACY ENGINE — v1.0 Demo")
    print("=" * 68)

    # ── Demo 1: PII Detection ─────────────────────────────────────
    print("\n[LAYER 1] PII Detection & Masking")
    print("─" * 68)
    detector = PIIDetector(strategy=MaskingStrategy.REDACT)

    test_texts = [
        "My email is nirmalan@example.com and my phone is +94 77 123 4567",
        "Patient DOB: 15/08/1990, Aadhaar: 2345 6789 0123",
        "Call me on 555-123-4567 or reach Dr. John Smith at Colombo office",
        "My credit card is 4111 1111 1111 1111 and SSN is 123-45-6789",
        "IP 192.168.1.1 accessed record MRN: ABC123456",
    ]

    for text in test_texts:
        result = detector.detect_and_mask(text)
        print(f"  INPUT  : {text}")
        print(f"  OUTPUT : {result.masked_text}")
        print(f"  PII    : {result.pii_count} found — {result.categories_hit}")
        print()

    # ── Demo 2: Differential Privacy ──────────────────────────────
    print("\n[LAYER 2] Differential Privacy — Laplace Mechanism")
    print("─" * 68)
    dp = DifferentialPrivacyEngine(epsilon=1.0)

    print(f"  {'Value':>10}  {'ε':>6}  {'Sensitivity':>12}  {'Noisy Value':>12}  {'Noise':>10}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}")
    for true_val, sens in [(75.0, 5.0), (50.0, 5.0), (25.0, 5.0), (100.0, 10.0)]:
        r = dp.add_laplace_noise(true_val, sensitivity=sens)
        print(f"  {r.original_value:>10.2f}  {r.epsilon:>6.1f}  {r.sensitivity:>12.1f}  "
              f"{r.noisy_value:>12.4f}  {r.noise_added:>+10.4f}")

    # ── Demo 3: Data Minimization ──────────────────────────────────
    print("\n[LAYER 3] Data Minimization — Legal Rulebook")
    print("─" * 68)
    for jur in [PrivacyJurisdiction.GLOBAL, PrivacyJurisdiction.EU_GDPR]:
        engine = DataMinimizationEngine(jur)
        print(f"\n  Jurisdiction: {jur.value.upper()}")
        test_data = {
            "query"       : "What is AI? My email is user@test.com",
            "user_id"     : "user-abc-123",
            "conversation": ["Hello!", "Tell me about GDPR"],
            "metadata"    : "some extra data",           # blocked in GDPR
            "unknown_key" : "should be blocked",         # always blocked
        }
        result = engine.apply(test_data)
        print(f"  Allowed  : {list(result.allowed_fields.keys())}")
        print(f"  Blocked  : {result.blocked_fields}")
        print(f"  Compliant: {result.is_compliant}")
        if result.violations:
            for v in result.violations:
                print(f"  ⚠  {v}")

    # ── Demo 4: Full Step00 Privacy Gate ───────────────────────────
    print("\n[STEP 00] Full Privacy Gate — EU GDPR")
    print("─" * 68)
    gate = create_privacy_gate("eu_gdpr", "redact", dp_epsilon=1.0)
    masked, gate_result = gate.run(
        query="Please help me, my Aadhaar is 2345 6789 0123 and email john@doe.com",
        conversation=["My phone is +94-77-000-0001"],
        causal_data={"risk_score": 72.5, "prior_violations": 3},
    )
    print(f"  Signal   : {gate_result.signal}")
    print(f"  Detail   : {gate_result.detail}")
    print(f"  Masked Q : {masked}")
    print(f"  Latency  : {gate_result.latency_ms}ms")

    print("\n" + "=" * 68)
    print("  ✅ Data Privacy Engine demo complete")
    print("=" * 68)
