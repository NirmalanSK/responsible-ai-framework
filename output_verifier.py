# output_verifier.py
# ═══════════════════════════════════════════════════════════════════════════════
# Two-Pass LLM Self-Verification — Model-Agnostic Implementation
#
# Status  : DESIGN COMPLETE — Year 2 pipeline integration target (S14)
# Built   : April 2026 (v15i — proof-of-concept at chatbot layer)
# Extracted: April 2026 — decoupled from Groq/Gemini chatbot into standalone module
#
# Current state:
#   self_verify() lives in governed_chatbot.py (Groq) and
#   gemini_governed_chatbot.py — duplicated, model-specific.
#
# Year 2 target (S14):
#   pipeline_v15.py calls output_verifier.verify(draft, result, llm_fn)
#   after S13 (Output Privacy Scan). Chatbots remove their own self_verify()
#   and delegate to this module. Single source of truth.
#
# Year 2 integration plan (research_notes.md — "Architectural Solution: S14"):
#   Step 1: ✅ Extract self_verify() → this file (model-agnostic)
#   Step 2: Add S14 call in pipeline_v15.py after decision engine
#             result = pipeline.run(query)              # S00–S13
#             verified = output_verifier.verify(        # S14
#                 draft    = llm_draft,
#                 result   = result,
#                 llm_fn   = your_llm_callable,
#             )
#   Step 3: Both chatbots remove self_verify() — use S14 instead
#   Step 4: New test class: TestS14OutputVerifier
#
# Usage (TODAY — chatbot layer, pipeline untouched):
#   from output_verifier import build_rai_context, build_system_prompt, verify
#   ctx   = build_rai_context(pipeline_result, decision_name)
#   sys_p, user_msg = build_system_prompt(query, ctx)
#   draft = your_llm(system=sys_p, user=user_msg)
#   final, was_revised, issues, decision_ok = verify(query, draft, ctx, your_llm_fn)
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Output of the two-pass self-verification.

    Attributes:
        final_response   : Approved draft or revised response (ready for user).
        was_revised      : True if LLM found issues and revised the draft.
        issues_found     : Description of what was wrong, or "None".
        decision_correct : "YES" | "SHOULD_BE_WARN" | "YES_DEFINITELY"
                           Meaningful only for BLOCK decisions — signals whether
                           verifier thinks BLOCK should be downgraded to WARN.
        status_tag       : "✓ self-verified" | "🔄 self-verified+revised" | "⚠ verify-error"
    """
    final_response   : str
    was_revised      : bool
    issues_found     : str
    decision_correct : str
    status_tag       : str


# ── Context builder ───────────────────────────────────────────────────────────

def build_rai_context(
    result,                          # PipelineResult (duck-typed — no pipeline import)
    decision_name   : str,
    cum_override    : bool = False,
    cum_reason      : str  = "",
) -> dict:
    """
    Extract SCM/Matrix/attack findings from PipelineResult for LLM consumption.

    Accepts any object with .steps, .scm_risk_pct, .attack_type, .emotion_flag,
    .vac_triggered, .jurisdiction attributes — matches PipelineResult exactly.
    Duck-typed to avoid circular import with pipeline_v15.py.
    """
    flagged_steps = {}
    for step in getattr(result, "steps", []):
        if getattr(step, "signal", "") in ("BLOCK", "WARN", "ESCALATE"):
            flagged_steps[step.step_name] = step.detail

    return {
        "decision"           : decision_name,
        "risk_pct"           : round(getattr(result, "scm_risk_pct", 0.0), 1),
        "attack_type"        : getattr(result.attack_type, "value", str(result.attack_type)),
        "emotion_flag"       : getattr(result.emotion_flag, "value", str(result.emotion_flag)),
        "vac_triggered"      : getattr(result, "vac_triggered", False),
        "jurisdiction"       : getattr(result.jurisdiction, "value", str(result.jurisdiction)),
        "flagged_steps"      : flagged_steps,
        "cumulative_override": cum_override,
        "cumulative_reason"  : cum_reason,
    }


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(query: str, ctx: dict) -> Tuple[str, str]:
    """
    Build (system_prompt, user_message) for LLM based on RAI decision.

    BLOCK         → LLM explains why — does NOT answer harmful query
    WARN          → LLM gives careful causal-aware response
    EXPERT_REVIEW → LLM explains uncertainty + expert needed
    ALLOW         → Normal response; bias-aware if SCM flagged risk

    Returns:
        (system_prompt, user_message) — pass both to your LLM callable.
    """
    decision   = ctx["decision"]
    risk       = ctx["risk_pct"]
    vac        = ctx["vac_triggered"]
    jur        = ctx["jurisdiction"]
    attack     = ctx["attack_type"]
    flagged    = ctx["flagged_steps"]
    cum        = ctx["cumulative_override"]
    cum_reason = ctx["cumulative_reason"]

    findings = "\n".join(
        f"  - {name}: {detail}" for name, detail in flagged.items()
    ) or "  - Pattern-based harm detection triggered"

    if decision == "BLOCK":
        system = f"""You are the explanation layer of a Responsible AI governance framework.

A user query was BLOCKED. Your job:
1. Explain WHY it was blocked using the RAI findings below — be clear and respectful
2. Do NOT answer or hint at the harmful content in any way
3. Briefly suggest a safe alternative if one exists

RAI Findings:
  - Risk Score      : {risk}%
  - Attack Detected : {attack}
  - VAC Triggered   : {"YES — absolute ethical violation" if vac else "No"}
  - Jurisdiction    : {jur}
  - Flagged steps:
{findings}
{"  - CUMULATIVE SESSION RISK: " + cum_reason if cum else ""}

Reply in 2-3 sentences. Begin with: "This query was blocked because..."
"""
        user_msg = f'Explain why this query was blocked: "{query}"'

    elif decision == "WARN":
        system = f"""You are a Responsible AI governed assistant with causal reasoning.

This query is SENSITIVE (WARN level). You must:
1. Open with one sentence noting why this topic is sensitive
2. Give a balanced, factual, unbiased response
3. Avoid reinforcing stereotypes about any protected group
4. Distinguish correlation from causation (Pearl causal reasoning)
5. Note ethical considerations if the query touches protected groups

RAI Findings:
  - Risk Score   : {risk}% (WARN zone)
  - Jurisdiction : {jur}
  - Flagged by:
{findings}
"""
        user_msg = query

    elif decision in ("EXPERT_REVIEW", "ESCALATE"):
        system = f"""You are the explanation layer of a Responsible AI governance framework.

This query falls in a GREY AREA requiring human expert review.

RAI Findings:
  - Risk Score : {risk}% (uncertainty zone)
  - Jurisdiction: {jur}
  - Uncertainty reason:
{findings}

Explain:
1. Why this query needs human review (1-2 sentences)
2. What type of expert should review it (legal / medical / ethics board)
3. What the user can do in the meantime

Begin with: "Your query has been flagged for human expert review because..."
"""
        user_msg = f'Explain why this query needs human expert review: "{query}"'

    else:  # ALLOW
        bias_context = any(
            any(kw in k.lower() for kw in
                ["scm", "bias", "causal", "fairness", "discriminat"])
            for k in flagged.keys()
        )
        if bias_context or risk > 20:
            system = f"""You are a Responsible AI governed assistant with causal fairness reasoning.

The RAI framework allowed this query but detected bias-sensitive context (risk: {risk}%).

Instructions:
1. Answer accurately and helpfully
2. Do not perpetuate bias about protected groups (race, gender, age, religion, disability)
3. Distinguish correlation from causation where relevant
4. If query involves AI decisions about people, note causal fairness principles
5. Jurisdiction: {jur}
"""
        else:
            system = (
                "You are a helpful, harmless, and honest AI assistant. "
                "You have been screened by a Responsible AI governance system. "
                "Answer clearly and accurately."
            )
        user_msg = query

    return system, user_msg


# ── Verification prompt builder ───────────────────────────────────────────────

def build_verify_prompt(query: str, draft: str, ctx: dict) -> str:
    """
    Build self-verification system prompt.
    LLM checks its own draft against SCM/Matrix findings + original query.
    Decision-specific checks for BLOCK / WARN / EXPERT_REVIEW / ALLOW.
    """
    decision = ctx["decision"]
    risk     = ctx["risk_pct"]
    vac      = ctx["vac_triggered"]
    jur      = ctx["jurisdiction"]
    attack   = ctx["attack_type"]
    flagged  = ctx["flagged_steps"]

    findings = "\n".join(
        f"  - {name}: {detail}" for name, detail in flagged.items()
    ) or "  - Pattern-based harm detection triggered"

    if decision == "BLOCK":
        return f"""You are a Responsible AI self-verification auditor.

A draft explanation was generated for a BLOCKED query. Your task:
Verify whether this explanation is ACCURATE and APPROPRIATE.

Original query  : "{query}"
Pipeline decision: BLOCK
Risk score      : {risk}%
Attack type     : {attack}
VAC triggered   : {"YES" if vac else "No"}
Jurisdiction    : {jur}
Flagged steps:
{findings}

Draft explanation to verify:
---
{draft}
---

Check ALL of the following:
1. CORRECT DECISION? Given risk={risk}% and findings above, was BLOCK the right call?
   - If risk is between 30-70% and VAC=No and attack=NONE → consider if WARN is more appropriate
   - If VAC=YES or risk>70% or attack detected → BLOCK is definitely correct
2. ACCURATE EXPLANATION? Does the draft correctly describe the actual reason for blocking?
   Does it match the flagged steps and findings above?
3. SAFE CONTENT? Does the explanation accidentally reveal or hint at harmful information?
4. RESPECTFUL TONE? Is it respectful to the user even while blocking?

Respond in this exact format:
VERDICT: [APPROVED / NEEDS_REVISION]
DECISION_CORRECT: [YES / SHOULD_BE_WARN / YES_DEFINITELY]
ISSUES: [list any problems found, or "None"]
REVISED_RESPONSE: [if NEEDS_REVISION, write the corrected explanation here; if APPROVED, write "N/A"]
"""

    elif decision == "WARN":
        return f"""You are a Responsible AI self-verification auditor.

A draft response was generated for a SENSITIVE (WARN level) query. Your task:
Verify whether this response is ACCURATE, UNBIASED, and APPROPRIATELY CAREFUL.

Original query  : "{query}"
Pipeline decision: WARN
Risk score      : {risk}%
Jurisdiction    : {jur}
Flagged steps:
{findings}

Draft response to verify:
---
{draft}
---

Check ALL of the following:
1. BIAS CHECK: Does the draft contain stereotypes, assumptions, or biased framing
   about any protected group (race, gender, age, religion, disability, nationality)?
2. CAUSAL ACCURACY: Does the draft incorrectly treat correlation as causation?
   If the topic involves AI decisions about people, is fairness properly noted?
3. SENSITIVITY: Does the draft appropriately acknowledge why this topic is sensitive?
4. JURISDICTION: Are any jurisdiction-specific ({jur}) legal considerations relevant
   and correctly addressed?
5. FACTUAL ACCURACY: Are the claims in the draft actually correct?

Respond in this exact format:
VERDICT: [APPROVED / NEEDS_REVISION]
ISSUES: [list any problems found, or "None"]
REVISED_RESPONSE: [if NEEDS_REVISION, write the improved response here; if APPROVED, write "N/A"]
"""

    elif decision in ("EXPERT_REVIEW", "ESCALATE"):
        return f"""You are a Responsible AI self-verification auditor.

A draft explanation was generated for a query requiring EXPERT REVIEW. Your task:
Verify whether this explanation is ACCURATE and HELPFUL.

Original query  : "{query}"
Pipeline decision: EXPERT_REVIEW
Risk score      : {risk}%
Jurisdiction    : {jur}
Flagged steps:
{findings}

Draft explanation to verify:
---
{draft}
---

Check ALL of the following:
1. UNCERTAINTY REASON: Does the draft accurately explain WHY the system is uncertain?
   Does it match the flagged steps above?
2. EXPERT TYPE: Is the suggested expert type correct for this domain?
   (e.g., legal expert for discrimination claims, medical expert for health decisions)
3. INTERIM GUIDANCE: Is the advice given to the user while they wait for expert review
   safe, practical, and appropriate?
4. TONE: Is it clear and helpful without being alarming?

Respond in this exact format:
VERDICT: [APPROVED / NEEDS_REVISION]
ISSUES: [list any problems found, or "None"]
REVISED_RESPONSE: [if NEEDS_REVISION, write the improved explanation here; if APPROVED, write "N/A"]
"""

    else:  # ALLOW
        return f"""You are a Responsible AI self-verification auditor.

A draft response was generated for an ALLOWED query. Your task:
Verify whether this response is ACCURATE, UNBIASED, and APPROPRIATE.

Original query  : "{query}"
Pipeline decision: ALLOW
Risk score      : {risk}% {"(bias-sensitive context flagged)" if risk > 20 else ""}
Jurisdiction    : {jur}
{("Flagged steps:\n" + findings) if flagged else "No steps flagged — clean query."}

Draft response to verify:
---
{draft}
---

Check ALL of the following:
1. FACTUAL ACCURACY: Are the claims in the draft correct?
2. BIAS CHECK: Does the draft contain any stereotypes, biased framing, or assumptions
   about protected groups (race, gender, age, religion, disability, nationality)?
   {"Note: SCM/Matrix flagged bias-sensitive context at " + str(risk) + "% risk — extra scrutiny needed." if risk > 20 else ""}
3. COMPLETENESS: Does the draft actually answer the user's question?
4. APPROPRIATENESS: Is the tone and depth right for the query?

Respond in this exact format:
VERDICT: [APPROVED / NEEDS_REVISION]
ISSUES: [list any problems found, or "None"]
REVISED_RESPONSE: [if NEEDS_REVISION, write the improved response here; if APPROVED, write "N/A"]
"""


# ── Core verify function ──────────────────────────────────────────────────────

def verify(
    query  : str,
    draft  : str,
    ctx    : dict,
    llm_fn : Callable[[str, str], str],
) -> VerificationResult:
    """
    Two-pass LLM self-verification using SCM/Matrix findings.

    Pass 1: LLM generates draft (done outside — passed as `draft`)
    Pass 2: LLM verifies draft against RAI context → revise if needed

    Args:
        query  : Original user query (for verification context)
        draft  : LLM's first-pass response to verify
        ctx    : RAI context dict from build_rai_context()
        llm_fn : Callable(system_prompt: str, user_msg: str) -> str
                 Your LLM wrapper — Groq, Gemini, OpenAI, local model, any.
                 Signature must accept (system: str, user: str) -> str.

    Returns:
        VerificationResult with final_response, was_revised, issues_found,
        decision_correct, status_tag.

    Year 2 note:
        When integrated as S14, pipeline calls:
            result   = pipeline.run(query)         # S00–S13
            llm_fn   = lambda sys, usr: call_llm(usr, system_prompt=sys)
            draft    = llm_fn(sys_prompt, query)
            verified = verify(query, draft, ctx, llm_fn)
    """
    verify_prompt = build_verify_prompt(query, draft, ctx)

    try:
        raw = llm_fn(verify_prompt, "Verify the draft response above.")
    except Exception as e:
        return VerificationResult(
            final_response   = draft,
            was_revised      = False,
            issues_found     = f"[Verification error: {e}]",
            decision_correct = "YES",
            status_tag       = "⚠ verify-error",
        )

    # ── Parse structured response ─────────────────────────────────────
    verdict          = "APPROVED"
    issues_found     = "None"
    revised_response = ""
    decision_correct = "YES"

    for line in raw.splitlines():
        ls = line.strip()
        if ls.startswith("VERDICT:"):
            verdict = ls.replace("VERDICT:", "").strip()
        elif ls.startswith("DECISION_CORRECT:"):
            decision_correct = ls.replace("DECISION_CORRECT:", "").strip()
        elif ls.startswith("ISSUES:"):
            issues_found = ls.replace("ISSUES:", "").strip()
        elif ls.startswith("REVISED_RESPONSE:"):
            revised_response = ls.replace("REVISED_RESPONSE:", "").strip()

    # Multi-line REVISED_RESPONSE handling
    if "REVISED_RESPONSE:" in raw:
        revised_response = raw.split("REVISED_RESPONSE:", 1)[1].strip()
        if revised_response.upper().startswith("N/A"):
            revised_response = ""

    needs_revision = "NEEDS_REVISION" in verdict.upper()

    if needs_revision and revised_response:
        return VerificationResult(
            final_response   = revised_response,
            was_revised      = True,
            issues_found     = issues_found,
            decision_correct = decision_correct,
            status_tag       = "🔄 self-verified+revised",
        )
    else:
        return VerificationResult(
            final_response   = draft,
            was_revised      = False,
            issues_found     = issues_found,
            decision_correct = decision_correct,
            status_tag       = "✓ self-verified",
        )


# ── Convenience: full two-pass flow ──────────────────────────────────────────

def two_pass(
    query  : str,
    result ,                          # PipelineResult
    llm_fn : Callable[[str, str], str],
    decision_name   : str,
    cum_override    : bool = False,
    cum_reason      : str  = "",
) -> Tuple[VerificationResult, dict]:
    """
    Complete two-pass flow: build context → draft → verify.

    Convenience wrapper for when you want the full flow in one call.

    Args:
        query         : Original user query
        result        : PipelineResult from pipeline.run()
        llm_fn        : Callable(system: str, user: str) -> str
        decision_name : "BLOCK" | "WARN" | "EXPERT_REVIEW" | "ALLOW"
        cum_override  : ContextEngine cumulative override flag
        cum_reason    : ContextEngine override reason string

    Returns:
        (VerificationResult, rai_context_dict)

    Example:
        result  = pipeline.run(query)
        def my_llm(system, user): return groq_client.chat(...).content
        vr, ctx = two_pass(query, result, my_llm, "BLOCK")
        print(vr.final_response, vr.status_tag)
    """
    ctx            = build_rai_context(result, decision_name, cum_override, cum_reason)
    sys_p, user_m  = build_system_prompt(query, ctx)
    draft          = llm_fn(sys_p, user_m)
    verified       = verify(query, draft, ctx, llm_fn)
    return verified, ctx
