# Governed AI Chatbot — Groq Edition (v15g + ContextEngine)
# RAI Framework v15g + Groq (Llama 3.3 70B) + Contextual Memory
# Modes: chat (interactive terminal) | batch (CSV → PDF + PNG)
#
# ContextEngine integration (indirect pipeline connection):
#   User → governed_chat() → ContextEngine check → Pipeline → LLM
#                                     ↑
#                          Same SQLite DB that pipeline will use in Year 2.
#   No changes to pipeline_v15.py required.

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

sys.path.insert(0, r'C:\responsible-ai-framework')

from groq import Groq
from pipeline_v15 import (
    ResponsibleAIPipeline, PipelineInput,
    FinalDecision, Jurisdiction
)
from context_engine import ContextEngine          # ← NEW

# ── Setup ──────────────────────────────────────────────────────────────
pipeline       = ResponsibleAIPipeline()
client         = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
context_engine = ContextEngine()                  # ← shared SQLite DB

# Clean up stale sessions on startup (older than 24 h)
_cleaned = context_engine.cleanup_old_sessions()
if _cleaned:
    print(f"  [ContextEngine] Cleaned {_cleaned} old session rows from DB.")

# Output files always save next to this script
OUTPUT_DIR = Path(__file__).parent

JURISDICTION_MAP = {
    "US":     Jurisdiction.US,
    "EU":     Jurisdiction.EU,
    "INDIA":  Jurisdiction.INDIA,
    "GLOBAL": Jurisdiction.GLOBAL,
}

# Cumulative risk threshold — avg session risk above this triggers block
CUMULATIVE_BLOCK_THRESHOLD = 0.60   # 60% → hard block
CUMULATIVE_WARN_THRESHOLD  = 0.45   # 45% → flag in response


# ── RAI Context Extractor ───────────────────────────────────────────────
def build_rai_context(result, decision_name: str,
                      cum_override: bool = False,
                      cum_reason: str = "") -> dict:
    """
    Extract key findings from PipelineResult for LLM context.
    Pulls SCM risk, attack type, VAC flag, and flagged step details.
    """
    flagged_steps = {}
    for step in getattr(result, "steps", []):
        if step.signal in ("BLOCK", "WARN", "ESCALATE"):
            flagged_steps[step.step_name] = step.detail

    return {
        "decision"          : decision_name,
        "risk_pct"          : round(result.scm_risk_pct, 1),
        "attack_type"       : result.attack_type.value,
        "emotion_flag"      : result.emotion_flag.value,
        "vac_triggered"     : result.vac_triggered,
        "jurisdiction"      : result.jurisdiction.value,
        "flagged_steps"     : flagged_steps,
        "cumulative_override": cum_override,
        "cumulative_reason" : cum_reason,
    }


def build_system_prompt(query: str, ctx: dict) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for LLM based on RAI decision.

    BLOCK       → LLM explains why — does NOT answer harmful query
    WARN        → LLM gives careful response with sensitivity note
    EXPERT_REVIEW → LLM explains why human review needed
    ALLOW       → Normal response; bias-aware if SCM flagged risk
    """
    decision   = ctx["decision"]
    risk       = ctx["risk_pct"]
    attack     = ctx["attack_type"]
    vac        = ctx["vac_triggered"]
    jur        = ctx["jurisdiction"]
    flagged    = ctx["flagged_steps"]
    cum        = ctx["cumulative_override"]
    cum_reason = ctx["cumulative_reason"]

    findings = "\n".join(
        f"  - {name}: {detail}" for name, detail in flagged.items()
    ) or "  - Pattern-based harm detection triggered"

    # ── BLOCK ─────────────────────────────────────────────────────────
    if decision == "BLOCK":
        system = f"""You are the explanation layer of a Responsible AI governance framework.

A user query was BLOCKED by the RAI pipeline. Your job:
1. Explain WHY it was blocked using the findings below — be clear and respectful
2. Do NOT answer or hint at the harmful content
3. Briefly suggest what the user CAN ask instead (if safe alternative exists)

RAI Findings:
  - Risk Score      : {risk}%
  - Attack Detected : {attack}
  - VAC Triggered   : {"YES — absolute ethical violation" if vac else "No"}
  - Jurisdiction    : {jur}
  - Flagged steps:
{findings}
{"  - CUMULATIVE SESSION RISK: " + cum_reason if cum else ""}

Reply in 2–3 sentences. Begin with: "This query was blocked because..."
"""
        user_msg = f"Explain why this query was blocked: \"{query}\""

    # ── WARN ──────────────────────────────────────────────────────────
    elif decision == "WARN":
        system = f"""You are a Responsible AI governed assistant with causal reasoning.

This query is SENSITIVE (WARN level). You must:
1. Open with one sentence noting why this topic is sensitive
2. Give a balanced, factual, unbiased response
3. Avoid reinforcing stereotypes or bias toward any protected group
4. Where relevant, distinguish correlation from causation (Pearl causal reasoning)
5. Note ethical considerations if the query touches protected groups or high-stakes decisions

RAI Findings:
  - Risk Score   : {risk}% (WARN zone — between {risk-5:.0f}% and 70%)
  - Jurisdiction : {jur}
  - Flagged by:
{findings}
"""
        user_msg = query

    # ── EXPERT_REVIEW / ESCALATE ──────────────────────────────────────
    elif decision in ("EXPERT_REVIEW", "ESCALATE"):
        system = f"""You are the explanation layer of a Responsible AI governance framework.

This query falls in a GREY AREA — the automated system cannot confidently
decide allow or block. It requires human expert review.

RAI Findings:
  - Risk Score      : {risk}% (uncertainty zone — between 20% and 70%)
  - Jurisdiction    : {jur}
  - Uncertainty reason:
{findings}

Explain to the user:
1. Why this specific query needs human review (1–2 sentences)
2. What type of expert should review it (e.g. legal, medical, ethics board)
3. What they can do in the meantime

Begin with: "Your query has been flagged for human expert review because..."
"""
        user_msg = f"Explain why this query needs human expert review: \"{query}\""

    # ── ALLOW ─────────────────────────────────────────────────────────
    else:
        bias_context = any(
            any(kw in k.lower() for kw in ["scm", "bias", "causal", "fairness", "discriminat"])
            for k in flagged.keys()
        )

        if bias_context or risk > 20:
            system = f"""You are a Responsible AI governed assistant with causal fairness reasoning.

The RAI framework allowed this query but detected bias-sensitive context (risk: {risk}%).

Instructions:
1. Answer accurately and helpfully
2. Do not perpetuate bias about protected groups (race, gender, age, religion, disability)
3. Distinguish between correlation and causation where relevant
4. If the query involves AI decisions about people, note the importance of causal fairness
5. Jurisdiction: {jur}

Provide a clear, unbiased, factual response with appropriate reasoning.
"""
        else:
            system = (
                "You are a helpful, harmless, and honest AI assistant. "
                "You have been screened by a Responsible AI governance system. "
                "Answer clearly and accurately."
            )
        user_msg = query

    return system, user_msg


# ── LLM Call ───────────────────────────────────────────────────────────
def call_llm(user_msg: str,
             model:       str = "llama-3.3-70b-versatile",
             system_prompt: str = "") -> str:
    """Send query to Llama 3 via Groq with a custom system prompt."""
    try:
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": system_prompt or
                    "You are a helpful, harmless, and honest AI assistant."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens = 1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error: {e}]"


# ── Governed Chat ───────────────────────────────────────────────────────
def governed_chat(
    query:        str,
    session_id:   str,                              # ← NEW: required
    jurisdiction: str = "GLOBAL",
    model:        str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Pipeline + ContextEngine as gatekeeper.

    Flow:
        1. pipeline.run()  → get decision + SCM risk score
        2. ContextEngine   → check cumulative session risk
        3. Override to BLOCK if cumulative risk exceeds threshold
        4. ContextEngine   → store this turn
        5. LLM call        → only if final decision is ALLOW / WARN

    The ContextEngine sits OUTSIDE pipeline (no pipeline changes).
    Same SQLite DB will be reused when ContextEngine moves inside
    pipeline in Year 2.
    """
    jur    = JURISDICTION_MAP.get(jurisdiction.upper(), Jurisdiction.GLOBAL)
    result = pipeline.run(PipelineInput(query=query, jurisdiction=jur))
    dec    = result.final_decision

    # ── Normalise risk score ─────────────────────────────────────────
    risk_norm = round(result.scm_risk_pct / 100.0, 4)   # 0.0 – 1.0

    base = {
        "decision"   : dec.name,
        "risk"       : round(result.scm_risk_pct, 1),
        "latency_ms" : round(result.total_ms, 1),
        "model"      : model,
    }

    # ── Cumulative Risk Check ─────────────────────────────────────────
    # Uses REAL SCM score (not keyword heuristic) — more accurate.
    cum_risky, cum_score, cum_reason = context_engine.detect_cumulative_risk(
        session_id, risk_norm
    )

    cumulative_override = False
    if cum_risky and dec != FinalDecision.BLOCK:
        if cum_score >= CUMULATIVE_BLOCK_THRESHOLD:
            # Override decision to BLOCK
            dec               = FinalDecision.BLOCK
            base["decision"]  = "BLOCK"
            cumulative_override = True
            base["cum_reason"]  = f"Cumulative block — {cum_reason}"
        elif cum_score >= CUMULATIVE_WARN_THRESHOLD:
            base["cum_reason"] = f"Cumulative warning — {cum_reason}"

    # ── Store this turn in ContextEngine ─────────────────────────────
    # Stored BEFORE LLM call so BLOCK turns are also recorded.
    context_engine.add_turn(
        session_id   = session_id,
        prompt       = query,
        risk_score   = risk_norm,
        decision     = base["decision"],
        source       = "groq",
        jurisdiction = jurisdiction.upper(),
        llm_model    = model,
        attack_type  = getattr(result, "attack_type", None) and result.attack_type.value,
        latency_ms   = result.total_ms,
        # Year 2: fill these from pipeline internals
        # scm_risk_raw  = ...,
        # matrix_score  = ...,
        # legal_score   = ...,
    )

    # ── Build Response ────────────────────────────────────────────────
    # Extract full pipeline analysis for LLM context
    rai_ctx = build_rai_context(
        result,
        decision_name = base["decision"],
        cum_override  = cumulative_override,
        cum_reason    = base.get("cum_reason", ""),
    )

    # Build decision-specific system prompt + user message
    system_prompt, user_msg = build_system_prompt(query, rai_ctx)

    # ── BLOCK: LLM explains why — never answers harmful query ─────────
    if base["decision"] == "BLOCK":
        explanation = call_llm(user_msg, model=model, system_prompt=system_prompt)
        return {**base, "response": f"🚫 BLOCKED\n\n{explanation}"}

    # ── EXPERT_REVIEW: LLM explains uncertainty ───────────────────────
    if base["decision"] == "EXPERT_REVIEW":
        explanation = call_llm(user_msg, model=model, system_prompt=system_prompt)
        return {**base, "response": f"👤 EXPERT REVIEW NEEDED\n\n{explanation}"}

    # ── ALLOW / WARN: LLM responds with RAI-aware system prompt ──────
    llm_response = call_llm(user_msg, model=model, system_prompt=system_prompt)

    if dec == FinalDecision.WARN:
        header = f"⚠️  Risk {base['risk']:.1f}% — Sensitive topic (governed response)\n\n"
        return {**base, "response": header + llm_response}

    return {**base, "response": llm_response}


# ── Batch Run ───────────────────────────────────────────────────────────
def run_batch(
    csv_path:     str,
    jurisdiction: str = "GLOBAL",
    model:        str = "llama-3.3-70b-versatile",
) -> List[dict]:
    """Run all prompts in CSV through governed pipeline."""
    # Each batch gets its own session
    session_id = context_engine.new_session(source="groq_batch")

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            res = governed_chat(
                query        = row["prompt"],
                session_id   = session_id,
                jurisdiction = row.get("jurisdiction", jurisdiction),
                model        = model,
            )
            res["case_id"]  = row.get("case_id", "")
            res["prompt"]   = row["prompt"]
            res["expected"] = row.get("expected", "").strip().upper()

            got = res["decision"].upper()
            exp = res["expected"]
            if exp:
                correct = (
                    got == exp or
                    (exp == "BLOCK" and got in ("BLOCK", "EXPERT_REVIEW"))
                )
                res["correct"] = "✅" if correct else "❌"
            else:
                res["correct"] = ""

            rows.append(res)
            print(
                f"  {res.get('correct','  ')} [{res['decision']:<14}] "
                f"risk={res['risk']:>5.1f}%  {row['prompt'][:55]}"
            )

    # ── Auto-save context memory for this batch session ───────────────
    mem_csv = str(OUTPUT_DIR / f"context_memory_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    n = context_engine.export_session_csv(session_id, mem_csv)
    print(f"\n  🧠 Context memory saved ({n} turns) → {mem_csv}")

    return rows


# ── Risk Histogram ──────────────────────────────────────────────────────
def plot_risk_histogram(rows: List[dict], out: str = "") -> None:
    out     = out or str(OUTPUT_DIR / "risk_hist.png")
    blocked = [r["risk"] for r in rows if r["decision"] == "BLOCK"]
    warned  = [r["risk"] for r in rows if r["decision"] == "WARN"]
    allowed = [r["risk"] for r in rows if r["decision"] == "ALLOW"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, 105, 5)
    if allowed:  ax.hist(allowed, bins=bins, alpha=0.7, label="ALLOW", color="#2ecc71")
    if warned:   ax.hist(warned,  bins=bins, alpha=0.7, label="WARN",  color="#f39c12")
    if blocked:  ax.hist(blocked, bins=bins, alpha=0.7, label="BLOCK", color="#e74c3c")

    ax.set_title("SCM Risk Score Distribution — Groq Governed Chatbot (v15g)", fontsize=13)
    ax.set_xlabel("Risk Score (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.clf()
    print(f"\n  📊 Risk histogram saved → {out}")


# ── PDF Report ───────────────────────────────────────────────────────────
def generate_pdf(rows: List[dict], out: str = "") -> None:
    out     = out or str(OUTPUT_DIR / "report.pdf")
    doc     = SimpleDocTemplate(out)
    styles  = getSampleStyleSheet()
    content = []

    content.append(Paragraph(
        "RAI Governed Chatbot — Evaluation Report (Groq)", styles["Title"]
    ))
    content.append(Paragraph(
        f"Framework: Responsible AI Framework v5.0 (pipeline_v15g)  |  "
        f"Model: Llama 3.3 70B (Groq)  |  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    total   = len(rows)
    blocked = sum(1 for r in rows if r["decision"] == "BLOCK")
    warned  = sum(1 for r in rows if r["decision"] == "WARN")
    allowed = sum(1 for r in rows if r["decision"] == "ALLOW")
    labeled = [r for r in rows if r.get("expected")]
    correct = sum(1 for r in labeled if r.get("correct") == "✅")

    content.append(Paragraph("Summary", styles["Heading2"]))
    summary_data = [
        ["Metric",        "Value"],
        ["Total cases",   str(total)],
        ["BLOCK",         str(blocked)],
        ["WARN",          str(warned)],
        ["ALLOW",         str(allowed)],
        ["Labeled cases", str(len(labeled))],
        ["Correct",       f"{correct}/{len(labeled)} ({correct/len(labeled)*100:.0f}%)"
                          if labeled else "N/A"],
    ]
    t = Table(summary_data, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING",       (0, 0), (-1, -1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 16))

    content.append(Paragraph("Per-Case Results", styles["Heading2"]))
    case_data = [["#", "Prompt", "Expected", "Got", "Risk%", "✓"]]
    for r in rows:
        case_data.append([
            r.get("case_id", ""),
            r["prompt"][:50] + ("…" if len(r["prompt"]) > 50 else ""),
            r.get("expected", "—"),
            r["decision"],
            f"{r['risk']:.1f}",
            r.get("correct", ""),
        ])
    ct = Table(case_data, colWidths=[25, 200, 65, 75, 45, 25])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING",       (0, 0), (-1, -1), 5),
    ]))
    content.append(ct)
    content.append(Spacer(1, 16))

    failures = [r for r in rows if r.get("correct") == "❌"]
    if failures:
        content.append(Paragraph("Failure Analysis", styles["Heading2"]))
        for r in failures:
            content.append(Paragraph(
                f"<b>Prompt:</b> {r['prompt'][:80]}<br/>"
                f"<b>Expected:</b> {r['expected']}  "
                f"<b>Got:</b> {r['decision']}  "
                f"<b>Risk:</b> {r['risk']:.1f}%",
                styles["Normal"]
            ))
            content.append(Spacer(1, 8))
    else:
        content.append(Paragraph(
            "✅ No failures — all labeled cases correct.", styles["Normal"]
        ))

    doc.build(content)
    print(f"  📄 PDF report saved → {out}")


# ── Failure Analysis (console) ───────────────────────────────────────────
def failure_analysis(rows: List[dict]) -> None:
    failures = [r for r in rows if r.get("correct") == "❌"]
    print(f"\n{'='*55}")
    print(f"  FAILURE ANALYSIS — {len(failures)} failures / {len(rows)} cases")
    print(f"{'='*55}")
    if not failures:
        print("  ✅ Zero failures — all labeled cases correct!")
        return
    for r in failures:
        print(f"  Prompt  : {r['prompt'][:70]}")
        print(f"  Expected: {r['expected']}  |  Got: {r['decision']}  |  Risk: {r['risk']:.1f}%")


# ── Interactive Chat ─────────────────────────────────────────────────────
def run_interactive(
    jurisdiction: str = "GLOBAL",
    model:        str = "llama-3.3-70b-versatile",
) -> None:
    """
    Live terminal chat — type prompts, get governed responses.
    Session memory auto-saved to CSV on exit.
    """
    # One session per interactive run — persists across the full conversation
    session_id = context_engine.new_session(source="groq")

    print("\n" + "=" * 60)
    print("  RESPONSIBLE AI GOVERNED CHATBOT — GROQ INTERACTIVE")
    print(f"  Framework v15g + ContextEngine  |  Model: {model}")
    print(f"  Jurisdiction: {jurisdiction}  |  Session: {session_id}")
    print("  Type 'quit' to exit | session report saved on exit")
    print("=" * 60 + "\n")

    session_rows = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = governed_chat(
            query        = query,
            session_id   = session_id,
            jurisdiction = jurisdiction,
            model        = model,
        )
        dec = result["decision"]

        print(f"\n[{dec}] risk={result['risk']:.1f}% | {result['latency_ms']:.0f}ms")
        print(result["response"])
        print()

        session_rows.append({**result, "prompt": query, "expected": ""})

    # ── End of session: auto-save ─────────────────────────────────────
    if session_rows:
        # Always auto-export context memory CSV
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        mem_csv = str(OUTPUT_DIR / f"context_memory_{ts}.csv")
        n       = context_engine.export_session_csv(session_id, mem_csv)
        print(f"\n  🧠 Context memory auto-saved ({n} turns) → {mem_csv}")

        # Session summary
        summary = context_engine.session_summary(session_id)
        print(f"  📊 Session summary: {summary}")

        # Optional: full report
        save = input("\nSave full session report (PDF + histogram)? (y/n): ").strip().lower()
        if save == "y":
            plot_risk_histogram(session_rows, out=str(OUTPUT_DIR / f"session_risk_hist_{ts}.png"))
            generate_pdf(session_rows, out=str(OUTPUT_DIR / f"session_report_{ts}.pdf"))
            print(f"  Reports saved in: {OUTPUT_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAI Governed Chatbot — Groq Edition (v15g + ContextEngine)"
    )
    parser.add_argument(
        "--mode", choices=["chat", "batch"],
        default="chat",
        help="chat = interactive terminal | batch = CSV run + PDF + PNG"
    )
    parser.add_argument("--csv",          default="groq_test_cases.csv")
    parser.add_argument("--jurisdiction", default="GLOBAL")
    parser.add_argument("--model",        default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RAI GOVERNED CHATBOT — GROQ EDITION (v15g + ContextEngine)")
    print("=" * 60)

    if args.mode == "chat":
        run_interactive(jurisdiction=args.jurisdiction, model=args.model)

    elif args.mode == "batch":
        print(f"\n  Batch mode → {args.csv}\n")
        rows = run_batch(args.csv, jurisdiction=args.jurisdiction, model=args.model)
        plot_risk_histogram(rows)
        generate_pdf(rows)
        failure_analysis(rows)
        print(f"\n  Done. Outputs saved in: {OUTPUT_DIR}")
