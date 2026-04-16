# Gemini Governed Chatbot v5 — Research Edition (+ ContextEngine)
# RAI Framework v15g + Gemini API + Contextual Memory
# Features: Batch testing, Risk histogram, PDF report,
#           Multi-model benchmark, Failure analysis,
#           Multi-turn attack detection via ContextEngine
#
# ContextEngine integration (indirect pipeline connection):
#   Same SQLite DB (rai_context.db) used by Groq chatbot.
#   source = "gemini" so turns from both chatbots are distinguishable.
#   No changes to pipeline_v15.py required.

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

sys.path.insert(0, r"C:\responsible-ai-framework")

OUTPUT_DIR = Path(__file__).parent

from google import genai

from pipeline_v15 import (
    ResponsibleAIPipeline,
    PipelineInput,
    FinalDecision,
    Jurisdiction,
)
from context_engine import ContextEngine          # ← NEW

# ── Setup ──────────────────────────────────────────────────────────────
pipeline       = ResponsibleAIPipeline()
client         = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
context_engine = ContextEngine()                  # ← shared SQLite DB

# Clean up stale sessions on startup
_cleaned = context_engine.cleanup_old_sessions()
if _cleaned:
    print(f"  [ContextEngine] Cleaned {_cleaned} old session rows from DB.")

JURISDICTION_MAP = {
    "US":     Jurisdiction.US,
    "EU":     Jurisdiction.EU,
    "INDIA":  Jurisdiction.INDIA,
    "GLOBAL": Jurisdiction.GLOBAL,
}

CUMULATIVE_BLOCK_THRESHOLD = 0.60
CUMULATIVE_WARN_THRESHOLD  = 0.45


# ── LLM Call ───────────────────────────────────────────────────────────
def call_llm(query: str, model: str = "gemini-2.0-flash") -> str:
    """Send query to Gemini and return response text."""
    try:
        response = client.models.generate_content(model=model, contents=[query])
        return response.text or ""
    except Exception as e:
        return f"[LLM Error: {e}]"


# ── Governed Chat ───────────────────────────────────────────────────────
def governed_chat(
    query:        str,
    session_id:   str,                              # ← NEW: required
    jurisdiction: str = "GLOBAL",
    model:        str = "gemini-2.0-flash",
) -> dict:
    """
    Pipeline + ContextEngine as gatekeeper.

    Flow:
        1. pipeline.run()  → decision + SCM risk score
        2. ContextEngine   → cumulative session risk check
        3. Override BLOCK  if cumulative risk ≥ 0.60
        4. ContextEngine   → store turn (always, including BLOCKs)
        5. LLM call        → only if ALLOW / WARN
    """
    jur    = JURISDICTION_MAP.get(jurisdiction.upper(), Jurisdiction.GLOBAL)
    result = pipeline.run(PipelineInput(query=query, jurisdiction=jur))
    dec    = result.final_decision

    risk_norm = round(result.scm_risk_pct / 100.0, 4)

    base = {
        "decision"   : dec.name,
        "risk"       : round(result.scm_risk_pct, 1),
        "latency_ms" : round(result.total_ms, 1),
        "model"      : model,
    }

    # ── Cumulative Risk Check ─────────────────────────────────────────
    cum_risky, cum_score, cum_reason = context_engine.detect_cumulative_risk(
        session_id, risk_norm
    )

    cumulative_override = False
    if cum_risky and dec != FinalDecision.BLOCK:
        if cum_score >= CUMULATIVE_BLOCK_THRESHOLD:
            dec               = FinalDecision.BLOCK
            base["decision"]  = "BLOCK"
            cumulative_override = True
            base["cum_reason"]  = f"Cumulative block — {cum_reason}"
        elif cum_score >= CUMULATIVE_WARN_THRESHOLD:
            base["cum_reason"] = f"Cumulative warning — {cum_reason}"

    # ── Store turn ────────────────────────────────────────────────────
    context_engine.add_turn(
        session_id   = session_id,
        prompt       = query,
        risk_score   = risk_norm,
        decision     = base["decision"],
        source       = "gemini",
        jurisdiction = jurisdiction.upper(),
        llm_model    = model,
        attack_type  = getattr(result, "attack_type", None) and result.attack_type.value,
        latency_ms   = result.total_ms,
    )

    # ── Build Response ────────────────────────────────────────────────
    if base["decision"] == "BLOCK":
        reason_msg = f" ({cum_reason})" if cumulative_override else ""
        return {**base, "response": f"🚫 BLOCKED by RAI governance{reason_msg}"}

    if base["decision"] == "EXPERT_REVIEW":
        return {**base, "response": "👤 Escalated for human review"}

    llm_response = call_llm(query, model=model)
    notice = "\n⚠️  Sensitive topic — response monitored." if dec == FinalDecision.WARN else ""
    if "cum_reason" in base and dec != FinalDecision.BLOCK:
        notice += f"\n⚠️  {base['cum_reason']}"

    return {**base, "response": llm_response + notice}


# ── Batch Run ───────────────────────────────────────────────────────────
def run_batch(
    csv_path:     str,
    jurisdiction: str = "GLOBAL",
    model:        str = "gemini-2.0-flash",
) -> List[dict]:
    """Run all prompts in CSV through governed pipeline."""
    session_id = context_engine.new_session(source="gemini_batch")

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

    # Auto-save context memory CSV
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem_csv = str(OUTPUT_DIR / f"context_memory_batch_{ts}.csv")
    n       = context_engine.export_session_csv(session_id, mem_csv)
    print(f"\n  🧠 Context memory saved ({n} turns) → {mem_csv}")

    return rows


# ── Risk Histogram ──────────────────────────────────────────────────────
def plot_risk_histogram(rows: List[dict], out: str = "") -> None:
    out      = out or str(OUTPUT_DIR / "risk_hist.png")
    blocked  = [r["risk"] for r in rows if r["decision"] == "BLOCK"]
    allowed  = [r["risk"] for r in rows if r["decision"] == "ALLOW"]
    warned   = [r["risk"] for r in rows if r["decision"] == "WARN"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, 105, 5)
    if allowed:  ax.hist(allowed,  bins=bins, alpha=0.7, label="ALLOW",  color="#2ecc71")
    if warned:   ax.hist(warned,   bins=bins, alpha=0.7, label="WARN",   color="#f39c12")
    if blocked:  ax.hist(blocked,  bins=bins, alpha=0.7, label="BLOCK",  color="#e74c3c")

    ax.set_title("SCM Risk Score Distribution — Governed Chatbot (v15g)", fontsize=13)
    ax.set_xlabel("Risk Score (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.clf()
    print(f"\n  📊 Risk histogram saved → {out}")


# ── PDF Report ────────────────────────────────────────────────────────────
def generate_pdf(rows: List[dict], out: str = "") -> None:
    out     = out or str(OUTPUT_DIR / "report.pdf")
    doc     = SimpleDocTemplate(out)
    styles  = getSampleStyleSheet()
    content = []

    content.append(Paragraph("RAI Governed Chatbot — Evaluation Report", styles["Title"]))
    content.append(Paragraph(
        f"Framework: Responsible AI Framework v5.0 (pipeline_v15g)  |  "
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
        ["Metric",          "Value"],
        ["Total cases",     str(total)],
        ["BLOCK",           str(blocked)],
        ["WARN",            str(warned)],
        ["ALLOW",           str(allowed)],
        ["Labeled cases",   str(len(labeled))],
        ["Correct",         f"{correct}/{len(labeled)} ({correct/len(labeled)*100:.0f}%)"
                            if labeled else "N/A"],
    ]
    t = Table(summary_data, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(ct)
    content.append(Spacer(1, 16))

    failures = [r for r in rows if r.get("correct") == "❌"]
    if failures:
        content.append(Paragraph("Failure Analysis", styles["Heading2"]))
        for r in failures:
            content.append(Paragraph(
                f"<b>Prompt:</b> {r['prompt'][:80]}<br/>"
                f"<b>Expected:</b> {r['expected']}  <b>Got:</b> {r['decision']}  "
                f"<b>Risk:</b> {r['risk']:.1f}%",
                styles["Normal"]
            ))
            content.append(Spacer(1, 8))
    else:
        content.append(Paragraph("✅ No failures — all labeled cases correct.", styles["Normal"]))

    doc.build(content)
    print(f"  📄 PDF report saved → {out}")


# ── Multi-model Benchmark ─────────────────────────────────────────────────
def benchmark_models(
    csv_path:     str,
    models:       List[str],
    jurisdiction: str = "GLOBAL",
) -> None:
    """Compare governance accuracy across multiple Gemini models."""
    print("\n" + "=" * 55)
    print("  MULTI-MODEL BENCHMARK")
    print("=" * 55)

    with open(csv_path, encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    labeled = [r for r in data if r.get("expected", "").strip()]
    if not labeled:
        print("  No labeled rows in CSV — skipping benchmark.")
        return

    results = {}
    for model in models:
        # Each model gets its own session in benchmark mode
        session_id = context_engine.new_session(source=f"gemini_bench")
        correct    = 0
        print(f"\n  Model: {model}")
        for row in labeled:
            res = governed_chat(
                query        = row["prompt"],
                session_id   = session_id,
                jurisdiction = row.get("jurisdiction", jurisdiction),
                model        = model,
            )
            exp = row["expected"].strip().upper()
            got = res["decision"].upper()
            ok  = (got == exp or (exp == "BLOCK" and got in ("BLOCK", "EXPERT_REVIEW")))
            if ok:
                correct += 1
            mark = "✅" if ok else "❌"
            print(f"    {mark} [{got:<14}] {row['prompt'][:50]}")

        acc = correct / len(labeled)
        results[model] = acc
        print(f"  → Accuracy: {correct}/{len(labeled)} ({acc*100:.0f}%)")

    print("\n" + "=" * 55)
    print("  BENCHMARK SUMMARY")
    print("=" * 55)
    for model, acc in results.items():
        bar = "█" * int(acc * 20)
        print(f"  {model:<30} {bar} {acc*100:.0f}%")
    print("=" * 55)


# ── Failure Analysis ──────────────────────────────────────────────────────
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
        print()


# ── Interactive Chat Mode ─────────────────────────────────────────────────
def run_interactive(
    jurisdiction: str = "GLOBAL",
    model:        str = "gemini-2.0-flash",
) -> None:
    """
    Live terminal chat with Gemini.
    Session memory auto-saved to CSV on exit.
    """
    session_id = context_engine.new_session(source="gemini")

    print("\n" + "=" * 60)
    print("  RESPONSIBLE AI GOVERNED CHATBOT — GEMINI INTERACTIVE")
    print(f"  Framework v15g + ContextEngine  |  Model: {model}")
    print(f"  Jurisdiction: {jurisdiction}  |  Session: {session_id}")
    print("  Type 'quit' to exit | 'switch <model>' to change model")
    print("=" * 60 + "\n")

    current_model = model
    session_rows  = []

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

        if query.lower().startswith("switch "):
            current_model = query.split(" ", 1)[1].strip()
            print(f"  ↩  Model switched → {current_model}\n")
            continue

        result = governed_chat(
            query        = query,
            session_id   = session_id,
            jurisdiction = jurisdiction,
            model        = current_model,
        )
        dec = result["decision"]

        print(f"\n[{dec}] risk={result['risk']:.1f}% | {result['latency_ms']:.0f}ms | model={current_model}")

        if dec == "BLOCK":
            print(f"🚫 {result['response']}")
        elif dec == "EXPERT_REVIEW":
            print(f"👤 {result['response']}")
        elif dec == "WARN":
            print(f"⚠️  Sensitive topic — monitored\nAI: {result['response']}")
        else:
            print(f"AI: {result['response']}")

        print()
        session_rows.append({**result, "prompt": query, "expected": ""})

    # ── Auto-save context memory ──────────────────────────────────────
    if session_rows:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        mem_csv = str(OUTPUT_DIR / f"context_memory_{ts}.csv")
        n       = context_engine.export_session_csv(session_id, mem_csv)
        print(f"\n  🧠 Context memory auto-saved ({n} turns) → {mem_csv}")

        summary = context_engine.session_summary(session_id)
        print(f"  📊 Session summary: {summary}")

        save = input("\nSave full session report (PDF + histogram)? (y/n): ").strip().lower()
        if save == "y":
            plot_risk_histogram(session_rows, out=str(OUTPUT_DIR / f"session_risk_hist_{ts}.png"))
            generate_pdf(session_rows, out=str(OUTPUT_DIR / f"session_report_{ts}.pdf"))
            print("  Reports saved in chatbots/gemini/ folder")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAI Governed Chatbot — Gemini Edition (v15g + ContextEngine)"
    )
    parser.add_argument(
        "--mode", choices=["chat", "batch", "benchmark"],
        default="chat",
        help="chat = interactive terminal | batch = CSV run + report | benchmark = multi-model"
    )
    parser.add_argument("--csv",          default="gemini_test_cases.csv")
    parser.add_argument("--jurisdiction", default="GLOBAL")
    parser.add_argument("--model",        default="gemini-2.0-flash")
    args = parser.parse_args()

    MODELS = [args.model]

    print("\n" + "=" * 60)
    print("  RAI GOVERNED CHATBOT — GEMINI EDITION (v15g + ContextEngine)")
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

    elif args.mode == "benchmark":
        benchmark_models(args.csv, MODELS, jurisdiction=args.jurisdiction)
