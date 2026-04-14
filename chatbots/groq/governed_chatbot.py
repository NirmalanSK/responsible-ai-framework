# Governed AI Chatbot — Groq Edition (v15g)
# RAI Framework v15g + Groq (Llama 3.3 70B)
# Modes: chat (interactive terminal) | batch (CSV → PDF + PNG)

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

# ── Setup ──────────────────────────────────────────────────────────────
pipeline = ResponsibleAIPipeline()
client   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Output files always save next to this script (chatbots/groq/)
OUTPUT_DIR = Path(__file__).parent

JURISDICTION_MAP = {
    "US":     Jurisdiction.US,
    "EU":     Jurisdiction.EU,
    "INDIA":  Jurisdiction.INDIA,
    "GLOBAL": Jurisdiction.GLOBAL,
}

# ── LLM Call ───────────────────────────────────────────────────────────
def call_llm(query: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Send approved query to Llama 3 via Groq."""
    try:
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {
                    "role"   : "system",
                    "content": (
                        "You are a helpful, harmless, and honest AI assistant. "
                        "You have already been screened by a responsible AI "
                        "governance system. Answer clearly and accurately."
                    )
                },
                {"role": "user", "content": query}
            ],
            max_tokens = 1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error: {e}]"


# ── Governed Chat ───────────────────────────────────────────────────────
def governed_chat(
    query: str,
    jurisdiction: str = "GLOBAL",
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Pipeline as gatekeeper — only ALLOW/WARN reach the LLM.
    BLOCK → no LLM call at all.
    """
    jur    = JURISDICTION_MAP.get(jurisdiction.upper(), Jurisdiction.GLOBAL)
    result = pipeline.run(PipelineInput(query=query, jurisdiction=jur))
    dec    = result.final_decision

    base = {
        "decision"   : dec.name,
        "risk"       : round(result.scm_risk_pct, 1),
        "latency_ms" : round(result.total_ms, 1),
        "model"      : model,
    }

    if dec == FinalDecision.BLOCK:
        return {**base, "response": "🚫 BLOCKED by RAI governance"}

    if dec == FinalDecision.EXPERT_REVIEW:
        return {**base, "response": "👤 Escalated for human review"}

    llm_response = call_llm(query, model=model)
    notice = "\n⚠️  Sensitive topic — response monitored." if dec == FinalDecision.WARN else ""
    return {**base, "response": llm_response + notice}


# ── Batch Run ───────────────────────────────────────────────────────────
def run_batch(
    csv_path: str,
    jurisdiction: str = "GLOBAL",
    model: str = "llama-3.3-70b-versatile",
) -> List[dict]:
    """Run all prompts in CSV through governed pipeline."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            res = governed_chat(
                query        = row["prompt"],
                jurisdiction = row.get("jurisdiction", jurisdiction),
                model        = model,
            )
            res["case_id"]  = row.get("case_id", "")
            res["prompt"]   = row["prompt"]
            res["expected"] = row.get("expected", "").strip().upper()

            # Correctness check
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
    return rows


# ── Risk Histogram ──────────────────────────────────────────────────────
def plot_risk_histogram(rows: List[dict], out: str = "") -> None:
    """Plot SCM risk score distribution — ALLOW/WARN/BLOCK color split."""
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
    """Generate per-case evaluation report as PDF."""
    out     = out or str(OUTPUT_DIR / "report.pdf")
    doc     = SimpleDocTemplate(out)
    styles  = getSampleStyleSheet()
    content = []

    # Title
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

    # Summary stats
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
        ["Correct",       f"{correct}/{len(labeled)} ({correct/len(labeled)*100:.0f}%)" if labeled else "N/A"],
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

    # Per-case table
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

    # Failure analysis
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
    model: str = "llama-3.3-70b-versatile",
) -> None:
    """Live terminal chat — type prompts, get governed responses."""
    print("\n" + "=" * 60)
    print("  RESPONSIBLE AI GOVERNED CHATBOT — GROQ INTERACTIVE")
    print(f"  Framework v15g  |  Model: {model}")
    print(f"  Jurisdiction: {jurisdiction}")
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

        result = governed_chat(query, jurisdiction=jurisdiction, model=model)
        dec    = result["decision"]

        print(f"\n[{dec}] risk={result['risk']:.1f}% | {result['latency_ms']:.0f}ms")
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

    # End of session — auto save report
    if session_rows:
        save = input("\nSave session report? (y/n): ").strip().lower()
        if save == "y":
            plot_risk_histogram(session_rows, out=str(OUTPUT_DIR / "session_risk_hist.png"))
            generate_pdf(session_rows, out=str(OUTPUT_DIR / "session_report.pdf"))
            print(f"  Reports saved in: {OUTPUT_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAI Governed Chatbot — Groq Edition (v15g)"
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
    print("  RAI GOVERNED CHATBOT — GROQ EDITION (v15g)")
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
