"""
Governed AI Chatbot
Nirmalan RAI Framework v5.0 + Groq (Free LLM)
"""
import os
import sys
sys.path.insert(0, r'C:\responsible-ai-framework')

from groq import Groq
from pipeline_v15 import (
    ResponsibleAIPipeline, PipelineInput,
    FinalDecision, Jurisdiction
)

# ── Setup ─────────────────────────────────────
pipeline = ResponsibleAIPipeline()
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def call_llm(query: str) -> str:
    """Send approved query to Llama 3 via Groq."""
    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
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


def governed_chat(user_query: str,
                  jurisdiction: str = "EU") -> dict:
    """
    YOUR FRAMEWORK as gatekeeper.
    Only ALLOW queries reach the LLM.
    """
    jur_map = {
        "US"    : Jurisdiction.US,
        "EU"    : Jurisdiction.EU,
        "INDIA" : Jurisdiction.INDIA,
        "GLOBAL": Jurisdiction.GLOBAL,
    }

    # ── Step 1: Run through YOUR pipeline ─────
    inp    = PipelineInput(
        query        = user_query,
        jurisdiction = jur_map.get(jurisdiction, Jurisdiction.EU),
    )
    result = pipeline.run(inp)
    dec    = result.final_decision

    # ── Step 2: Gate decision ─────────────────
    if dec == FinalDecision.BLOCK:
        return {
            "decision"    : "BLOCK",
            "response"    : "🚫 Query blocked by RAI governance system.",
            "reason"      : result.steps[-1].detail[:120],
            "scm_risk"    : f"{result.scm_risk_pct:.1f}%",
            "attack_type" : result.attack_type.value,
            "latency_ms"  : result.total_ms,
        }

    elif dec == FinalDecision.EXPERT_REVIEW:
        return {
            "decision" : "ESCALATE",
            "response" : "👤 Query flagged for human expert review.",
            "latency_ms": result.total_ms,
        }

    elif dec == FinalDecision.WARN:
        llm_response = call_llm(user_query)
        return {
            "decision"  : "WARN",
            "response"  : llm_response,
            "notice"    : "⚠️ Sensitive topic — response monitored.",
            "scm_risk"  : f"{result.scm_risk_pct:.1f}%",
            "latency_ms": result.total_ms,
        }

    else:  # ALLOW
        llm_response = call_llm(user_query)
        return {
            "decision"  : "ALLOW",
            "response"  : llm_response,
            "scm_risk"  : f"{result.scm_risk_pct:.1f}%",
            "latency_ms": result.total_ms,
        }


# ── Terminal Chat Loop ─────────────────────────
def run_chat():
    print("\n" + "="*60)
    print("  RESPONSIBLE AI GOVERNED CHATBOT")
    print("  Framework v5.0 + Llama 3.3 (Groq)")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        result = governed_chat(query)

        dec = result["decision"]
        print(f"\n[{dec}] ({result.get('scm_risk','N/A')} risk | "
              f"{result.get('latency_ms',0):.0f}ms)")

        if dec == "BLOCK":
            print(f"🚫 {result['response']}")
            print(f"   Reason: {result['reason']}")
        elif dec == "WARN":
            print(f"⚠️  {result['notice']}")
            print(f"AI: {result['response']}")
        elif dec == "ESCALATE":
            print(f"👤 {result['response']}")
        else:
            print(f"AI: {result['response']}")
        print()


if __name__ == "__main__":
    run_chat()