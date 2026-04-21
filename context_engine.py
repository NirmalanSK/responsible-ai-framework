# context_engine.py — RAI Framework v5.0
# Contextual Memory Layer for Multi-Turn Attack Detection
#
# Architecture:
#   NOW  → used by chatbot layer (governed_chatbot.py / gemini_governed_chatbot.py)
#   FUTURE → same file, same DB, plug directly into pipeline_v15.py
#
# SQLite schema designed ONCE — no migration needed when moving to pipeline layer.
# The columns populated in Year 2 (scm_risk_raw, matrix_score, legal_score)
# are already in the schema as NULL — just start writing to them.

from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── Default DB location (sits in framework root, shared by both chatbots) ──
# Override via ContextEngine(db_path="...") if needed.
_DEFAULT_DB = Path(__file__).parent / "rai_context.db"


@dataclass
class MemoryEntry:
    """One conversation turn stored in memory."""
    turn:       int
    prompt:     str
    risk_score: float   # 0.0 – 1.0  (scm_risk_pct / 100)
    decision:   str     # ALLOW / WARN / BLOCK / EXPERT_REVIEW
    timestamp:  str


class ContextEngine:
    """
    Persistent SQLite-backed contextual memory for the RAI Framework.

    Detects gradual-escalation (slow-boil) multi-turn jailbreak attacks
    by tracking risk scores across conversation turns.

    Layer compatibility
    -------------------
    Chatbot layer (NOW):
        - source = "groq" | "gemini"
        - llm_model, llm_response populated
        - scm_risk_raw / matrix_score / legal_score = NULL

    Pipeline layer (FUTURE — Year 2):
        - source = "pipeline"
        - All columns populated
        - Same DB, same schema, zero migration

    Usage
    -----
        engine = ContextEngine()
        sid = engine.new_session(source="groq")

        # After each pipeline call:
        engine.add_turn(sid, query, risk_score=0.12, decision="ALLOW", ...)

        # Before each pipeline call (or after, see note in governed_chat):
        is_risky, score, reason = engine.detect_cumulative_risk(sid, current_risk)
    """

    def __init__(
        self,
        db_path: str | Path = None,
        session_ttl_hours: int = 24,
    ):
        self.db_path     = str(db_path or _DEFAULT_DB)
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._init_db()

    # ── DB Initialisation ───────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- Session / source identification
                    session_id  TEXT    NOT NULL,
                    source      TEXT    NOT NULL DEFAULT 'chatbot',
                    turn        INTEGER NOT NULL,

                    -- Query
                    prompt      TEXT    NOT NULL,
                    jurisdiction TEXT,

                    -- Pipeline outputs (available NOW via chatbot wrapper)
                    risk_score  REAL,       -- normalised 0.0–1.0
                    decision    TEXT,       -- ALLOW / WARN / BLOCK / EXPERT_REVIEW
                    attack_type TEXT,       -- e.g. PROMPT_INJECTION (nullable now)
                    latency_ms  REAL,

                    -- LLM layer (chatbot only; NULL when called from pipeline)
                    llm_model   TEXT,
                    llm_response TEXT,

                    -- ── Year 2 pipeline fields (NULL now, written later) ──
                    -- These columns exist TODAY so the DB never needs ALTER TABLE.
                    scm_risk_raw  REAL,     -- raw SCM causal score
                    matrix_score  REAL,     -- 17×5 sparse matrix score
                    legal_score   REAL,     -- legal layer score
                    cascade_bonus REAL,     -- cascade risk bonus
                    uncertainty   REAL,     -- epistemic uncertainty

                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Fast lookup by session + chronological order
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_turn
                ON conversation_memory(session_id, turn)
            """)
            # Fast lookup for cleanup + time-window queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON conversation_memory(timestamp)
            """)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ── Session Management ──────────────────────────────────────────────

    def new_session(self, source: str = "chatbot") -> str:
        """
        Create a unique session ID. Call ONCE per chatbot startup.

        Format: <source>_<12-char uuid hex>
        Example: groq_a3f9c1d82e4b
        """
        return f"{source}_{uuid.uuid4().hex[:12]}"

    def _next_turn(self, session_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM conversation_memory WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            return (row[0] or 0) + 1

    # ── Write ───────────────────────────────────────────────────────────

    def add_turn(
        self,
        session_id:   str,
        prompt:       str,
        risk_score:   float,            # 0.0 – 1.0
        decision:     str,
        source:       str  = "chatbot",
        jurisdiction: str  = "GLOBAL",
        llm_model:    Optional[str]   = None,
        llm_response: Optional[str]   = None,
        attack_type:  Optional[str]   = None,
        latency_ms:   Optional[float] = None,
        # ── Year 2 fields — pass None now, fill later ──
        scm_risk_raw:  Optional[float] = None,
        matrix_score:  Optional[float] = None,
        legal_score:   Optional[float] = None,
        cascade_bonus: Optional[float] = None,
        uncertainty:   Optional[float] = None,
    ) -> int:
        """
        Persist one conversation turn.  Returns the turn number.

        Notes
        -----
        - risk_score must be in [0, 1].  Pass (pipeline.scm_risk_pct / 100).
        - llm_response can be None for BLOCK decisions (no LLM was called).
        - All Year-2 fields default to None and are simply stored as NULL.
        """
        turn = self._next_turn(session_id)

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO conversation_memory
                (session_id, source, turn, prompt, jurisdiction,
                 risk_score, decision, attack_type, latency_ms,
                 llm_model, llm_response,
                 scm_risk_raw, matrix_score, legal_score,
                 cascade_bonus, uncertainty)
                VALUES (?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?, ?,?)
            """, (
                session_id, source, turn, prompt, jurisdiction,
                risk_score, decision, attack_type, latency_ms,
                llm_model, llm_response,
                scm_risk_raw, matrix_score, legal_score,
                cascade_bonus, uncertainty,
            ))
            conn.commit()

        return turn

    # ── Read ────────────────────────────────────────────────────────────

    def get_history(
        self,
        session_id: str,
        max_turns:  int = 10,
    ) -> List[MemoryEntry]:
        """Return the most recent `max_turns` turns for this session."""
        cutoff = datetime.now() - self.session_ttl
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT turn, prompt, risk_score, decision, timestamp
                FROM conversation_memory
                WHERE session_id = ? AND timestamp > ?
                ORDER BY turn ASC
                LIMIT ?
            """, (session_id, cutoff.isoformat(), max_turns)).fetchall()

        return [MemoryEntry(*r) for r in rows]

    # ── Cumulative Risk Detection ────────────────────────────────────────

    def detect_cumulative_risk(
        self,
        session_id:   str,
        current_risk: float,        # 0.0 – 1.0, from the CURRENT pipeline call
        threshold:    float = 0.55,
    ) -> Tuple[bool, float, str]:
        """
        Detect slow-boil multi-turn jailbreak attempts.

        Uses THREE signals (all must be checked; first positive wins):
            1. Linear-regression slope  → rising risk trend
            2. Recent spike             → sudden jump in last 3 turns
            3. Average threshold        → sustained high-risk conversation

        Returns
        -------
        (is_risky, avg_score, reason_string)

        Usage note
        ----------
        Call this AFTER pipeline.run() so current_risk = real SCM score.
        If is_risky and avg_score > 0.60, override decision to BLOCK.
        """
        history = self.get_history(session_id, max_turns=8)

        if len(history) < 2:
            return False, current_risk, "Insufficient history"

        # Build risk series: historical + current
        risks = [h.risk_score for h in history] + [current_risk]
        n     = len(risks)
        avg   = sum(risks) / n

        # ── Signal 1: Escalating trend (linear regression slope) ────────
        if n >= 3:
            x_bar = (n - 1) / 2.0
            y_bar = avg
            num   = sum((i - x_bar) * (risks[i] - y_bar) for i in range(n))
            den   = sum((i - x_bar) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0.0

            if slope > 0.03 and avg > 0.35:
                return (
                    True, avg,
                    f"Escalating trend — slope={slope:.3f}, avg={avg:.2f}"
                )

        # ── Signal 2: Recent spike (last 3 vs earlier) ──────────────────
        if n >= 4:
            recent = sum(risks[-3:]) / 3
            older  = sum(risks[:-3]) / max(n - 3, 1)
            if recent > older + 0.15 and recent > 0.40:
                return (
                    True, avg,
                    f"Spike detected — {older:.2f} → {recent:.2f}"
                )

        # ── Signal 3: Sustained average above threshold ──────────────────
        if avg > threshold:
            return True, avg, f"Avg risk {avg:.2f} exceeds threshold {threshold}"

        # ── Signal 4: Block-count pattern (v15h NEW) ─────────────────────
        # Even when risk scores are low (pipeline missed individual patterns),
        # 3+ BLOCKs in a session indicate a persistent attacker probing the system.
        # This catches: topic-reset attacks, academic framing after prior BLOCKs,
        # and session consolidation attacks where each query looks benign alone.
        # do(analyse_session) → BLOCK decisions stored in DB → cumulative intent visible
        block_decisions = sum(1 for h in history if h.decision == "BLOCK")
        if block_decisions >= 3:
            return (
                True, avg,
                f"Block-count signal — {block_decisions} BLOCKs in last {len(history)} turns "
                f"(session escalation pattern)"
            )

        return False, avg, "OK"

    # ── Export ──────────────────────────────────────────────────────────

    def export_session_csv(self, session_id: str, out_path: str) -> int:
        """
        Export all turns of a session to CSV.

        Returns number of rows written (0 = empty session).
        Columns match what pipeline_v15 will eventually write,
        so the CSV format is stable across both deployment layers.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    turn, source, prompt, decision, risk_score,
                    attack_type, jurisdiction, llm_model,
                    latency_ms, timestamp,
                    scm_risk_raw, matrix_score, legal_score
                FROM conversation_memory
                WHERE session_id = ?
                ORDER BY turn ASC
            """, (session_id,)).fetchall()

        if not rows:
            return 0

        headers = [
            "turn", "source", "prompt", "decision", "risk_score",
            "attack_type", "jurisdiction", "llm_model",
            "latency_ms", "timestamp",
            "scm_risk_raw", "matrix_score", "legal_score",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

        return len(rows)

    # ── Maintenance ─────────────────────────────────────────────────────

    def cleanup_old_sessions(self) -> int:
        """
        Delete sessions older than TTL from the database.
        Call periodically (e.g., at chatbot startup).
        Returns number of rows deleted.
        """
        cutoff = datetime.now() - self.session_ttl
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM conversation_memory WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            conn.commit()
            return cur.rowcount

    def session_summary(self, session_id: str) -> dict:
        """
        Return a quick summary dict for a session.
        Useful for end-of-session reporting in interactive mode.
        """
        history = self.get_history(session_id, max_turns=1000)
        if not history:
            return {"turns": 0}

        risks     = [h.risk_score for h in history]
        decisions = [h.decision   for h in history]

        return {
            "turns":     len(history),
            "avg_risk":  round(sum(risks) / len(risks), 3),
            "max_risk":  round(max(risks), 3),
            "blocked":   decisions.count("BLOCK"),
            "warned":    decisions.count("WARN"),
            "allowed":   decisions.count("ALLOW"),
        }
