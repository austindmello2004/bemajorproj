import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ComplianceDatabase:
    """SQLite-backed store for policies, rules, schedules, and violations."""

    def __init__(self):
        current_dir = Path(__file__).parent
        self.db_path = current_dir / "compliance.sqlite"
        self.schema_path = current_dir / "compliance_schema.sql"
        self._init_db()

    def _init_db(self) -> None:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")
        with open(self.schema_path) as f:
            schema = f.read()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)

    # --- Policy storage ---
    def add_policy(self, title: str, source: str, raw_text: str) -> int:
        query = "INSERT INTO policies (title, source, raw_text) VALUES (?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (title, source, raw_text))
            return cur.lastrowid

    def add_policy_chunks(self, policy_id: int, chunks: List[Dict[str, Any]]) -> None:
        query = """
        INSERT INTO policy_chunks (policy_id, chunk_index, content, embedding)
        VALUES (?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    query,
                    (
                        policy_id,
                        chunk.get("chunk_index", idx),
                        chunk["content"],
                        json.dumps(chunk.get("embedding", [])),
                    ),
                )

    def add_policy_rules(self, policy_id: int, rules: List[Dict[str, Any]]) -> None:
        query = """
        INSERT INTO policy_rules (policy_id, rule_name, rule_value, severity, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for rule in rules:
                cur.execute(
                    query,
                    (
                        policy_id,
                        rule.get("rule_name") or rule.get("name") or "rule",
                        rule.get("rule_value") or rule.get("value") or "",
                        rule.get("severity", "medium"),
                        json.dumps(rule.get("metadata", {})),
                    ),
                )

    def list_policy_rules(self, policy_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, policy_id, rule_name, rule_value, severity, metadata FROM policy_rules"
        params: Tuple[Any, ...] = ()
        if policy_id:
            query += " WHERE policy_id = ?"
            params = (policy_id,)
        query += " ORDER BY created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "policy_id": row["policy_id"],
                    "rule_name": row["rule_name"],
                    "rule_value": row["rule_value"],
                    "severity": row["severity"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    def list_policies(self) -> List[Dict[str, Any]]:
        query = "SELECT id, title, source, created_at FROM policies ORDER BY created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def get_latest_policy_id(self) -> Optional[int]:
        query = "SELECT id FROM policies ORDER BY created_at DESC LIMIT 1"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            row = cur.execute(query).fetchone()
            return row[0] if row else None

    # --- Schedule storage ---
    def add_schedule(self, name: str, source_type: str, raw_payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> int:
        query = "INSERT INTO schedules (name, source_type, raw_payload, metadata) VALUES (?, ?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (name, source_type, json.dumps(raw_payload), json.dumps(metadata or {})))
            return cur.lastrowid

    def add_schedule_items(self, schedule_id: int, items: List[Dict[str, Any]]) -> None:
        query = """
        INSERT INTO schedule_items (schedule_id, employee_id, shift_date, start_time, end_time, role, location, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for item in items:
                cur.execute(
                    query,
                    (
                        schedule_id,
                        item.get("employee_id"),
                        item.get("shift_date"),
                        item.get("start_time"),
                        item.get("end_time"),
                        item.get("role"),
                        item.get("location"),
                        json.dumps(item.get("metadata", {})),
                    ),
                )

    def add_violation(self, schedule_id: int, employee_id: str, rule_id: Optional[int], description: str, severity: str, details: Optional[Dict[str, Any]] = None) -> int:
        query = """
        INSERT INTO violations (schedule_id, employee_id, rule_id, description, severity, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                query,
                (schedule_id, employee_id, rule_id, description, severity, json.dumps(details or {})),
            )
            return cur.lastrowid

    def list_violations(self, schedule_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, schedule_id, employee_id, rule_id, description, severity, details, created_at FROM violations"
        params: Tuple[Any, ...] = ()
        if schedule_id:
            query += " WHERE schedule_id = ?"
            params = (schedule_id,)
        query += " ORDER BY created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "schedule_id": row["schedule_id"],
                    "employee_id": row["employee_id"],
                    "rule_id": row["rule_id"],
                    "description": row["description"],
                    "severity": row["severity"],
                    "details": json.loads(row["details"]) if row["details"] else {},
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def add_corrected_schedule(self, schedule_id: int, corrected_payload: Dict[str, Any], explanation: str) -> int:
        query = "INSERT INTO corrected_schedules (schedule_id, corrected_payload, explanation) VALUES (?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (schedule_id, json.dumps(corrected_payload), explanation))
            return cur.lastrowid

    def list_corrected_schedules(self, schedule_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, schedule_id, corrected_payload, explanation, created_at FROM corrected_schedules"
        params: Tuple[Any, ...] = ()
        if schedule_id:
            query += " WHERE schedule_id = ?"
            params = (schedule_id,)
        query += " ORDER BY created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "schedule_id": row["schedule_id"],
                    "corrected_payload": json.loads(row["corrected_payload"]),
                    "explanation": row["explanation"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # --- Retrieval helpers ---
    def _fetch_chunk_embeddings(self) -> List[Tuple[int, str, List[float]]]:
        query = "SELECT id, content, embedding FROM policy_chunks"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            results = []
            for row in rows:
                embedding = json.loads(row["embedding"]) if row["embedding"] else []
                results.append((row["id"], row["content"], embedding))
            return results

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return chunks most similar to the query embedding using cosine similarity."""
        if not query_embedding:
            return []
        stored = self._fetch_chunk_embeddings()
        if not stored:
            return []

        query_vec = np.array(query_embedding)
        scores: List[Tuple[float, int, str]] = []
        for chunk_id, content, emb in stored:
            emb_vec = np.array(emb)
            if emb_vec.size == 0:
                continue
            denom = np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
            if denom == 0:
                continue
            score = float(np.dot(query_vec, emb_vec) / denom)
            scores.append((score, chunk_id, content))
        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": score, "chunk_id": chunk_id, "content": content}
            for score, chunk_id, content in scores[:top_k]
        ]