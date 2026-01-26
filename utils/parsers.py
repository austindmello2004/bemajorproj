import json
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pandas as pd
from pdfminer.high_level import extract_text as pdf_extract_text

try:
    from docx import Document  # type: ignore
except ImportError:  # pragma: no cover
    Document = None


def load_policy_file(uploaded_file) -> str:
    """Return plain text from an uploaded policy file (PDF or DOCX)."""
    suffix = uploaded_file.name.lower()
    data = uploaded_file.read()
    if suffix.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            return pdf_extract_text(tmp.name)
    if suffix.endswith(".docx") and Document:
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _extract_value(row: pd.Series, keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return default


def parse_schedule_file(uploaded_file) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse CSV/JSON/XLSX schedule files into a normalized structure."""
    name = uploaded_file.name
    suffix = name.lower()
    if suffix.endswith(".json"):
        payload = json.load(uploaded_file)
    elif suffix.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        payload = df.to_dict(orient="records")
    elif suffix.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        payload = df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported file format. Use CSV, JSON, or XLSX.")

    items = normalize_schedule_records(payload)
    return {"name": name, "source_type": suffix, "payload": payload}, items


def normalize_schedule_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in records:
        row = {k.lower(): v for k, v in raw.items()}
        normalized.append(
            {
                "employee_id": str(row.get("employee", row.get("employee_id", row.get("id", "unknown")))).strip(),
                "shift_date": str(row.get("date", row.get("shift_date", ""))).strip(),
                "start_time": str(row.get("start", row.get("start_time", ""))).strip(),
                "end_time": str(row.get("end", row.get("end_time", ""))).strip(),
                "role": str(row.get("role", row.get("position", ""))).strip(),
                "location": str(row.get("location", "")).strip(),
                "metadata": raw,
            }
        )
    return normalized
