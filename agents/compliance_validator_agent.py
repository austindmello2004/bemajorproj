from typing import Dict, Any
import json
from .base_agent import BaseAgent


class ComplianceValidatorAgent(BaseAgent):
    """Validate schedules against policy rules and emit violations."""

    def __init__(self):
        super().__init__(
            name="ComplianceValidator",
            instructions="""You are a compliance auditor. Given workforce schedules and policy rules, find violations.
            Output JSON with fields: violations (list of objects with employee_id, rule_name, description, severity, suggestion),
            compliance_score (0-100), and summary.""",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        payload = messages[-1]["content"]
        try:
            context = json.loads(payload)
        except json.JSONDecodeError:
            context = {}
        prompt = f"""
        You are validating workforce schedules against policy rules.
        Input JSON:
        {json.dumps(context)}
        Return ONLY JSON with this shape:
        {{
            "compliance_score": 0-100,
            "violations": [
                {{
                    "employee_id": "string",
                    "rule_name": "string",
                    "description": "string",
                    "severity": "low|medium|high",
                    "suggestion": "how to fix"
                }}
            ],
            "summary": "short overview"
        }}
        """
        response = self._query_ollama(prompt)
        parsed = self._parse_json_safely(response)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed.setdefault("violations", [])
        parsed.setdefault("compliance_score", 50)
        parsed.setdefault("summary", "No summary provided.")
        parsed["validated_at"] = self.now_iso()
        return parsed
