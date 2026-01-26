from typing import Dict, Any, List
import json
from .base_agent import BaseAgent


class PolicyRuleAgent(BaseAgent):
    """LLM agent that converts policy text into structured rules."""

    def __init__(self):
        super().__init__(
            name="PolicyRuleAgent",
            instructions="""You convert workforce policies into explicit, machine-readable compliance rules.
            Extract maximum hours, shift constraints, rest periods, overtime rules, and compliance thresholds.
            Respond ONLY with JSON using fields: rule_name, rule_value, severity, metadata.""",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        policy_text = messages[-1]["content"]
        prompt = f"""
        Convert the following policy text into structured compliance rules.
        Return JSON with this shape:
        {{
            "rules": [
                {{"rule_name": "max_hours_per_week", "rule_value": "48", "severity": "high", "metadata": {{"unit": "hours", "period": "week"}}}},
                {{"rule_name": "min_rest_hours", "rule_value": "10", "severity": "high", "metadata": {{"unit": "hours", "context": "between_shifts"}}}},
                {{"rule_name": "max_consecutive_days", "rule_value": "6", "severity": "medium", "metadata": {{"unit": "days"}}}}
            ]
        }}
        Policy text:
        {policy_text}
        Return ONLY valid JSON.
        """
        response = self._query_ollama(prompt)
        parsed = self._parse_json_safely(response)
        rules: List[Dict[str, Any]] = parsed.get("rules") if isinstance(parsed, dict) else []
        if not rules:
            rules = [
                {
                    "rule_name": "max_hours_per_week",
                    "rule_value": "48",
                    "severity": "medium",
                    "metadata": {"note": "fallback default"},
                }
            ]
        return {"rules": rules, "generated_at": self.now_iso()}
