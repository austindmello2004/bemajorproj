from typing import Dict, Any
import json
from .base_agent import BaseAgent


class ScheduleGeneratorAgent(BaseAgent):
    """Generate compliant schedules and explanations."""

    def __init__(self):
        super().__init__(
            name="ScheduleGenerator",
            instructions="""Given a non-compliant schedule and policy rules, propose a corrected schedule.
            Return JSON with fields: corrected_schedule (list of shifts), explanation, and rationale.""",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        payload = messages[-1]["content"]
        try:
            context = json.loads(payload)
        except json.JSONDecodeError:
            context = {}
        prompt = f"""
        Based on the provided policy rules and the current schedule with violations, propose a corrected schedule.
        Input JSON:
        {json.dumps(context)}
        Output JSON example:
        {{
            "corrected_schedule": [
                {{"employee_id": "E1", "shift_date": "2025-01-05", "start_time": "09:00", "end_time": "17:00", "role": "RN", "location": "HQ"}}
            ],
            "explanation": "Adjusted shifts to respect max hours and rest windows.",
            "rationale": "why the changes are compliant"
        }}
        Return ONLY JSON.
        """
        response = self._query_ollama(prompt)
        parsed = self._parse_json_safely(response)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed.setdefault("corrected_schedule", [])
        parsed.setdefault("explanation", "")
        parsed.setdefault("rationale", "")
        parsed["generated_at"] = self.now_iso()
        return parsed
