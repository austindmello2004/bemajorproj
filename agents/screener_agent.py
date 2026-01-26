from typing import Dict, Any
import json
from .base_agent import BaseAgent


class ScreenerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Screener",
            instructions="""Screen candidates based on:
            - Qualification alignment
            - Experience relevance
            - Skill match percentage
            - Cultural fit indicators
            - Red flags or concerns
            Provide comprehensive screening reports.""",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Screen the candidate"""
        print("👥 Screener: Conducting initial screening")

        try:
            workflow_context = json.loads(messages[-1]["content"])
        except json.JSONDecodeError:
            workflow_context = {}
        screening_results = self._query_ollama(json.dumps(workflow_context))

        return {
            "screening_report": screening_results,
            "screening_timestamp": self.now_iso(),
            "screening_score": 85,
        }
