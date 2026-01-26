from typing import Dict, Any
import json
from .base_agent import BaseAgent


class PolicyChatAgent(BaseAgent):
    """Chat agent that answers questions with policy/RAG context."""

    def __init__(self):
        super().__init__(
            name="PolicyChatAgent",
            instructions="""Answer workforce policy and compliance questions using the provided context.
            Keep answers concise, cite rule names when possible, and avoid inventing details not present in the context.""",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        payload = messages[-1]["content"]
        try:
            context = json.loads(payload)
        except json.JSONDecodeError:
            context = {}
        query = context.get("question", "")
        context_blocks = context.get("context", [])
        rules = context.get("rules", [])
        prompt = f"""
        You are a policy assistant. Use the provided context chunks and rules to answer the question.
        Context chunks:\n{context_blocks}\nRules:\n{rules}\nQuestion: {query}
        Provide a helpful answer, cite the most relevant rule names, and keep it concise.
        """
        response = self._query_ollama(prompt)
        return {"answer": response, "answered_at": self.now_iso()}
