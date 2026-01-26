from typing import Dict, Any, List
import json
from datetime import datetime
from openai import OpenAI


class BaseAgent:
    def __init__(self, name: str, instructions: str, embedding_model: str = "nomic-embed-text"):
        self.name = name
        self.instructions = instructions
        self.embedding_model = embedding_model
        # Ollama provides an OpenAI-compatible endpoint; api_key is unused but required by the client.
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Default run method to be overridden by child classes"""
        raise NotImplementedError("Subclasses must implement run()")

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama model with the given prompt"""
        try:
            response = self.ollama_client.chat.completions.create(
                model="llama3.2",  # Updated to llama3.2
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector using the local Ollama embedding model"""
        try:
            result = self.ollama_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            # openai v2 style response
            return result.data[0].embedding
        except Exception as exc:
            print(f"Error creating embedding: {exc}")
            return []

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from text, handling potential errors"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Try to find JSON-like content between curly braces
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    json_str = text[start : end + 1]
                    return json.loads(json_str)
                return {"error": "No JSON content found"}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON content", "raw": text}

    @staticmethod
    def now_iso() -> str:
        """Return a consistent ISO timestamp"""
        return datetime.utcnow().isoformat()
