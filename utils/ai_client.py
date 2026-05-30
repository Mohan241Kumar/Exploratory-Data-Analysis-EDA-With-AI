import os
import json
from typing import Any, Dict

# Defer importing groq or provide a clear error if missing
try:
    from groq import Groq  # type: ignore
except Exception as import_error:  # ImportError or other issues
    raise ImportError(
        "Missing dependency 'groq'. Install it with 'pip install groq' and run Streamlit using the same interpreter (e.g., 'py -m streamlit run app.py')."
    ) from import_error


class AIClient:
    def __init__(self, api_key: str, model_name: str = "qwen/qwen3-32b") -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        # Allow override via environment variable GROQ_MODEL
        self.model_name = os.environ.get("GROQ_MODEL", model_name)

    def chat_json(self, prompt: str) -> Dict[str, Any]:
        safety_instructions = (
            "Respond ONLY with valid JSON. Do not include markdown fences or prose."
        )
        full_prompt = f"{safety_instructions}\n\n{prompt}"
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content if resp.choices else "{}"
        try:
            return json.loads(text)
        except Exception:
            # Best-effort: try to extract JSON blob
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            return {"error": "Invalid JSON from model", "raw": text}


def get_ai_client() -> AIClient:
    api_key = os.environ.get("GROQ_API_KEY")
    return AIClient(api_key)


