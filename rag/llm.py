"""
LLM Client — Google Gemini API wrapper using the google-genai SDK.

Configuration (via .env or environment variables):
  LLM_PROVIDER     — "vertexai", "gemini", or "ollama".
  PROJECT_ID       — GCP project ID (required when LLM_PROVIDER=vertexai).
  VERTEX_LOCATION  — Vertex AI location (optional, default: "global").
  GEMINI_API_KEY   — required when LLM_PROVIDER=gemini.
  GEMINI_MODEL     — optional. Default: gemini-2.5-flash-lite.

Falls back to a formatted text response if no LLM provider is available.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

# Load .env from project root (graceful — dotenv is optional)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually

log = logging.getLogger("bds_llm")


class LLMClient:
    """
    LLM wrapper supporting Google Gemini API and Ollama.

    Falls back to format_without_llm() if the chosen LLM provider is not available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._provider = provider or os.environ.get("LLM_PROVIDER", "").lower()
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

        self._project_id = os.environ.get("PROJECT_ID", "")
        self._vertex_location = os.environ.get("VERTEX_LOCATION", "global")

        self._ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

        self._client = None
        self._available = False

        # Default provider resolution
        if not self._provider:
            if self._project_id:
                self._provider = "vertexai"
            elif self._api_key and self._api_key != "your_gemini_api_key_here":
                self._provider = "gemini"
            else:
                self._provider = "ollama"

        if self._provider == "ollama":
            import requests
            try:
                resp = requests.get(self._ollama_base_url, timeout=2.0)
                if resp.status_code == 200:
                    self._available = True
                    log.info(f"Ollama LLM initialized (Model: {self._ollama_model}, Endpoint: {self._ollama_base_url})")
                else:
                    log.warning(f"Ollama server returned status {resp.status_code} at {self._ollama_base_url}")
            except Exception as e:
                log.warning(f"Failed to connect to Ollama server at {self._ollama_base_url}: {e}")

        elif self._provider == "vertexai":
            if not self._project_id:
                log.warning("LLM_PROVIDER=vertexai but PROJECT_ID is not set. Disabling LLM.")
            else:
                try:
                    from google import genai
                    self._client = genai.Client(
                        vertexai=True,
                        project=self._project_id,
                        location=self._vertex_location,
                    )
                    self._available = True
                    log.info(
                        f"Vertex AI LLM initialized (Model: {self._model_name}, "
                        f"Project: {self._project_id}, Location: {self._vertex_location})"
                    )
                except ImportError:
                    log.warning("google-genai not installed. Run: pip install google-genai")
                except Exception as e:
                    log.warning(f"Failed to initialize Vertex AI client: {e}")

        else:  # gemini (API key)
            self._provider = "gemini"
            if self._api_key and self._api_key != "your_gemini_api_key_here":
                try:
                    from google import genai
                    self._client = genai.Client(api_key=self._api_key)
                    self._available = True
                    log.info(f"Gemini LLM initialized ({self._model_name})")
                except ImportError:
                    log.warning(
                        "google-genai not installed. "
                        "Run: pip install google-genai"
                    )
                except Exception as e:
                    log.warning(f"Failed to initialize Gemini: {e}")
            else:
                log.info(
                    "GEMINI_API_KEY not set. LLM generation disabled, "
                    "using formatted text fallback."
                )

    @property
    def is_available(self) -> bool:
        """Whether the LLM is available for generation."""
        return self._available

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a response using the selected LLM provider.

        Args:
            prompt: User prompt (with RAG context injected)
            system_prompt: System instruction for the model
            max_tokens: Maximum output tokens
            temperature: 0 = deterministic/factual, 1 = creative

        Returns:
            Generated text, or empty string on failure.
        """
        if not self._available:
            return ""

        if self._provider == "ollama":
            import requests
            import re
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": self._ollama_model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": ["Observation:", "Observation:\n", "Observation: ", "Observation:  "],
                    },
                    "stream": False,
                }
                
                resp = requests.post(
                    f"{self._ollama_base_url}/api/chat",
                    json=payload,
                    timeout=300.0  # Allow reasoning models plenty of time (300 seconds)
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                
                # Strip reasoning <think>...</think> tags if present to keep UI clean
                content_clean = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()
                return content_clean
            except Exception as e:
                log.error(f"Ollama generation error [{self._ollama_model}]: {e}")
                return ""

        else:
            import time
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                return ""

            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system_prompt or "",
                stop_sequences=["Observation:", "Observation:\n"],
            )

            # Try primary model, then fallback models on rate-limit
            models_to_try = [self._model_name]
            if self._model_name not in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"):
                models_to_try.append("gemini-2.0-flash")

            last_error = None
            for model in models_to_try:
                for attempt in range(3):
                    try:
                        response = self._client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=config,
                        )
                        # For Gemini 2.5 thinking models, extract only non-thought parts
                        text = LLMClient._extract_non_thought_text(response)
                        if text:
                            return text.strip()
                        return "(Không thể tạo câu trả lời. Vui lòng thử lại.)"

                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            # Fail-fast: only wait 1 second on first retry, then fail fast to fallback
                            if attempt == 0:
                                wait = 1
                                log.warning(f"Rate limited on {model} (attempt {attempt+1}/3), waiting {wait}s…")
                                time.sleep(wait)
                            else:
                                log.warning(f"Rate limited on {model} (attempt {attempt+1}/3), failing fast to next model/fallback.")
                            last_error = e
                            continue
                        # Non-retryable error — log and try next model
                        log.error(f"LLM generation error [{model}]: {e}")
                        last_error = e
                        break  # move to next model

            log.error(f"All LLM models exhausted. Last error: {last_error}")
            return ""


    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Stream a response token-by-token from the selected LLM provider.

        Yields:
            Text chunks (strings) as they arrive from the model.
            Yields nothing if the LLM is not available.
        """
        if not self._available:
            return

        if self._provider == "ollama":
            import requests
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": self._ollama_model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": True,
                }

                resp = requests.post(
                    f"{self._ollama_base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=300.0,
                )
                resp.raise_for_status()

                think_state = "before"  # states: "before", "inside", "after"
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    token = data.get("message", {}).get("content", "")
                    if not token:
                        continue

                    # On-the-fly <think>...</think> strip state machine
                    if think_state == "before":
                        if "<think>" in token:
                            think_state = "inside"
                            # Yield any text before the <think> tag
                            before = token.split("<think>")[0]
                            if before:
                                yield before
                        else:
                            yield token
                    elif think_state == "inside":
                        if "</think>" in token:
                            think_state = "after"
                            after = token.split("</think>", 1)[-1]
                            if after:
                                yield after
                        # Skip tokens inside <think>
                    elif think_state == "after":
                        yield token

                    if data.get("done"):
                        break

            except Exception as e:
                log.error(f"Ollama streaming error [{self._ollama_model}]: {e}")
                return

        else:
            # Gemini / Vertex AI streaming — filter out thought tokens from thinking models
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                return

            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system_prompt or "",
            )

            models_to_try = [self._model_name]
            if self._model_name not in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"):
                models_to_try.append("gemini-2.0-flash")

            for model in models_to_try:
                try:
                    stream = self._client.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    for chunk in stream:
                        # Iterate parts directly to skip thought tokens
                        try:
                            parts = chunk.candidates[0].content.parts
                            for part in parts:
                                if getattr(part, "thought", False):
                                    continue  # skip thinking tokens
                                if part.text:
                                    yield part.text
                        except (IndexError, AttributeError):
                            # Fallback if structure differs
                            if chunk.text:
                                yield chunk.text
                    return  # Success — stop trying other models
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        log.warning(f"Rate limited streaming on {model}, trying next model…")
                        continue
                    log.error(f"Gemini streaming error [{model}]: {e}")
                    return

    @staticmethod
    def _extract_non_thought_text(response) -> str:
        """
        Extract only non-thought text from a Gemini response.

        Gemini 2.5 thinking models return thought tokens in parts with
        `part.thought == True`. This helper skips those parts and returns
        only the final answer text.
        """
        try:
            parts = response.candidates[0].content.parts
            texts = [
                part.text
                for part in parts
                if not getattr(part, "thought", False) and part.text
            ]
            return "".join(texts)
        except (IndexError, AttributeError):
            # Fallback: use response.text which may include thought tokens
            return response.text or ""

    @staticmethod
    def format_without_llm(
        query: str,
        documents: list,
        intent: str = "search_listing",
    ) -> str:
        """
        Format retrieved results without using an LLM.
        Provides a clean, readable output even without API access.

        Args:
            query: Original user query
            documents: List of RetrievedDocument objects
            intent: Query intent for formatting style

        Returns:
            Formatted text response
        """
        if not documents:
            return (
                f"🔍 Không tìm thấy kết quả phù hợp cho: \"{query}\"\n\n"
                "Gợi ý: Thử mở rộng tiêu chí tìm kiếm hoặc sử dụng từ khóa khác."
            )

        lines = [f"🔍 Kết quả tìm kiếm: \"{query}\"\n"]
        lines.append(f"📊 Tìm thấy {len(documents)} kết quả phù hợp\n")
        lines.append("=" * 60)

        for i, doc in enumerate(documents):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            text = doc.text if hasattr(doc, "text") else str(doc)
            score = doc.score if hasattr(doc, "score") else 0
            coll = doc.collection if hasattr(doc, "collection") else ""

            lines.append(f"\n--- Kết quả #{i+1} (Độ phù hợp: {score:.1%}) ---")
            lines.append(f"📂 Nguồn: {coll}")

            if intent == "search_listing":
                title = meta.get("tieu_de", "")
                if title:
                    lines.append(f"📌 {title}")
                gia = meta.get("gia_raw", "")
                dt = meta.get("dien_tich_raw", "")
                if gia or dt:
                    lines.append(f"💰 Giá: {gia or 'N/A'} | 📐 DT: {dt or 'N/A'}")
                loc = meta.get("tinh_thanh", "") or ""
                dist = meta.get("quan_huyen", "") or ""
                if loc or dist:
                    lines.append(f"📍 {dist}, {loc}".strip(", "))

            elif intent == "lifestyle_search":
                # No special metadata structure; text preview is enough
                pass

            # Show truncated text
            preview = text[:300].strip()
            if len(text) > 300:
                preview += "..."
            lines.append(f"\n{preview}")

            url = meta.get("url", "")
            if url:
                lines.append(f"🔗 {url}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
