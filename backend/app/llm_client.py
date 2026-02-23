import json
from typing import Any, Dict
from urllib import error, request

from app.config import get_settings


class LLMClientError(RuntimeError):
    pass


class LLMClientConfigError(LLMClientError):
    pass


class LLMClientTransientError(LLMClientError):
    pass


class LLMClientResponseError(LLMClientError):
    pass


def provider_is_stub() -> bool:
    settings = get_settings()
    return settings.llm_provider.strip().lower() == "stub"


def _extract_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def chat_complete(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider == "stub":
        raise LLMClientConfigError("LLM provider が stub に設定されています")
    if provider not in {"openai", "openai_compatible"}:
        raise LLMClientConfigError(f"未対応の LLM provider です: {settings.llm_provider}")
    if not settings.llm_api_key:
        raise LLMClientConfigError("LLM_PROVIDER が stub 以外の場合は LLM_API_KEY が必要です")

    base_url = settings.llm_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    timeout = max(1, int(settings.llm_timeout_seconds))

    def post_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=timeout) as res:
                raw = res.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"LLM HTTP エラー: {exc.code} {detail[:400]}"
            if exc.code in {400, 401, 403, 404, 422}:
                raise LLMClientConfigError(message) from exc
            if exc.code == 429 or 500 <= exc.code <= 599:
                raise LLMClientTransientError(message) from exc
            raise LLMClientResponseError(message) from exc
        except error.URLError as exc:
            raise LLMClientTransientError(f"LLM 接続エラー: {exc.reason}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LLMClientResponseError("LLM 応答が JSON ではありません") from exc

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # OpenAI hosted models (e.g. gpt-5 family) expect max_completion_tokens.
    if provider == "openai":
        payload["reasoning_effort"] = "low"
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens

    parsed = post_chat(payload)
    text = _extract_content(parsed)
    if text:
        return text

    # Some reasoning models may consume all completion tokens for reasoning.
    # Retry once with a larger completion budget for openai provider.
    if provider == "openai":
        retry_payload = dict(payload)
        retry_payload["max_completion_tokens"] = min(max_tokens * 2, 4000)
        parsed = post_chat(retry_payload)
        text = _extract_content(parsed)
        if text:
            return text

    finish_reason = ""
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = str(choices[0].get("finish_reason") or "")
    if finish_reason:
        raise LLMClientResponseError(f"LLM 応答に本文が含まれていません (finish_reason={finish_reason})")
    raise LLMClientResponseError("LLM 応答に本文が含まれていません")
