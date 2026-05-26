"""Minimal OpenAI-compatible chat client used by the simplified extractor.

Single-attempt by design: any failure (timeout, HTTP error, malformed payload)
raises a typed `ApiCallError`. The orchestrator records the failure to
`problems.jsonl` and re-runs the case in a later batch — there is no internal
retry, no exponential backoff.

DashScope extras supported:
  - Context cache: create_context() → context_id, then pass context_id to chat_json()
  - Batch API: upload_batch_file() + create_batch() + get_batch() + get_file_content()
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any
from functools import wraps

def retry_on_network_error(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            err: Exception | None = None
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (socket.timeout, urllib.error.URLError, ConnectionError, ConnectionResetError) as e:
                    err = e
                    if retries >= max_retries:
                        break
                    sleep_time = backoff_factor ** retries
                    print(f"Network error in {func.__name__}: {str(e)}. Retrying in {sleep_time}s... ({retries + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    retries += 1
                except ApiCallError as e:
                    if e.kind not in ("timeout", "http_5xx", "http_429"):
                        raise e
                    err = e
                    if retries >= max_retries:
                        break
                    sleep_time = backoff_factor ** retries
                    print(f"ApiCallError ({e.kind}) in {func.__name__}: {e.detail}. Retrying in {sleep_time}s... ({retries + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    retries += 1

            if err:
                raise err
        return wrapper
    return decorator


DEFAULT_TIMEOUT_SECONDS = 180


def _read_rate_limit_state(handle: Any) -> dict[str, Any]:
    handle.seek(0)
    text = handle.read().strip()
    if not text:
        return {"timestamps": [], "last_call": 0.0}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"timestamps": [], "last_call": 0.0}
    if not isinstance(data, dict):
        return {"timestamps": [], "last_call": 0.0}
    timestamps = data.get("timestamps")
    if not isinstance(timestamps, list):
        timestamps = []
    data["timestamps"] = [float(ts) for ts in timestamps if isinstance(ts, (int, float))]
    data["last_call"] = float(data.get("last_call") or 0.0)
    return data


def _write_rate_limit_state(handle: Any, data: dict[str, Any]) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(data, handle, ensure_ascii=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _apply_file_rate_limit_from_env() -> None:
    """Throttle API calls across subprocesses with a shared JSON state file.

    Environment variables:
      EXTRACTION_RATE_LIMIT_FILE
      EXTRACTION_MAX_CALLS_PER_WINDOW
      EXTRACTION_RATE_WINDOW_SECONDS
      EXTRACTION_MIN_CALL_INTERVAL_SECONDS
      EXTRACTION_RATE_LIMIT_SAFETY_SLEEP_SECONDS
    """
    path_text = os.environ.get("EXTRACTION_RATE_LIMIT_FILE")
    if not path_text:
        return
    max_calls = int(os.environ.get("EXTRACTION_MAX_CALLS_PER_WINDOW", "0") or "0")
    window_seconds = float(os.environ.get("EXTRACTION_RATE_WINDOW_SECONDS", "0") or "0")
    min_interval = float(os.environ.get("EXTRACTION_MIN_CALL_INTERVAL_SECONDS", "0") or "0")
    safety_sleep = float(os.environ.get("EXTRACTION_RATE_LIMIT_SAFETY_SLEEP_SECONDS", "0.25") or "0.25")
    if max_calls <= 0 and min_interval <= 0:
        return

    import fcntl

    rate_path = Path(path_text)
    rate_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        now = time.time()
        wait_seconds = 0.0
        wait_reason = None
        with rate_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = _read_rate_limit_state(handle)
                timestamps = [
                    ts
                    for ts in state.get("timestamps", [])
                    if window_seconds <= 0 or now - float(ts) < window_seconds
                ]
                last_call = float(state.get("last_call") or 0.0)
                if max_calls > 0 and window_seconds > 0 and len(timestamps) >= max_calls:
                    wait_seconds = max(0.0, window_seconds - (now - min(timestamps))) + safety_sleep
                    wait_reason = "rolling_window_quota"
                elif min_interval > 0 and last_call > 0 and now - last_call < min_interval:
                    wait_seconds = max(0.0, min_interval - (now - last_call)) + safety_sleep
                    wait_reason = "min_call_interval"
                else:
                    timestamps.append(now)
                    state["timestamps"] = timestamps
                    state["last_call"] = now
                    state["updated_at"] = now
                    state["count_in_window"] = len(timestamps)
                    state["max_calls_per_window"] = max_calls
                    state["rate_window_seconds"] = window_seconds
                    state["min_call_interval_seconds"] = min_interval
                    state.pop("last_wait_reason", None)
                    _write_rate_limit_state(handle, state)
                    return
                state["timestamps"] = timestamps
                state["updated_at"] = now
                state["count_in_window"] = len(timestamps)
                state["max_calls_per_window"] = max_calls
                state["rate_window_seconds"] = window_seconds
                state["min_call_interval_seconds"] = min_interval
                state["last_wait_reason"] = wait_reason
                state["last_wait_seconds"] = wait_seconds
                _write_rate_limit_state(handle, state)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        time.sleep(max(wait_seconds, safety_sleep))


class ApiCallError(Exception):
    """Single-shot LLM call failure.

    `kind` is one of:
        timeout       — socket/urllib timeout
        http_429      — rate limited (worth retrying in next batch)
        http_4xx      — client error other than 429 (likely a payload issue)
        http_5xx      — server error
        connect       — DNS / TCP / TLS / connection refused
        payload       — server returned 200 but body wasn't valid JSON / no choices
        no_json       — model output didn't contain a JSON object
    """

    def __init__(self, kind: str, http_status: int | None, detail: str) -> None:
        self.kind = kind
        self.http_status = http_status
        self.detail = detail
        super().__init__(f"{kind}{f' ({http_status})' if http_status else ''}: {detail}")

    def to_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "http_status": self.http_status, "detail": self.detail}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        use_json_schema: bool = False,
        default_temperature: float = 0.0,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.use_json_schema = use_json_schema
        self.default_temperature = default_temperature
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.seed = seed

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        base_url = os.environ["EXTRACTION_API_BASE"]
        api_key = os.environ["EXTRACTION_API_KEY"]
        model = os.environ["EXTRACTION_MODEL"]
        timeout_seconds = int(os.environ.get("EXTRACTION_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SECONDS)))
        use_json_schema = os.environ.get("EXTRACTION_USE_JSON_SCHEMA", "0") == "1"
        default_temperature = float(os.environ.get("EXTRACTION_TEMPERATURE", "0"))
        enable_thinking_env = os.environ.get("EXTRACTION_ENABLE_THINKING")
        enable_thinking = None
        if enable_thinking_env is not None:
            enable_thinking = enable_thinking_env.strip().lower() in {"1", "true", "yes", "on"}
        thinking_budget_env = os.environ.get("EXTRACTION_THINKING_BUDGET")
        thinking_budget = int(thinking_budget_env) if thinking_budget_env else None
        seed_env = os.environ.get("EXTRACTION_SEED")
        seed = int(seed_env) if seed_env else None
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            use_json_schema=use_json_schema,
            default_temperature=default_temperature,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            seed=seed,
        )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @retry_on_network_error()
    def _post_json(self, path: str, body: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        """POST JSON to `{base_url}/{path}`, return parsed response dict."""
        request = urllib.request.Request(
            url=f"{self.base_url}/{path.lstrip('/')}",
            data=json.dumps(body).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )
        t = timeout or self.timeout_seconds
        try:
            with urllib.request.urlopen(request, timeout=t) as resp:
                body_bytes = resp.read().decode("utf-8")
        except socket.timeout as exc:
            raise ApiCallError("timeout", None, f"socket timeout after {t}s") from exc
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            kind = "http_429" if exc.code == 429 else ("http_4xx" if 400 <= exc.code < 500 else "http_5xx")
            raise ApiCallError(kind, exc.code, detail[:1000]) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                raise ApiCallError("timeout", None, f"socket timeout after {t}s") from exc
            raise ApiCallError("connect", None, str(reason)) from exc
        try:
            return json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise ApiCallError("payload", None, f"response not JSON: {exc}") from exc

    @retry_on_network_error()
    def _get_json(self, path: str, timeout: int | None = None) -> dict[str, Any]:
        """GET `{base_url}/{path}`, return parsed response dict."""
        request = urllib.request.Request(
            url=f"{self.base_url}/{path.lstrip('/')}",
            headers={k: v for k, v in self._auth_headers().items() if k != "Content-Type"},
            method="GET",
        )
        t = timeout or self.timeout_seconds
        try:
            with urllib.request.urlopen(request, timeout=t) as resp:
                body_bytes = resp.read().decode("utf-8")
        except socket.timeout as exc:
            raise ApiCallError("timeout", None, f"socket timeout after {t}s") from exc
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            kind = "http_429" if exc.code == 429 else ("http_4xx" if 400 <= exc.code < 500 else "http_5xx")
            raise ApiCallError(kind, exc.code, detail[:1000]) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                raise ApiCallError("timeout", None, f"socket timeout after {t}s") from exc
            raise ApiCallError("connect", None, str(reason)) from exc
        try:
            return json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise ApiCallError("payload", None, f"response not JSON: {exc}") from exc

    # ------------------------------------------------------------------
    # Context cache (DashScope-specific)
    # ------------------------------------------------------------------

    def create_context(self, messages: list[dict[str, str]], ttl_seconds: int = 3600) -> str:
        """Create a DashScope context cache from `messages` (typically [system]).

        Returns the `context_id` string. The cache lives for `ttl_seconds`
        (default 1 h). Subsequent calls to `chat_json(..., context_id=...)` will
        prepend this cached content, saving its tokens on every call.

        Endpoint: POST {base_url}/context/create
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if ttl_seconds:
            body["ttl"] = ttl_seconds
        data = self._post_json("/context/create", body, timeout=60)
        context_id = data.get("id") or data.get("context_id")
        if not context_id:
            raise ApiCallError("payload", None, f"context/create response missing id: {data}")
        return str(context_id)

    # ------------------------------------------------------------------
    # Chat completion (online / direct mode)
    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float | None = None,
        context_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Single-shot JSON chat completion. Raises `ApiCallError` on failure.

        If `context_id` is provided the cached content is prepended by the API
        server; supply only the non-cached messages (typically just the user
        message) in `messages`.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
        }
        if context_id:
            payload["context_id"] = context_id
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }

        _apply_file_rate_limit_from_env()
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except socket.timeout as exc:
            raise ApiCallError("timeout", None, f"socket timeout after {self.timeout_seconds}s") from exc
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            if exc.code == 429:
                kind = "http_429"
            elif 400 <= exc.code < 500:
                kind = "http_4xx"
            else:
                kind = "http_5xx"
            raise ApiCallError(kind, exc.code, detail[:1000]) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                raise ApiCallError("timeout", None, f"socket timeout after {self.timeout_seconds}s") from exc
            raise ApiCallError("connect", None, str(reason)) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiCallError("payload", None, f"response body is not JSON: {exc}") from exc

        try:
            content = _extract_content(data)
        except _ContentError as exc:
            raise ApiCallError("payload", None, str(exc)) from exc

        try:
            parsed = _extract_json(content)
        except _JsonError as exc:
            raise ApiCallError("no_json", None, str(exc)) from exc

        usage = _normalize_usage(data.get("usage"))
        return parsed, usage

    # ------------------------------------------------------------------
    # Batch API (DashScope OpenAI-compatible batch)
    # ------------------------------------------------------------------

    @retry_on_network_error(max_retries=3, backoff_factor=3)
    def upload_batch_file(self, jsonl_lines: list[str]) -> str:
        """Upload a JSONL batch input file. Returns the DashScope file ID.

        Each line in `jsonl_lines` must be a JSON string matching:
          {"custom_id": "...", "method": "POST",
           "url": "/v1/chat/completions", "body": {...}}
        """
        import subprocess
        import json
        import os
        import tempfile

        print(f"[upload] joining {len(jsonl_lines)} jsonl lines in memory …", flush=True)
        import time as _time
        _t0 = _time.time()
        # Stream to tempfile line-by-line to avoid doubling memory via join.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as tmp:
            for line in jsonl_lines:
                tmp.write(line.encode("utf-8"))
                tmp.write(b"\n")
            tmp_path = tmp.name
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"[upload] wrote {tmp_path} ({size_mb:.1f} MB) in {_time.time()-_t0:.1f}s", flush=True)

        try:
            # --progress-bar gives live upload feedback to stderr so you can see
            # bytes actually moving. -N disables buffering on stdout/stderr.
            cmd = [
                "curl", "-N", "--progress-bar", "--fail-with-body",
                "--connect-timeout", "30",
                "--max-time", "7200",
                "--expect100-timeout", "30",
                "-X", "POST",
                f"{self.base_url}/files",
                "-H", f"Authorization: Bearer {self.api_key}",
                "-F", "purpose=batch",
                "-F", f"file=@{tmp_path};type=application/jsonl",
            ]
            print(f"[upload] launching curl (max 2h) …", flush=True)
            _t1 = _time.time()
            # Don't capture stderr so the progress bar streams to the terminal live.
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, text=True, check=False, timeout=7300
            )
            print(f"[upload] curl exited rc={result.returncode} in {_time.time()-_t1:.1f}s", flush=True)
            
            if result.returncode != 0:
                raise ApiCallError("connect", None, f"curl rc={result.returncode}: {(result.stdout or '')[:1000]}")
            
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise ApiCallError("payload", None, f"Failed to parse dashscope response: {result.stdout}")
                
            if "error" in data:
                raise ApiCallError("http_4xx", None, str(data["error"]))

            file_id = data.get("id")
            if not file_id:
                raise ApiCallError("payload", None, f"file upload response missing id: {data}")
            return str(file_id)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def create_batch(self, input_file_id: str, completion_window: str = "24h") -> str:
        """Submit a batch job. Returns the batch ID."""
        data = self._post_json("/batches", {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
        }, timeout=60)
        batch_id = data.get("id")
        if not batch_id:
            raise ApiCallError("payload", None, f"batch create response missing id: {data}")
        return str(batch_id)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        """Retrieve batch status. Relevant fields: status, output_file_id, error_file_id."""
        return self._get_json(f"/batches/{batch_id}", timeout=60)

    def get_file_content(self, file_id: str) -> list[str]:
        """Download a file's content and return non-empty lines (each is a JSON string)."""
        request = urllib.request.Request(
            url=f"{self.base_url}/files/{file_id}/content",
            headers={k: v for k, v in self._auth_headers().items() if k != "Content-Type"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ApiCallError("http_4xx" if 400 <= exc.code < 500 else "http_5xx", exc.code, detail[:1000]) from exc
        except urllib.error.URLError as exc:
            raise ApiCallError("connect", None, str(getattr(exc, "reason", exc))) from exc
        return [line for line in raw.splitlines() if line.strip()]

    def poll_batch_until_done(
        self,
        batch_id: str,
        poll_interval_seconds: int = 60,
        max_wait_hours: int = 25,
        on_status: Any = None,
    ) -> dict[str, Any]:
        """Poll `get_batch` until status is terminal. Returns the final batch record.

        `on_status` is an optional callable(status_str) called on each poll.
        Raises RuntimeError if max_wait_hours exceeded.
        """
        terminal = {"completed", "failed", "expired", "cancelled"}
        deadline = time.time() + max_wait_hours * 3600
        while True:
            batch = self.get_batch(batch_id)
            status = batch.get("status", "unknown")
            if on_status:
                on_status(status)
            if status in terminal:
                return batch
            if time.time() > deadline:
                raise RuntimeError(f"Batch {batch_id} not done after {max_wait_hours}h (last status: {status})")
            time.sleep(poll_interval_seconds)


class _ContentError(Exception):
    pass


class _JsonError(Exception):
    pass


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise _ContentError("API response has no choices")
    message = choices[0].get("message") or {}

    if isinstance(message.get("content"), str):
        content_str = message["content"].strip()
        if content_str:
            return content_str

    content = message.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        if text_parts:
            return "\n".join(text_parts)

    # Some OpenAI-compatible backends (for reasoning models) return the final
    # answer JSON in reasoning_content while leaving content empty.
    if isinstance(message.get("reasoning_content"), str):
        reasoning_str = message["reasoning_content"].strip()
        if reasoning_str:
            return reasoning_str

    if isinstance(message.get("parsed"), dict):
        return json.dumps(message["parsed"], ensure_ascii=False)

    raise _ContentError("could not extract text content from API response")


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        text = text.strip()
    return text


def _find_balanced_json(text: str) -> tuple[int, int]:
    first_brace = text.find("{")
    if first_brace == -1:
        return -1, -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return first_brace, i
    return first_brace, -1


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    text = _strip_markdown_fences(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = _find_balanced_json(text)
    if start == -1 or end == -1 or end <= start:
        raise _JsonError("model output did not contain a JSON object")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise _JsonError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _JsonError("model output JSON was not an object")
    return parsed


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return usage

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if prompt_tokens is None and isinstance(usage.get("input_tokens"), int):
        prompt_tokens = usage.get("input_tokens")
    if completion_tokens is None and isinstance(usage.get("output_tokens"), int):
        completion_tokens = usage.get("output_tokens")
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens

    normalized = dict(usage)
    normalized["prompt_tokens"] = int(prompt_tokens or 0)
    normalized["completion_tokens"] = int(completion_tokens or 0)
    normalized["total_tokens"] = int(total_tokens or 0)
    return normalized
