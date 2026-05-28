#!/usr/bin/env python3
"""
Local LLM Proxy — multi-key × multi-model fallback
OpenAI:  /v1/chat/completions, /v1/embeddings
Google:  /v1beta/models/{model}:generateContent|streamGenerateContent|batchEmbedContents
"""
import os
import sys
import json
import time
import random
from datetime import datetime
from flask import Flask, request, Response, jsonify
import requests

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEYS = [k.strip() for k in os.environ.get("GOOGLE_API_KEYS", "").split(",") if k.strip()]
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "sk-proxy-kimi")
PORT = int(os.environ.get("PORT", 18080))
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_GATEWAY_ID = os.environ.get("CLOUDFLARE_GATEWAY_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

# Fallback from .env individual keys if GOOGLE_API_KEYS not set
if not API_KEYS:
    for i in range(1, 10):
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "")
        if k:
            API_KEYS.append(k)

if not API_KEYS:
    print("ERROR: No API keys. Set GOOGLE_API_KEYS or GOOGLE_API_KEY_1..N", file=sys.stderr)
    sys.exit(1)

PRIMARY_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3-flash-preview"]
FALLBACK_MODELS = ["gemini-2.5-flash", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
ALL_MODELS = PRIMARY_MODELS + FALLBACK_MODELS + ["gemini-embedding-2", "text-embedding-3-small"]
AUTO_MODEL = "auto"

def is_auto_model(model):
    return not model or model == AUTO_MODEL or model == "gemini-auto"

FINISH_MAP = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
              "RECITATION": "content_filter", "OTHER": "stop"}

# ── State ────────────────────────────────────────────────────────────────────
req_count = 0
request_log = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def log_req(method, path, model, status, duration_ms, key_hint):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    request_log.append((ts, method, path, model, status, duration_ms, key_hint))
    if len(request_log) > 500:
        request_log.pop(0)
    print(f"[{ts}] {status:3d} {method:4s} {path[:60]} {model or ''} {duration_ms}ms")


def next_key_offset():
    global req_count
    idx = req_count % len(API_KEYS)
    req_count += 1
    return idx


def auth_check():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {AUTH_TOKEN}":
        return jsonify({"error": {"message": "Unauthorized", "type": "auth_error"}}), 401
    return None


def cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


def json_resp(data, status=200, extra_headers=None):
    resp = jsonify(data)
    resp.status_code = status
    cors_headers(resp)
    if extra_headers:
        for k, v in extra_headers.items():
            resp.headers[k] = v
    return resp


# ── Google API call ──────────────────────────────────────────────────────────
def call_google(key, model, body, action="generateContent", stream=False):
    use_cf = bool(CF_API_TOKEN)
    if use_cf:
        url = (f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}/"
               f"google-ai-studio/v1beta/models/{model}:{action}"
               f"{'?alt=sse' if stream else ''}")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key,
            "Authorization": f"Bearer {CF_API_TOKEN}",
        }
    else:
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:{action}{'?alt=sse' if stream else ''}")
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}

    resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT, stream=stream)
    return resp


# ── Message conversion ───────────────────────────────────────────────────────
def openai_to_google(body):
    contents = []
    system = ""
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system += ("\n" if system else "") + msg.get("content", "")
            continue

        # assistant 消息带 tool_calls → functionCall parts
        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg["tool_calls"]:
                if tc.get("type") == "function" and tc.get("function"):
                    import json as _json
                    try:
                        args = _json.loads(tc["function"].get("arguments", "{}"))
                    except Exception:
                        args = {}
                    parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})
            contents.append({"role": "model", "parts": parts})
            continue

        # tool 消息 → functionResponse parts
        if role == "tool":
            import json as _json
            resp_content = msg.get("content", "")
            try:
                resp_content = _json.loads(resp_content)
            except Exception:
                pass
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {"name": msg.get("name", "unknown"), "response": {"result": resp_content}}}]
            })
            continue

        content = msg.get("content", "")
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        import base64
                        import re
                        m = re.match(r"data:([^;]+);base64,(.+)", url)
                        if m:
                            parts.append({"inlineData": {"mimeType": m.group(1), "data": m.group(2)}})
                    else:
                        parts.append({"text": f"[image: {url}]"})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

    if system and contents:
        first = contents[0]
        if first["parts"] and "text" in first["parts"][0]:
            first["parts"][0]["text"] = system + "\n\n" + first["parts"][0]["text"]
        elif first["parts"]:
            first["parts"].insert(0, {"text": system})
        else:
            first["parts"] = [{"text": system}]

    config = {}
    if body.get("max_tokens") is not None:
        config["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        config["topP"] = body["top_p"]
    if body.get("top_k") is not None:
        config["topK"] = body["top_k"]
    if body.get("stop"):
        config["stopSequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    if body.get("response_format"):
        rf = body["response_format"]
        if rf.get("type") == "json_object":
            config["responseMimeType"] = "application/json"
        elif rf.get("type") == "json_schema":
            config["responseMimeType"] = "application/json"
            if rf.get("json_schema", {}).get("schema"):
                config["responseSchema"] = rf["json_schema"]["schema"]

    result = {"contents": contents}
    if config:
        result["generationConfig"] = config

    # OpenAI tools → Google functionDeclarations + 搜索接地
    tools = []
    search_tool_names = {"google_search", "google_search_retrieval"}
    if body.get("tools"):
        declarations = []
        for t in body["tools"]:
            if t.get("type") == "function" and t.get("function"):
                if t["function"].get("name") not in search_tool_names:
                    declarations.append({
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "parameters": t["function"].get("parameters", {}),
                    })
        if declarations:
            tools.append({"functionDeclarations": declarations})

    # web_search_options → Google 搜索接地
    if body.get("web_search_options"):
        ctx = body["web_search_options"].get("search_context_size", "medium")
        if ctx == "high":
            tools.append({"googleSearchRetrieval": {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.3}}})
        else:
            tools.append({"googleSearch": {}})

    # tools 中包含 google_search / web_search 特殊标记
    if body.get("tools"):
        for t in body["tools"]:
            if t.get("type") == "function" and t.get("function", {}).get("name") == "google_search":
                tools.append({"googleSearch": {}})
            if t.get("type") == "function" and t.get("function", {}).get("name") == "google_search_retrieval":
                tools.append({"googleSearchRetrieval": t["function"].get("parameters", {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.5}})})

    if tools:
        result["tools"] = tools
    return result


def to_openai(google_resp, model):
    c = google_resp.get("candidates", [{}])[0] if google_resp.get("candidates") else {}
    usage = google_resp.get("usageMetadata", {})
    finish = c.get("finishReason") or "STOP"
    parts = c.get("content", {}).get("parts", [])

    message = {"role": "assistant"}
    text_parts = [p for p in parts if p.get("text")]
    func_parts = [p for p in parts if p.get("functionCall")]

    if text_parts:
        message["content"] = "".join(p["text"] for p in text_parts)
    elif not func_parts:
        message["content"] = ""

    # Google functionCall → OpenAI tool_calls
    if func_parts:
        message["tool_calls"] = [{
            "id": f"call_{random.randint(10**24, 10**25)}",
            "type": "function",
            "function": {
                "name": p["functionCall"]["name"],
                "arguments": json.dumps(p["functionCall"].get("args", {})),
            },
        } for p in func_parts]
        if "content" not in message:
            message["content"] = None

    result = {
        "id": f"chatcmpl-{random.randint(10**24, 10**25)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": FINISH_MAP.get(finish, "stop")}],
        "usage": {"prompt_tokens": usage.get("promptTokenCount", 0),
                  "completion_tokens": usage.get("candidatesTokenCount", 0),
                  "total_tokens": usage.get("totalTokenCount", 0)},
    }

    # Google 搜索接地 → search_results
    gm = c.get("groundingMetadata")
    if gm:
        search_results = [{"url": ch["web"]["uri"], "title": ch["web"]["title"]}
                          for ch in gm.get("groundingChunks", []) if ch.get("web")]
        if search_results:
            result["search_results"] = search_results
        queries = gm.get("webSearchQueries")
        if queries:
            result["search_queries"] = queries

    return result


# ── Fallback engine ──────────────────────────────────────────────────────────
def build_chains():
    key_start = random.randint(0, len(API_KEYS) - 1)
    primary = [{"key": API_KEYS[(key_start + i) % len(API_KEYS)], "model": m}
               for i, m in enumerate(PRIMARY_MODELS)]
    fallback = [{"key": API_KEYS[(key_start + i) % len(API_KEYS)], "model": m}
                for i, m in enumerate(FALLBACK_MODELS)]
    return primary, fallback


def with_fallback(fn):
    """Try primary chain then fallback chain. fn(key, model) → result or raise."""
    primary, fallback = build_chains()
    last_error = None

    for chain, label in [(primary, "primary"), (fallback, "fallback")]:
        for item in chain:
            key, model = item["key"], item["model"]
            key_hint = key[:10] + "..."
            t0 = time.time()
            try:
                result = fn(key, model)
                duration = int((time.time() - t0) * 1000)
                log_req("POST", request.path, model, 200, duration, key_hint)
                return result, model, key_hint
            except requests.exceptions.Timeout:
                duration = int((time.time() - t0) * 1000)
                log_req("POST", request.path, model, 504, duration, key_hint)
                print(f"  [{label}:{model}] timeout {duration}ms, skip")
                last_error = {"message": "Request timed out", "type": "timeout_error", "status": 504}
                continue
            except requests.exceptions.HTTPError as e:
                duration = int((time.time() - t0) * 1000)
                status = e.response.status_code if e.response is not None else 503
                log_req("POST", request.path, model, status, duration, key_hint)
                if status == 429:
                    print(f"  [{label}:{model}] 429 rate limited, skip")
                    last_error = {"message": "Rate limit exceeded", "type": "rate_limit_error", "status": 429}
                    continue
                try:
                    err_body = e.response.json() if e.response is not None else {}
                except Exception:
                    err_body = {}
                last_error = {"message": err_body.get("error", {}).get("message", str(e)),
                              "type": "upstream_error", "status": status,
                              "google_error": err_body.get("error")}
                print(f"  [{label}:{model}] HTTP {status}, skip")
                continue
            except Exception as e:
                duration = int((time.time() - t0) * 1000)
                log_req("POST", request.path, model, 503, duration, key_hint)
                print(f"  [{label}:{model}] {type(e).__name__}: {e}")
                last_error = {"message": str(e), "type": "fallback_exhausted", "status": 503}
                continue

    return None, None, last_error


# ── Streaming ────────────────────────────────────────────────────────────────
def stream_openai_sse(google_resp, model):
    """Convert Google streaming SSE → OpenAI streaming SSE."""
    first = True
    finish_sent = False
    for line in google_resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [SSE] malformed JSON: {raw[:200]}")
            continue
        if ev.get("error"):
            yield f"data: {json.dumps({'error': ev['error']})}\n\n"
            continue
        cand = ev.get("candidates", [{}])[0] if ev.get("candidates") else {}
        text = cand.get("content", {}).get("parts", [{}])[0].get("text", "") if cand.get("content") else ""
        finish = cand.get("finishReason")
        usage = ev.get("usageMetadata")
        if first:
            delta = {"role": "assistant", "content": text} if text else {"role": "assistant"}
            chunk = _make_chunk(model, delta, None)
            yield f"data: {json.dumps(chunk)}\n\n"
            first = False
        elif text:
            chunk = _make_chunk(model, {"content": text}, None)
            yield f"data: {json.dumps(chunk)}\n\n"
        if finish:
            finish_sent = True
            fr = FINISH_MAP.get(finish, "stop")
            chunk = _make_chunk(model, {}, fr)
            yield f"data: {json.dumps(chunk)}\n\n"
            # 搜索接地结果
            gm = cand.get("groundingMetadata")
            if gm:
                search_results = [{"url": ch["web"]["uri"], "title": ch["web"]["title"]}
                                  for ch in gm.get("groundingChunks", []) if ch.get("web")]
                if search_results:
                    search_chunk = {
                        "id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "search_results": search_results,
                        "search_queries": gm.get("webSearchQueries", []),
                    }
                    yield f"data: {json.dumps(search_chunk)}\n\n"
            if usage:
                usage_chunk = {
                    "id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0),
                    },
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
    if not finish_sent:
        chunk = _make_chunk(model, {}, "stop")
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _make_chunk(model, delta, finish_reason):
    return {
        "id": f"chatcmpl-{random.randint(10**24, 10**25)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def stream_raw_sse(google_resp):
    """Pass-through Google streaming SSE."""
    for line in google_resp.iter_lines():
        if line:
            yield line + b"\n\n"
    yield b"data: [DONE]\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        resp = Response("", 204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "keys": len(API_KEYS), "models": PRIMARY_MODELS + FALLBACK_MODELS})


@app.route("/logs", methods=["GET"])
def logs():
    limit = int(request.args.get("limit", 50))
    out = []
    for entry in reversed(request_log[-limit:]):
        out.append({"timestamp": entry[0], "method": entry[1], "path": entry[2],
                     "model": entry[3], "status": entry[4],
                     "duration_ms": entry[5], "key_hint": entry[6]})
    return jsonify(out)


@app.route("/v1/models", methods=["GET"])
def list_models():
    models = [{"id": AUTO_MODEL, "object": "model", "created": 0, "owned_by": "proxy"}]
    models.extend({"id": m, "object": "model", "created": 0, "owned_by": "google"}
                  for m in ALL_MODELS)
    return jsonify({"object": "list", "data": models})


# ── OpenAI Chat Completions ──────────────────────────────────────────────────
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except Exception:
        return json_resp({"error": {"message": "Invalid JSON"}}, 400)

    if not body.get("messages") or not isinstance(body["messages"], list):
        return json_resp({"error": {"message": "messages array required"}}, 400)

    is_stream = body.get("stream", False)
    requested_model = body.get("model", AUTO_MODEL)
    google_body = openai_to_google(body)

    if is_auto_model(requested_model):
        # auto 模式：走完整 key × model fallback
        def do_call(key, model):
            resp = call_google(key, model, google_body, stream=is_stream)
            resp.raise_for_status()
            return resp

        result, used_model, info = with_fallback(do_call)
        if result is None:
            status = info.get("status", 503) if isinstance(info, dict) else 503
            return json_resp({"error": info or {"message": "All keys/models exhausted"}}, status)
    else:
        # 指定模型：只做 key 轮询
        used_model = requested_model
        result = None
        last_error = None
        start_idx = next_key_offset()
        for offset in range(len(API_KEYS)):
            idx = (start_idx + offset) % len(API_KEYS)
            key = API_KEYS[idx]
            key_hint = key[:10] + "..."
            t0 = time.time()
            try:
                resp = call_google(key, requested_model, google_body, stream=is_stream)
                resp.raise_for_status()
                duration = int((time.time() - t0) * 1000)
                log_req("POST", "/v1/chat/completions", requested_model, 200, duration, key_hint)
                result = resp
                break
            except requests.exceptions.Timeout:
                duration = int((time.time() - t0) * 1000)
                log_req("POST", "/v1/chat/completions", requested_model, 504, duration, key_hint)
                last_error = {"message": "Request timed out", "type": "timeout_error", "status": 504}
                continue
            except requests.exceptions.HTTPError as e:
                duration = int((time.time() - t0) * 1000)
                status = e.response.status_code if e.response is not None else 503
                log_req("POST", "/v1/chat/completions", requested_model, status, duration, key_hint)
                if status == 429:
                    last_error = {"message": "Rate limit exceeded", "type": "rate_limit_error", "status": 429}
                    continue
                last_error = {"message": str(e), "type": "upstream_error", "status": status}
                continue
            except Exception as e:
                duration = int((time.time() - t0) * 1000)
                log_req("POST", "/v1/chat/completions", requested_model, 503, duration, key_hint)
                last_error = {"message": str(e), "type": "fallback_exhausted", "status": 503}
                continue

        if result is None:
            status = last_error.get("status", 503) if isinstance(last_error, dict) else 503
            return json_resp({"error": last_error or {"message": "All keys exhausted"}}, status)

    if is_stream:
        return Response(
            stream_openai_sse(result, used_model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "x-model": used_model,
                     "Access-Control-Allow-Origin": "*"},
        )
    else:
        data = result.json()
        return json_resp(to_openai(data, used_model), extra_headers={"x-model": used_model})


# ── OpenAI Embeddings ────────────────────────────────────────────────────────
@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except Exception:
        return json_resp({"error": {"message": "Invalid JSON"}}, 400)

    if body.get("stream"):
        return json_resp({"error": {"message": "Streaming not supported for embeddings"}}, 400)

    model = body.get("model", "gemini-embedding-2")
    input_texts = body.get("input", body.get("input_array", ""))
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    # Map embedding model names → gemini-embedding-2-preview (the actual model name)
    EMBED_MODEL_MAP = {
        "embedding-2": "gemini-embedding-2-preview",
        "gemini-embedding-2": "gemini-embedding-2-preview",
        "text-embedding-3-small": "gemini-embedding-2-preview",
    }
    google_model = EMBED_MODEL_MAP.get(model, model)

    gemini_body = {
        "requests": [{"model": f"models/{google_model}", "content": {"parts": [{"text": str(t)}]}}
                     for t in input_texts]
    }

    start_idx = next_key_offset()
    last_error = None
    for offset in range(len(API_KEYS)):
        idx = (start_idx + offset) % len(API_KEYS)
        key = API_KEYS[idx]
        key_hint = key[:10] + "..."
        t0 = time.time()
        try:
            resp = call_google(key, google_model, gemini_body, action="batchEmbedContents", stream=False)
            duration = int((time.time() - t0) * 1000)
            log_req("POST", "/v1/embeddings", model, resp.status_code, duration, key_hint)
            if resp.ok:
                data = resp.json()
                embeddings_data = data.get("embeddings", [])
                return json_resp({
                    "object": "list",
                    "data": [{"object": "embedding", "index": i, "embedding": e.get("values", [])}
                             for i, e in enumerate(embeddings_data)],
                    "model": model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }, extra_headers={"x-model": model})
            if resp.status_code == 429:
                print(f"  [embedding:{model}] 429 rate limited, skip")
                last_error = "Rate limit exceeded"
                continue
            last_error = resp.text[:500]
        except Exception as e:
            duration = int((time.time() - t0) * 1000)
            log_req("POST", "/v1/embeddings", model, 503, duration, key_hint)
            last_error = str(e)
            continue

    return json_resp({"error": {"message": last_error or "All keys exhausted",
                                 "type": "fallback_exhausted"}}, 503)


# ── Google Native API ────────────────────────────────────────────────────────
@app.route("/v1beta/models/<model>:generateContent", methods=["POST"])
def google_generate(model):
    return _handle_google_native(model, stream=False)


@app.route("/v1beta/models/<model>:streamGenerateContent", methods=["POST"])
def google_stream(model):
    return _handle_google_native(model, stream=True)


@app.route("/v1beta/models/<model>:batchEmbedContents", methods=["POST"])
def google_embed(model):
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except Exception:
        return json_resp({"error": {"message": "Invalid JSON"}}, 400)

    start_idx = next_key_offset()
    last_error = None
    for offset in range(len(API_KEYS)):
        idx = (start_idx + offset) % len(API_KEYS)
        key = API_KEYS[idx]
        key_hint = key[:10] + "..."
        t0 = time.time()
        try:
            resp = call_google(key, model, body, action="batchEmbedContents", stream=False)
            duration = int((time.time() - t0) * 1000)
            log_req("POST", request.path, model, resp.status_code, duration, key_hint)
            if resp.ok:
                return Response(resp.content, resp.status_code,
                                headers={"Content-Type": "application/json",
                                         "x-model": model,
                                         "Access-Control-Allow-Origin": "*"})
            if resp.status_code == 429:
                continue
            last_error = resp.text[:500]
        except Exception as e:
            duration = int((time.time() - t0) * 1000)
            log_req("POST", request.path, model, 503, duration, key_hint)
            last_error = str(e)
            continue

    return json_resp({"error": {"message": last_error or "All keys exhausted"}}, 503)


def _handle_google_native(model, stream):
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except Exception:
        return json_resp({"error": {"message": "Invalid JSON"}}, 400)

    def _respond(resp, used_model):
        if stream:
            return Response(
                stream_raw_sse(resp),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "x-model": used_model,
                         "Access-Control-Allow-Origin": "*"},
            )
        else:
            return Response(resp.content, resp.status_code,
                            headers={"Content-Type": resp.headers.get("Content-Type", "application/json"),
                                     "x-model": used_model,
                                     "Access-Control-Allow-Origin": "*"})

    # auto 模式：走完整 key × model fallback
    if is_auto_model(model):
        def do_call(key, m):
            resp = call_google(key, m, body, stream=stream)
            resp.raise_for_status()
            return resp

        result, used_model, info = with_fallback(do_call)
        if result is None:
            status = info.get("status", 503) if isinstance(info, dict) else 503
            return json_resp({"error": info or {"message": "All keys/models exhausted"}}, status)
        return _respond(result, used_model)

    # 指定模型：只做 key 轮询
    start_idx = next_key_offset()
    last_error = None
    for offset in range(len(API_KEYS)):
        idx = (start_idx + offset) % len(API_KEYS)
        key = API_KEYS[idx]
        key_hint = key[:10] + "..."
        t0 = time.time()
        try:
            resp = call_google(key, model, body, stream=stream)
            resp.raise_for_status()
            duration = int((time.time() - t0) * 1000)
            log_req("POST", request.path, model, 200, duration, key_hint)
            return _respond(resp, model)
        except requests.exceptions.Timeout:
            duration = int((time.time() - t0) * 1000)
            log_req("POST", request.path, model, 504, duration, key_hint)
            print(f"  [google:{model}] timeout {duration}ms, skip")
            last_error = {"message": "Request timed out", "type": "timeout_error", "status": 504}
            continue
        except requests.exceptions.HTTPError as e:
            duration = int((time.time() - t0) * 1000)
            status = e.response.status_code if e.response is not None else 503
            log_req("POST", request.path, model, status, duration, key_hint)
            if status == 429:
                print(f"  [google:{model}] 429 rate limited, skip")
                last_error = {"message": "Rate limit exceeded", "type": "rate_limit_error", "status": 429}
                continue
            try:
                err_body = e.response.json() if e.response is not None else {}
            except Exception:
                err_body = {}
            last_error = {"message": err_body.get("error", {}).get("message", str(e)),
                          "type": "upstream_error", "status": status,
                          "google_error": err_body.get("error")}
            print(f"  [google:{model}] HTTP {status}, skip")
            continue
        except Exception as e:
            duration = int((time.time() - t0) * 1000)
            log_req("POST", request.path, model, 503, duration, key_hint)
            print(f"  [google:{model}] {type(e).__name__}: {e}")
            last_error = {"message": str(e), "type": "fallback_exhausted", "status": 503}
            continue

    status = last_error.get("status", 503) if isinstance(last_error, dict) else 503
    return json_resp({"error": last_error or {"message": "All keys exhausted"}}, status)


# ── Responses API ──────────────────────────────────────────────────────────

def _resp_id():
    return f"resp_{random.randint(10**24, 10**25)}"


def _msg_id():
    return f"msg_{random.randint(10**24, 10**25)}"


def responses_input_to_messages(inp, instructions):
    msgs = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})

    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
        return msgs

    if not isinstance(inp, list):
        return msgs

    for item in inp:
        if isinstance(item, str):
            msgs.append({"role": "user", "content": item})
            continue

        # EasyInputMessage
        if item.get("role") and "content" in item:
            role = "system" if item["role"] == "developer" else item["role"]
            content = item["content"]
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") in ("input_text", "text"):
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "input_image":
                        parts.append({"type": "image_url", "image_url": {"url": p.get("image_url", p.get("url", ""))}})
                msgs.append({"role": role, "content": parts})
            continue

        # ResponseOutputMessage
        if item.get("type") == "message":
            role = item.get("role", "user")
            text = "".join(c.get("text", "") for c in item.get("content", []) if c.get("type") in ("output_text", "input_text"))
            if text:
                msgs.append({"role": role, "content": text})
            continue

        # FunctionCall
        if item.get("type") == "function_call":
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id", item.get("id")),
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item.get("arguments", "{}")},
                }],
            })
            continue

        # FunctionCallOutput
        if item.get("type") == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output)
            msgs.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": output,
            })
            continue

    return msgs


def responses_tools_to_chat_tools(tools):
    if not tools:
        return None
    result = []
    for t in tools:
        if t.get("type") == "function":
            # Responses API 格式: name 在顶层
            # Chat Completions 格式: name 在 function 下
            name = t.get("name") or t.get("function", {}).get("name", "")
            desc = t.get("description") or t.get("function", {}).get("description", "")
            params = t.get("parameters") or t.get("function", {}).get("parameters", {})
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                },
            })
    return result if result else None


def chat_response_to_responses(chat_resp, response_id):
    msg = chat_resp.get("choices", [{}])[0].get("message", {})
    finish = chat_resp.get("choices", [{}])[0].get("finish_reason")
    output = []

    # 搜索接地 → url_citation annotations
    annotations = []
    for sr in chat_resp.get("search_results", []):
        annotations.append({"type": "url_citation", "url": sr["url"], "title": sr["title"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            output.append({
                "type": "function_call",
                "id": tc.get("id"),
                "call_id": tc.get("id"),
                "name": tc.get("function", {}).get("name"),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
                "status": "completed",
            })
    else:
        output.append({
            "type": "message",
            "id": _msg_id(),
            "status": "completed" if finish == "stop" else "incomplete",
            "role": "assistant",
            "content": [{"type": "output_text", "text": msg.get("content", ""), "annotations": annotations}],
        })

    usage = chat_resp.get("usage", {})
    result = {
        "id": response_id,
        "object": "response",
        "created_at": chat_resp.get("created", int(time.time())),
        "completed_at": int(time.time()),
        "status": "completed" if finish in ("stop", "tool_calls") else "incomplete",
        "model": chat_resp.get("model"),
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

    if chat_resp.get("search_queries"):
        result["search_queries"] = chat_resp["search_queries"]

    return result


def stream_responses_sse(google_resp, model, response_id):
    """Convert Google streaming SSE → Responses API streaming SSE."""
    item_id = _msg_id()
    full_text = ""

    yield _sse_event("response.created", {
        "type": "response.created",
        "response": {"id": response_id, "object": "response", "created_at": int(time.time()),
                      "status": "in_progress", "model": model, "output": [],
                      "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
    })

    yield _sse_event("response.in_progress", {
        "type": "response.in_progress",
        "response": {"id": response_id, "object": "response", "created_at": int(time.time()),
                      "status": "in_progress", "model": model, "output": [],
                      "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
    })

    yield _sse_event("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []},
    })

    yield _sse_event("response.content_part.added", {
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })

    finish_sent = False
    annotations = []
    for line in google_resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if ev.get("error"):
            yield _sse_event("error", {"type": "error", "message": ev["error"].get("message", "unknown")})
            continue

        cand = ev.get("candidates", [{}])[0] if ev.get("candidates") else {}
        text = cand.get("content", {}).get("parts", [{}])[0].get("text", "") if cand.get("content") else ""
        finish = cand.get("finishReason")
        usage = ev.get("usageMetadata")

        if text:
            full_text += text
            yield _sse_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            })

        if finish:
            finish_sent = True
            # 提取搜索接地
            gm = cand.get("groundingMetadata")
            if gm:
                for ch in gm.get("groundingChunks", []):
                    if ch.get("web"):
                        annotations.append({"type": "url_citation", "url": ch["web"]["uri"], "title": ch["web"]["title"]})

            yield _sse_event("response.output_text.done", {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            })
            yield _sse_event("response.content_part.done", {
                "type": "response.content_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text, "annotations": annotations},
            })
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "id": item_id, "status": "completed", "role": "assistant",
                         "content": [{"type": "output_text", "text": full_text, "annotations": annotations}]},
            })

            usage_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            if usage:
                usage_data = {"input_tokens": usage.get("promptTokenCount", 0),
                              "output_tokens": usage.get("candidatesTokenCount", 0),
                              "total_tokens": usage.get("totalTokenCount", 0)}

            yield _sse_event("response.completed", {
                "type": "response.completed",
                "response": {"id": response_id, "object": "response", "created_at": int(time.time()),
                              "completed_at": int(time.time()), "status": "completed", "model": model,
                              "output": [{"type": "message", "id": item_id, "status": "completed", "role": "assistant",
                                          "content": [{"type": "output_text", "text": full_text, "annotations": annotations}]}],
                              "usage": usage_data},
            })

    if not finish_sent:
        yield _sse_event("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "object": "response", "created_at": int(time.time()),
                          "completed_at": int(time.time()), "status": "completed", "model": model,
                          "output": [{"type": "message", "id": item_id, "status": "completed", "role": "assistant",
                                      "content": [{"type": "output_text", "text": full_text, "annotations": annotations}]}],
                          "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
        })


def _sse_event(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.route("/v1/responses", methods=["POST"])
def responses_api():
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except Exception:
        return json_resp({"error": {"message": "Invalid JSON"}}, 400)

    response_id = _resp_id()
    is_stream = body.get("stream", False)
    requested_model = body.get("model", AUTO_MODEL)
    messages = responses_input_to_messages(body.get("input"), body.get("instructions"))
    chat_tools = responses_tools_to_chat_tools(body.get("tools"))

    google_body = openai_to_google({"messages": messages, "tools": chat_tools,
                                     "temperature": body.get("temperature"),
                                     "top_p": body.get("top_p"),
                                     "max_tokens": body.get("max_output_tokens"),
                                     "response_format": body.get("text", {}).get("format") if body.get("text") else None})

    def do_call(key, model):
        resp = call_google(key, model, google_body, stream=is_stream)
        resp.raise_for_status()
        return resp

    if is_auto_model(requested_model):
        result, used_model, info = with_fallback(do_call)
        if result is None:
            status = info.get("status", 503) if isinstance(info, dict) else 503
            err_msg = info.get("message", "All keys/models exhausted") if isinstance(info, dict) else "All keys/models exhausted"
            return Response(f"event: error\ndata: {json.dumps({'type': 'error', 'message': err_msg})}\n\n",
                            status=status, mimetype="text/event-stream")
    else:
        used_model = requested_model
        result = None
        start_idx = next_key_offset()
        for offset in range(len(API_KEYS)):
            idx = (start_idx + offset) % len(API_KEYS)
            key = API_KEYS[idx]
            try:
                resp = call_google(key, requested_model, google_body, stream=is_stream)
                resp.raise_for_status()
                result = resp
                break
            except Exception:
                continue
        if result is None:
            return json_resp({"error": {"message": "All keys exhausted"}}, 503)

    if is_stream:
        return Response(
            stream_responses_sse(result, used_model, response_id),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "x-model": used_model, "Access-Control-Allow-Origin": "*"},
        )
    else:
        data = result.json()
        chat_resp = to_openai(data, used_model)
        resp_data = chat_response_to_responses(chat_resp, response_id)
        return json_resp(resp_data, extra_headers={"x-model": used_model})


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting local proxy on :{PORT}")
    print(f"Keys: {len(API_KEYS)} ({API_KEYS[0][:10]}... etc)")
    print(f"Primary: {PRIMARY_MODELS}")
    print(f"Fallback: {FALLBACK_MODELS}")
    if CF_API_TOKEN:
        print(f"Via Cloudflare Gateway: {CF_ACCOUNT_ID}/{CF_GATEWAY_ID}")
    else:
        print("Direct to Google API (no CF gateway)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
