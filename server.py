#!/usr/bin/env python3
"""
Local LLM Proxy — multi-key round-robin fallback
OpenAI compatible: /v1/chat/completions
Google native:     /v1beta/models/{model}:generateContent
"""
import os
import json
import time
import random
from datetime import datetime
from flask import Flask, request, Response, jsonify
import requests

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
API_KEYS = os.environ.get("GOOGLE_API_KEYS", "").split(",")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "local-dev-token")
PORT = int(os.environ.get("PORT", 18080))

if len(API_KEYS) == 1 and API_KEYS[0] == "":
    raise ValueError("Set GOOGLE_API_KEYS env var (comma-separated)")

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "6b88a7cbdb5322b3c89342b38406f2b9")
CF_GATEWAY_ID = os.environ.get("CF_GATEWAY_ID", "gemma-aggregation")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")

TIMEOUT = 15
# ── State ────────────────────────────────────────────────────────────────────
req_count = 0  # simple round-robin counter (int, not guarded — occasional dup OK)
request_log = []  # [(ts, method, path, model, status, duration_ms, key_hint), ...]

# ── Helpers ──────────────────────────────────────────────────────────────────
def log_request(method, path, model, status, duration_ms, key_hint):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    request_log.append((ts, method, path, model, status, duration_ms, key_hint))
    if len(request_log) > 200:
        request_log.pop(0)
    print(f"[{ts}] {status:3d} {method:4s} {path[:50]} {model or ''} {duration_ms}ms")


def next_key():
    global req_count
    idx = req_count % len(API_KEYS)
    req_count += 1
    return API_KEYS[idx]


def auth_check():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {AUTH_TOKEN}":
        return Response(json.dumps({"error": {"message": "Unauthorized"}}), 401,
                        mimetype="application/json")
    return None


FINISH_MAP = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
              "RECITATION": "content_filter", "OTHER": "stop"}


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(part.get("text", ""))
        return "".join(chunks)
    return ""


def _parse_openai_tool_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {"raw": arguments}
    return {}


def _parse_tool_message_content(content):
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return {"content": content}
    return {"content": _as_text(content)}


def _convert_openai_content_parts(content):
    parts = []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return parts
    for part in content:
        if not isinstance(part, dict):
            continue
        p_type = part.get("type")
        if p_type == "text":
            parts.append({"text": part.get("text", "")})
        elif p_type == "image_url":
            image_url = part.get("image_url", {})
            if isinstance(image_url, str):
                url = image_url
            else:
                url = image_url.get("url", "")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                header, payload = url.split(",", 1)
                mime = header[5:].split(";")[0] or "application/octet-stream"
                parts.append({"inlineData": {"mimeType": mime, "data": payload}})
            elif isinstance(url, str) and url:
                parts.append({"fileData": {"mimeType": "image/*", "fileUri": url}})
            else:
                parts.append({"text": "[image]"})
    return parts


def _build_generation_config(openai_body):
    generation_config = {}
    if openai_body.get("temperature") is not None:
        generation_config["temperature"] = openai_body["temperature"]
    if openai_body.get("top_p") is not None:
        generation_config["topP"] = openai_body["top_p"]
    if openai_body.get("top_k") is not None:
        generation_config["topK"] = openai_body["top_k"]
    if openai_body.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = openai_body["max_tokens"]
    if openai_body.get("n") is not None:
        generation_config["candidateCount"] = openai_body["n"]
    stop = openai_body.get("stop")
    if isinstance(stop, str):
        generation_config["stopSequences"] = [stop]
    elif isinstance(stop, list):
        generation_config["stopSequences"] = [item for item in stop if isinstance(item, str)]

    response_format = openai_body.get("response_format", {})
    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        if rf_type == "json_object":
            generation_config["responseMimeType"] = "application/json"
        elif rf_type == "json_schema":
            generation_config["responseMimeType"] = "application/json"
            schema = response_format.get("json_schema", {})
            if isinstance(schema, dict):
                generation_config["responseSchema"] = schema.get("schema", schema)

    return generation_config


def _build_tool_config(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name", "")
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def _build_tools(openai_tools):
    declarations = []
    for tool in openai_tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        declarations.append({
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def to_openai(google_resp, model):
    c = google_resp.get("candidates", [{}])[0]
    usage = google_resp.get("usageMetadata", {})
    finish = c.get("finishReason") or "STOP"
    parts = c.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
    tool_calls = []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        fn_call = part.get("functionCall", {})
        name = fn_call.get("name")
        if not name:
            continue
        args = fn_call.get("args", {})
        tool_calls.append({
            "id": f"call_{random.randint(10**8, 10**10)}_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    finish_reason = FINISH_MAP.get(finish, "stop")
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"
    message = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{random.randint(10**24, 10**25)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": usage.get("promptTokenCount", 0),
                  "completion_tokens": usage.get("candidatesTokenCount", 0),
                  "total_tokens": usage.get("totalTokenCount", 0)},
    }


def openai_to_google(body):
    """Convert OpenAI messages → Google contents format."""
    contents = []
    system_parts = []
    tool_call_id_to_name = {}
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_text = _as_text(msg.get("content", ""))
            if system_text:
                system_parts.append({"text": system_text})
            continue
        parts = _convert_openai_content_parts(msg.get("content", ""))

        if role == "assistant":
            for tool_call in msg.get("tool_calls", []):
                if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
                    continue
                fn = tool_call.get("function", {})
                fn_name = fn.get("name")
                if not fn_name:
                    continue
                fn_args = _parse_openai_tool_arguments(fn.get("arguments", "{}"))
                parts.append({"functionCall": {"name": fn_name, "args": fn_args}})
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_id_to_name[tool_call_id] = fn_name
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            continue

        if role == "tool":
            fn_name = msg.get("name") or tool_call_id_to_name.get(msg.get("tool_call_id"), "tool")
            tool_resp = _parse_tool_message_content(msg.get("content", ""))
            parts = [{"functionResponse": {"name": fn_name, "response": tool_resp}}]
            contents.append({"role": "user", "parts": parts})
            continue

        contents.append({"role": "user", "parts": parts or [{"text": ""}]})

    google_body = {"contents": contents}
    if system_parts:
        google_body["systemInstruction"] = {"parts": system_parts}

    generation_config = _build_generation_config(body)
    if generation_config:
        google_body["generationConfig"] = generation_config

    tools = _build_tools(body.get("tools", []))
    if tools:
        google_body["tools"] = tools

    tool_config = _build_tool_config(body.get("tool_choice"))
    if tool_config:
        google_body["toolConfig"] = tool_config

    return google_body


def call_google(key, model, body, via_cf=True, stream=False, action=None, url_model=None):
    """model is for body/display, url_model is for URL path (defaults to model)."""
    if action is None:
        action = "streamGenerateContent" if stream else "generateContent"
    google_url_model = url_model or model
    if via_cf and CF_API_TOKEN:
        url = (f"https://gateway.ai.cloudflare.com/v1/{CF_ACCOUNT_ID}/{CF_GATEWAY_ID}/"
               f"google-ai-studio/v1beta/models/{google_url_model}:{action}"
               f"{'?alt=sse' if stream else ''}")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key,
            "Authorization": f"Bearer {CF_API_TOKEN}",
        }
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_url_model}:{action}"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    return resp


def call_chain(body, model=None, via_cf=True, stream=False):
    """Try all keys in round-robin order. Returns (Response, model, key_hint)."""
    global req_count
    global req_count
    start_idx = req_count % len(API_KEYS)
    req_count += 1
    req_count += 1
    for offset in range(len(API_KEYS)):
        idx = (start_idx + offset) % len(API_KEYS)
        key = API_KEYS[idx]
        key_hint = key[:10] + "..."
        t0 = time.time()
        try:
            resp = call_google(key, model, body, via_cf, stream)
            duration = int((time.time() - t0) * 1000)
            status = resp.status_code
            log_request("POST", request.path, model, status, duration, key_hint)
            if resp.ok:
                return resp, model, key_hint
            # 429 = rate limit → skip to next key immediately
            if status == 429:
                print(f"  [key:{idx}] 429 rate limit, skipping")
                continue
            # Other errors → still return to client (could be upstream bug)
            return resp, model, key_hint
        except requests.exceptions.Timeout:
            duration = int((time.time() - t0) * 1000)
            log_request("POST", request.path, model, 504, duration, key_hint)
            print(f"  [key:{idx}] timeout after {duration}ms, skip")
            continue
        except Exception as e:
            duration = int((time.time() - t0) * 1000)
            log_request("POST", request.path, model, 503, duration, key_hint)
            print(f"  [key:{idx}] {type(e).__name__}: {e}, skip")
            continue
    return None, model, None


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "keys": len(API_KEYS)})


@app.route("/logs", methods=["GET"])
def logs():
    limit = int(request.args.get("limit", 50))
    out = []
    for entry in reversed(request_log[-limit:]):
        out.append({
            "timestamp": entry[0], "method": entry[1], "path": entry[2],
            "model": entry[3], "status": entry[4],
            "duration_ms": entry[5], "key_hint": entry[6],
        })
    return jsonify(out)


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list", "data": [
            {"id": "gemini-2.5-flash", "object": "model", "created": 0, "owned_by": "google"},
            {"id": "gemini-3.1-flash-lite", "object": "model", "created": 0, "owned_by": "google"},
            {"id": "gemma-4-31b-it", "object": "model", "created": 0, "owned_by": "google"},
            {"id": "gemma-4-26b-a4b-it", "object": "model", "created": 0, "owned_by": "google"},
            {"id": "embedding-2", "object": "model", "created": 0, "owned_by": "google"},
        ]})


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    print(f"[EMBED] headers={dict(request.headers)}, raw_data={request.get_data()}")
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except:
        print(f"[EMBED] invalid JSON: {request.get_data()}")
        return jsonify({"error": {"message": "Invalid JSON"}}), 400

    model = body.get("model", "gemini-embedding-2")
    input_texts = body.get("input", [])
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    # Map embedding-2 → gemini-embedding-2-preview (the actual model name)
    google_model = "gemini-embedding-2-preview" if model == "embedding-2" else model

    # Convert to Gemini batchEmbedContents format
    gemini_body = {
        "requests": [
            {"model": f"models/{google_model}",
             "content": {"parts": [{"text": t}]}}
            for t in input_texts
        ]
    }

    # Try each key
    start_idx = req_count % len(API_KEYS)
    last_error = None
    for offset in range(len(API_KEYS)):
        idx = (start_idx + offset) % len(API_KEYS)
        key = API_KEYS[idx]
        key_hint = key[:10] + "..."
        t0 = time.time()
        try:
            resp = call_google(key, model, gemini_body, via_cf=False, stream=False, action="batchEmbedContents", url_model=google_model)
            duration = int((time.time() - t0) * 1000)
            log_request("POST", "/v1/embeddings", model, resp.status_code, duration, key_hint)
            if resp.ok:
                data = resp.json()
                embeddings_data = data.get("embeddings", [])
                usage_meta = data.get("usageMetadata", {})
                return jsonify({
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": e.get("values", [])}
                        for i, e in enumerate(embeddings_data)
                    ],
                    "model": model,
                    "usage": {
                        "prompt_tokens": usage_meta.get("promptTokens", 0),
                        "total_tokens": usage_meta.get("totalTokens", 0),
                    },
                })
            if resp.status_code == 429:
                continue
            last_error = resp.text
        except Exception as e:
            last_error = str(e)
            continue

    return jsonify({"error": {"message": last_error or "All keys exhausted", "type": "fallback_exhausted"}}), 503


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/v1/openai/chat/completions", methods=["POST"])
@app.route("/v1beta/openai/chat/completions", methods=["POST"])
def chat_completions():
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except:
        return jsonify({"error": {"message": "Invalid JSON"}}), 400

    is_stream = body.get("stream", False)
    model = body.get("model", "gemini-2.5-flash")
    google_body = openai_to_google(body)

    resp, used_model, key_hint = call_chain(google_body, model, stream=is_stream)
    if resp is None:
        return jsonify({"error": {"message": "All keys exhausted", "type": "fallback_exhausted"}}), 503

    if is_stream:
        return Response(
            _stream_response(resp, used_model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "x-model": used_model},
        )
    else:
        data = resp.json()
        openai_resp = to_openai(data, used_model)
        return jsonify(openai_resp)


@app.route("/v1beta/models/<model>:generateContent", methods=["POST"])
@app.route("/v1/models/<model>:generateContent", methods=["POST"])
def google_generate(model):
    return handle_google_native(model, stream=False)


@app.route("/v1beta/models/<model>:streamGenerateContent", methods=["POST"])
@app.route("/v1/models/<model>:streamGenerateContent", methods=["POST"])
def google_stream_generate(model):
    return handle_google_native(model, stream=True)


def handle_google_native(model, stream):
    err = auth_check()
    if err:
        return err
    try:
        body = request.get_json()
    except:
        return jsonify({"error": {"message": "Invalid JSON"}}), 400

    resp, used_model, key_hint = call_chain(body, model, stream=stream)
    if resp is None:
        return jsonify({"error": {"message": "All keys exhausted", "type": "fallback_exhausted"}}), 503

    if stream:
        return Response(
            _stream_raw(resp),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "x-model": used_model},
        )
    else:
        return Response(resp.content, resp.status_code,
                        headers={"Content-Type": resp.headers.get("Content-Type", "application/json"),
                                 "x-model": used_model})


def _stream_response(google_resp, model):
    """Yield OpenAI-format SSE from Google streaming response."""
    import json
    first = True
    finish_sent = False
    tool_call_index = 0
    for line in google_resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            ev = json.loads(raw)
        except:
            continue
        ev_error = ev.get("error")
        if ev_error:
            yield f"data: {json.dumps({'error': ev_error})}\n\n"
            continue
        cand = ev.get("candidates", [{}])[0]
        cand_parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in cand_parts if isinstance(p, dict) and p.get("text"))
        stream_tool_calls = []
        for part in cand_parts:
            if not isinstance(part, dict):
                continue
            fn_call = part.get("functionCall", {})
            name = fn_call.get("name")
            if not name:
                continue
            stream_tool_calls.append({
                "index": tool_call_index,
                "id": f"call_{random.randint(10**8, 10**10)}_{tool_call_index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(fn_call.get('args', {}), ensure_ascii=False)},
            })
            tool_call_index += 1
        finish = cand.get("finishReason")
        usage = ev.get("usageMetadata")
        finish_reason = FINISH_MAP.get(finish, "stop") if finish else None
        if finish_reason == "stop" and stream_tool_calls:
            finish_reason = "tool_calls"
        if first:
            delta = {"role": "assistant"}
            if text:
                delta["content"] = text
            if stream_tool_calls:
                delta["tool_calls"] = stream_tool_calls
            chunk = {"id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                     "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            first = False
        elif text or stream_tool_calls:
            delta = {}
            if text:
                delta["content"] = text
            if stream_tool_calls:
                delta["tool_calls"] = stream_tool_calls
            chunk = {"id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                     "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        if finish:
            finish_sent = True
            chunk = {"id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                     "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            if usage:
                usage_chunk = {"id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                              "object": "chat.completion.chunk",
                              "created": int(time.time()), "model": model,
                              "choices": [], "usage": {
                                  "prompt_tokens": usage.get("promptTokenCount", 0),
                                  "completion_tokens": usage.get("candidatesTokenCount", 0),
                                  "total_tokens": usage.get("totalTokenCount", 0)}}
                yield f"data: {json.dumps(usage_chunk)}\n\n"
    if not finish_sent:
        chunk = {"id": f"chatcmpl-{random.randint(10**24, 10**25)}",
                 "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_raw(resp):
    """Pass-through SSE from Google streaming."""
    for line in resp.iter_lines():
        if line:
            yield line + b"\n"
    yield b"data: [DONE]\n\n"


if __name__ == "__main__":
    print(f"Starting local proxy on :{PORT}")
    print(f"Keys: {len(API_KEYS)} ({API_KEYS[0][:10]}... etc)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)