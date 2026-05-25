# Local LLM Proxy

本地 API Key 轮询代理，支持 OpenAI 格式 + Google 原生格式。

## 启动

```bash
# 安装依赖
cd local
uv sync

# 设置环境变量
export GOOGLE_API_KEYS="key1,key2,key3"
export AUTH_TOKEN="your-token"
export CF_API_TOKEN="your-cf-token"   # 可选，走 Cloudflare AI Gateway

# 运行
uv run python server.py
```

## 接口

| 格式 | 端点 |
|------|------|
| OpenAI | `POST /v1/chat/completions` |
| OpenAI 兼容别名 | `POST /v1/openai/chat/completions` / `POST /v1beta/openai/chat/completions` |
| Google | `POST /v1beta/models/{model}:generateContent` / `POST /v1/models/{model}:generateContent` |
| Streaming | `POST /v1/chat/completions` (body: `stream=true`) / `/v1beta/models/{model}:streamGenerateContent` / `/v1/models/{model}:streamGenerateContent` |

### OpenAI 转 Gemini 兼容增强

- 支持 `systemInstruction` 映射（system 消息不再拼接到首条 user 消息）
- 支持 `temperature` / `top_p` / `top_k` / `max_tokens` / `stop` / `n` → `generationConfig`
- 支持 `response_format` JSON 输出映射
- 支持 `tools` / `tool_choice` 与 Gemini function calling 转换
- 支持多模态 `image_url`（data URL 与 URI）

## 路由策略

- 轮询 (round-robin)，按请求轮换 Key
- 429 立即跳过下一个 Key，不重试
- timeout 15s

## 认证

`Authorization: Bearer <AUTH_TOKEN>`

## 日志

`GET /logs?limit=50`