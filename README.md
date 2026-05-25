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
| Google | `POST /v1beta/models/{model}:generateContent` |
| Streaming | `POST /v1/chat/completions?stream=true` / `/v1beta/models/{model}:streamGenerateContent` |

## 路由策略

- 轮询 (round-robin)，按请求轮换 Key
- 429 立即跳过下一个 Key，不重试
- timeout 15s

## 认证

`Authorization: Bearer <AUTH_TOKEN>`

## 日志

`GET /logs?limit=50`