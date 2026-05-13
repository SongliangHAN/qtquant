import os
import json
import httpx

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


# uvicorn provider:app --host 0.0.0.0 --port 3000

app = FastAPI()

# =========================================================
# 配置
# =========================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 也可以直接写死
DEEPSEEK_API_KEY = "sk-2a37e06687ef4441ba90672c5fe9a62a"

DEEPSEEK_BASE = "https://api.deepseek.com/anthropic"

TARGET_MODEL = "deepseek-v4-pro"


# =========================================================
# 首页
# =========================================================

@app.get("/")
async def home():

    return {
        "status": "ok",
        "provider": "deepseek-anthropic",
        "model": TARGET_MODEL
    }


# =========================================================
# Claude Code models probe
# =========================================================

@app.get("/v1/models")
async def models():

    return {
        "object": "list",
        "data": [
            {
                "id": TARGET_MODEL,
                "object": "model",
                "owned_by": "deepseek"
            }
        ]
    }


# =========================================================
# Claude Code token count
# =========================================================

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):

    try:
        data = await request.json()
    except:
        data = {}

    txt = json.dumps(data, ensure_ascii=False)

    return {
        "input_tokens": max(1, len(txt) // 4)
    }


# =========================================================
# 通用代理（禁用 streaming）
# =========================================================

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy(path: str, request: Request):

    try:

        # =====================================================
        # DeepSeek endpoint
        # =====================================================

        target_url = f"{DEEPSEEK_BASE}/{path}"

        # =====================================================
        # headers
        # =====================================================

        headers = dict(request.headers)

        # DeepSeek 使用 x-api-key
        headers["x-api-key"] = DEEPSEEK_API_KEY

        # 删除不该转发的 header
        for h in [
            "host",
            "content-length",
            "connection",
            "accept-encoding"
        ]:
            headers.pop(h, None)

        # =====================================================
        # body
        # =====================================================

        body_json = None
        raw_body = None

        if request.method != "GET":

            content_type = request.headers.get(
                "content-type",
                ""
            )

            # JSON body
            if "application/json" in content_type:

                try:

                    body_json = await request.json()

                    # 强制替换模型
                    if (
                        isinstance(body_json, dict)
                        and "model" in body_json
                    ):
                        body_json["model"] = TARGET_MODEL

                    # =================================================
                    # 禁用 streaming
                    # =================================================

                    body_json["stream"] = False

                except Exception as e:

                    print("JSON parse error:", e)

                    raw_body = await request.body()

            else:

                raw_body = await request.body()

        # =====================================================
        # 普通请求
        # =====================================================

        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            r = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                json=body_json,
                content=raw_body
            )

        # =====================================================
        # 返回
        # =====================================================

        content_type = r.headers.get(
            "content-type",
            "application/json"
        )

        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=content_type
        )

    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(e)
                }
            }
        )


# =========================================================
# 启动
# =========================================================

if __name__ == "__main__":

    import uvicorn

    print("======================================")
    print("Claude Code DeepSeek Provider")
    print("======================================")
    print("Base URL : http://localhost:3000")
    print("Target   :", TARGET_MODEL)
    print("Streaming: DISABLED")
    print("======================================")

    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not set")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000
    )