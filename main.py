import os, asyncio, json
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

# Render의 환경 변수로 API 키를 안전하게 불러옵니다.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI(title="ZINO-GE Router", version="1.2")

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "ZINO-GE is alive!"}

class RouteIn(BaseModel):
    user_input: str
    intent: str | None = None

class RouteOut(BaseModel):
    report_md: str
    meta: dict

async def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type":"application/json"}
    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

async def call_claude(prompt: str):
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model":"claude-3-opus-20240229","max_tokens":4000, "messages":[{"role":"user","content": prompt}]}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return "".join([b.get("text","") for b in data["content"]])

async def call_gpt_orchestrate(context: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system_prompt = "You are a master synthesizer. Based on the analyses from two AI agents (Gemini and Claude), create a comprehensive, final report in Korean. Synthesize the key findings, identify agreements and disagreements, and provide a concluding summary."
    user_prompt = f"Please synthesize the following analyses:\n\n{context}"
    payload = {"model":"gpt-4o","messages":[{"role":"system","content": system_prompt},{"role":"user","content": user_prompt}],"temperature":0.3}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core"])
async def route(payload: RouteIn):
    if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY]):
        return {"report_md": "Error: One or more API keys are not set.", "meta": {}}
    tasks = [call_gemini(payload.user_input), call_claude(payload.user_input)]
    g_res, c_res = await asyncio.gather(*tasks, return_exceptions=True)
    results = {"gemini_result": g_res if not isinstance(g_res, Exception) else f"Error: {str(g_res)}", "claude_result": c_res if not isinstance(c_res, Exception) else f"Error: {str(c_res)}"}
    context = json.dumps(results, ensure_ascii=False, indent=2)
    final_report = await call_gpt_orchestrate(context)
    return RouteOut(report_md=final_report, meta=results)
