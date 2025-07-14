from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict, Dict, List
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph, END
import os
import asyncio
import httpx

load_dotenv()

app = FastAPI()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/qwen3-30b-a3b"

# Define input model
class TranslationRequest(BaseModel):
    language: List[str]
    content: str

# Define state for LangGraph
class TranslationState(TypedDict):
    content: str
    languages: List[str]
    translations: Dict[str, str]
    errors: List[str]

# Translation prompts
PROMPTS = {
    "jp": """以下のテキストを正確に日本語に翻訳してください。内容を追加・変更せず、100%正しい文法とスペルで翻訳してください。説明や余計な情報は不要です。翻訳結果のみを出力してください。
    
    {{content}}""",
    "en": """Dịch chính xác đoạn văn sau sang tiếng Anh. Đảm bảo đúng chính tả 100%, không thêm thắt hay sáng tạo nội dung, không kèm theo lời giải thích hoặc ghi chú. Chỉ xuất kết quả là nội dung đã được dịch:
    
    {{content}}"""
}


import httpx

async def call_fireworks_chat(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "fireworks-playground": "true",  # optional
    }
    payload = {
        "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 1,
        "top_k": 40,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": False,
        "echo": False,
        "logprobs": True
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            raise Exception(f"Fireworks error {response.status_code}: {response.text}")
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


import httpx

async def call_fireworks_chat(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "fireworks-playground": "true",  # optional
    }
    payload = {
        "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 1,
        "top_k": 40,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": False,
        "echo": False,
        "logprobs": True
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            raise Exception(f"Fireworks error {response.status_code}: {response.text}")
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

@traceable(name="translate_node")
async def translate_node(state: TranslationState) -> TranslationState:
    translations = {}
    errors = []

    async def translate_single_language(content: str, lang: str):
        try:
            prompt = PROMPTS[lang].replace("{{content}}", content)
            messages = [{"role": "user", "content": prompt}]
            result = await call_fireworks_chat(messages)
            return lang, result
        except Exception as e:
            return lang, str(e)

    tasks = [translate_single_language(state["content"], lang) for lang in state["languages"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            errors.append(f"Unexpected error: {str(result)}")
        else:
            lang, output = result
            if "error" in output.lower():
                errors.append(f"Translation failed for {lang}: {output}")
            else:
                translations[lang] = output

    return {
        "content": state["content"],
        "languages": state["languages"],
        "translations": translations,
        "errors": errors
    }

# Build LangGraph workflow
def build_workflow():
    workflow = StateGraph(TranslationState)
    workflow.add_node("translate", translate_node)
    workflow.set_entry_point("translate")
    workflow.add_edge("translate", END)
    return workflow.compile()

graph = build_workflow()

@traceable(name="translate_text_endpoint")
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        valid_languages = ["jp", "en"]
        for lang in request.language:
            if lang not in valid_languages:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

        initial_state = {
            "content": request.content,
            "languages": request.language,
            "translations": {},
            "errors": []
        }

        result = await graph.ainvoke(initial_state)

        if result["errors"]:
            raise HTTPException(status_code=500, detail="; ".join(result["errors"]))

        return JSONResponse(content={"translations": result["translations"]})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
