import os
import asyncio
import hashlib
import logging
from typing import TypedDict, Dict, List

import httpx
import redis
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langsmith import traceable
from langgraph.graph import StateGraph, END

load_dotenv()

# -----------------------------
# Config & Redis
# -----------------------------
Local_API_KEY = os.getenv("Local_API_KEY")
LLM_URL = os.getenv("LLM_URL", "")
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Models
# -----------------------------
class TranslationRequest(BaseModel):
    language: List[str]
    content: str

class TranslationState(TypedDict):
    content: str
    languages: List[str]
    translations: Dict[str, str]
    errors: List[str]

# -----------------------------
# Translation prompts
# -----------------------------
PROMPTS = {
    "jp": """以下のテキストを日本語に正確に翻訳してください。内容を追加または変更せず、完全に正しい文法と綴りで翻訳してください。説明や余計な情報は不要で、翻訳した内容のみを出力してください。

{{content}}""",
    "en": """Translate the text below into English. Only output the translated content. Do not explain or add anything.\n\n{{content}}"""
}

# -----------------------------
# Helper functions
# -----------------------------
def get_cache_key(content: str, lang: str) -> str:
    content_hash = hashlib.md5(content.encode()).hexdigest()
    key = f"translation:{content_hash}:{lang}"
    logger.debug(f"Generated cache key: {key}")
    return key

async def call_Local_chat(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Local-playground": "true",
    }
    payload = {
        "model": "unsloth/Qwen2.5-7B-Instruct",
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

    try:
        logger.info("Calling Local LLM...")
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post(LLM_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.info("Received response from Local LLM")
            return content
    except Exception as e:
        logger.warning(f"Local LLM failed: {e}, fallback to OpenAI")
        client = AsyncOpenAI(api_key=OPENAI_API_KEY) 
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=4096,
                temperature=0.6,
                top_p=1,
                n=1,
                presence_penalty=0,
                frequency_penalty=0,
            )
            content = response.choices[0].message.content.strip()
            logger.info("Received response from OpenAI fallback")
            return content
        except Exception as openai_error:
            logger.error(f"OpenAI fallback failed: {openai_error}")
            raise Exception(f"OpenAI error: {str(openai_error)}")

async def translate_single_language(content: str, lang: str) -> (str, str):
    cache_key = get_cache_key(content, lang)
    cached = redis_client.get(cache_key)
    if cached:
        logger.info(f"Cache hit for language '{lang}'")
        return lang, cached

    logger.info(f"Cache miss for language '{lang}', translating...")
    prompt = PROMPTS[lang].replace("{{content}}", content)
    messages = [{"role": "user", "content": prompt}]
    try:
        result = await call_Local_chat(messages)
        redis_client.setex(cache_key, 86400, result)
        logger.info(f"Stored translation in cache for '{lang}'")
        return lang, result
    except Exception as e:
        logger.error(f"Translation failed for '{lang}': {e}")
        return lang, str(e)

# -----------------------------
# Translation workflow
# -----------------------------
@traceable(name="translate_node")
async def translate_node(state: TranslationState) -> TranslationState:
    translations = {}
    errors = []

    logger.info(f"Translating content into languages: {state['languages']}")
    tasks = [translate_single_language(state["content"], lang) for lang in state["languages"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Unexpected error during translation: {res}")
            errors.append(f"Unexpected error: {res}")
        else:
            lang, output = res
            if "error" in output.lower():
                logger.error(f"Translation error for '{lang}': {output}")
                errors.append(f"Translation failed for {lang}: {output}")
            else:
                translations[lang] = output
                logger.info(f"Translation successful for '{lang}'")

    return {
        "content": state["content"],
        "languages": state["languages"],
        "translations": translations,
        "errors": errors
    }

def build_workflow():
    workflow = StateGraph(TranslationState)
    workflow.add_node("translate", translate_node)
    workflow.set_entry_point("translate")
    workflow.add_edge("translate", END)
    return workflow.compile()

graph = build_workflow()

# -----------------------------
# FastAPI endpoints
# -----------------------------
app = FastAPI()

@traceable(name="translate_text_endpoint")
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    logger.info(f"Received translation request: languages={request.language}")
    valid_languages = ["jp", "en"]
    for lang in request.language:
        if lang not in valid_languages:
            logger.warning(f"Unsupported language requested: {lang}")
            raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    initial_state: TranslationState = {
        "content": request.content,
        "languages": request.language,
        "translations": {},
        "errors": []
    }

    result = await graph.ainvoke(initial_state)

    if result["errors"]:
        logger.error(f"Errors during translation: {result['errors']}")
        raise HTTPException(status_code=500, detail="; ".join(result["errors"]))

    logger.info(f"Translation completed successfully for languages: {list(result['translations'].keys())}")
    return JSONResponse(content={"translations": result["translations"]})
