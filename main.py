import os
import asyncio
import hashlib
import logging
import datetime
from typing import Dict, List, Tuple

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# -----------------------------
# Load env
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY is required")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
RATE_LIMIT = 300
WINDOW_SIZE = 60  # 60s = 1 phút

redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Translation prompts
# -----------------------------
PROMPTS = {
    "jp": """以下のテキストを日本語に正確に翻訳してください。内容を追加または変更せず、完全に正しい文法と綴りで翻訳してください。説明や余計な情報は不要で、翻訳した内容のみを出力してください。

{{content}}""",
    "en": """Translate the text below into English. Only output the translated content. Do not explain or add anything.\n\n{{content}}"""
}

# -----------------------------
# Models
# -----------------------------
class TranslationRequest(BaseModel):
    language: List[str]
    content: str

# -----------------------------
# Rate limit (Redis counter)
# -----------------------------
async def acquire_token():
    """Check and consume token in Redis bucket (300/min)"""
    key = f"translation:rate_limit:{int(datetime.datetime.utcnow().timestamp() // WINDOW_SIZE)}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, WINDOW_SIZE)
    if count > RATE_LIMIT:
        return False
    return True

# -----------------------------
# Cache helpers
# -----------------------------
def get_cache_key(content: str, lang: str) -> str:
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"translation:{content_hash}:{lang}"

async def get_cached_translation(content: str, lang: str):
    key = get_cache_key(content, lang)
    return await redis_client.get(key)

async def set_cached_translation(content: str, lang: str, translation: str):
    key = get_cache_key(content, lang)
    await redis_client.setex(key, 86400, translation)  # TTL 24h

# -----------------------------
# OpenAI call per language
# -----------------------------
async def translate_single_language(content: str, lang: str) -> Tuple[str, str]:
    # Check cache
    cached = await get_cached_translation(content, lang)
    if cached:
        logger.info(f"Cache hit for {lang}")
        return lang, cached

    # Check rate limit
    allowed = await acquire_token()
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (300 req/min)")

    logger.info(f"Translating to {lang}...")
    prompt = PROMPTS[lang].replace("{{content}}", content)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024,
            temperature=0.6,
        )
        result = response.choices[0].message.content.strip()
        await set_cached_translation(content, lang, result)
        return lang, result
    except Exception as e:
        logger.error(f"OpenAI translation error for {lang}: {e}")
        return lang, f"Error: {str(e)}"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    logger.info(f"Received request for {request.language}")

    # validate languages
    valid_languages = list(PROMPTS.keys())
    for lang in request.language:
        if lang not in valid_languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    # run concurrently
    tasks = [translate_single_language(request.content, lang) for lang in request.language]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    translations: Dict[str, str] = {}
    errors: List[str] = []

    for lang, res in zip(request.language, results):
        if isinstance(res, Exception):
            errors.append(f"{lang}: {res}")
        else:
            l, output = res
            if output.lower().startswith("error"):
                errors.append(f"{l}: {output}")
            else:
                translations[l] = output

    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    return JSONResponse(content={"translations": translations})
