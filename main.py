import datetime
import os
import asyncio
import hashlib
import logging
from typing import TypedDict, Dict, List

import redis
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langsmith import traceable

load_dotenv()

# -----------------------------
# Config & Redis
# -----------------------------
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY is required")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# -----------------------------
# Rate Limiting for OpenAI (300 req/min = 5 req/sec)
# We'll enforce min 200ms between requests → max 5 req/sec
# -----------------------------
last_call_time = 0.0
rate_limit_lock = asyncio.Lock()
MIN_INTERVAL = 0.2  # 200ms → 5 requests per second (safe under 300/min)

async def rate_limited_openai_call(messages: List[Dict[str, str]]) -> str:
    global last_call_time
    async with rate_limit_lock:
        now = asyncio.get_event_loop().time()
        elapsed = now - last_call_time
        if elapsed < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - elapsed
            await asyncio.sleep(sleep_time)
            now = asyncio.get_event_loop().time()
        last_call_time = now

        # Now safe to call OpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("Calling OpenAI (rate-limited)...")
        start_time = datetime.datetime.now()
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
        logger.info(f"OpenAI response received in {datetime.datetime.now() - start_time}")
        return content

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
# Translation prompts (unchanged)
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
        result = await rate_limited_openai_call(messages)
        redis_client.setex(cache_key, 86400, result)  # cache 24h
        logger.info(f"Stored translation in cache for '{lang}'")
        return lang, result
    except Exception as e:
        logger.error(f"Translation failed for '{lang}': {e}")
        return lang, str(e)

# -----------------------------
# FastAPI endpoint (no LangGraph needed for single step)
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

    # Run all translations concurrently (but rate-limited internally)
    tasks = [translate_single_language(request.content, lang) for lang in request.language]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    translations = {}
    errors = []

    for res in results:
        if isinstance(res, Exception):
            errors.append(f"Unexpected error: {res}")
        else:
            lang, output = res
            if "error" in output.lower():
                errors.append(f"Translation failed for {lang}: {output}")
            else:
                translations[lang] = output

    if errors:
        logger.error(f"Errors during translation: {errors}")
        raise HTTPException(status_code=500, detail="; ".join(errors))

    logger.info(f"Translation completed for: {list(translations.keys())}")
    return JSONResponse(content={"translations": translations})