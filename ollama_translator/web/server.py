import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Annotated

import aiofiles
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Form
from fastapi.responses import JSONResponse
from ollama import AsyncClient

from ollama_translator.core.languages import resolve_lang
from ollama_translator.core.translation import OLLAMA_HOST, build_prompts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRANSLATE_MODEL: str = os.getenv("TRANSLATE_MODEL", "llama3.2:1b")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "30"))
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", ".cache"))
HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
PORT: int = int(os.getenv("SERVER_PORT", "5000"))

CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LibreTranslate-compatible Ollama proxy")

# ---------------------------------------------------------------------------
# System prompt for the web server — stricter than the CLI prompt since the
# input can be any text, not just nicely-formatted SRT from a known source.
# ---------------------------------------------------------------------------

WEB_SYSTEM_PROMPT = """\
You are a professional translator.
Translate the text provided by the user from {source} to {target}.

Strict rules — follow them exactly:
1. If the input contains SRT subtitle format (lines matching "HH:MM:SS,mmm --> HH:MM:SS,mmm"),
   preserve ALL subtitle indices, timestamps, and blank lines exactly as they are.
   Translate ONLY the dialogue/caption lines.
2. Do NOT add any preamble, commentary, or closing remarks.
   Never write phrases like "Here is the translation:", "Translated text:", or similar.
3. Return ONLY the translated content. Nothing before it, nothing after it.
4. Preserve all original line breaks, blank lines, and whitespace structure.
5. Do NOT translate proper names inside angle brackets or HTML/XML tags.\
"""

SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_srt(text: str) -> bool:
    return bool(SRT_TIMESTAMP_RE.search(text))


def _split_chunks(text: str, size: int) -> list[str]:
    if _is_srt(text):
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
        return ["\n\n".join(blocks[i : i + size]) for i in range(0, len(blocks), size)]
    lines = text.splitlines()
    return ["\n".join(lines[i : i + size]) for i in range(0, len(lines), size)]


def _cache_key(q: str, source: str, target: str) -> str:
    return hashlib.sha256(f"{source}:{target}:{q}".encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.txt"


def _pending_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pending"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


async def _translate(q: str, source_name: str, target_name: str) -> str:
    client = AsyncClient(host=OLLAMA_HOST)
    system = WEB_SYSTEM_PROMPT.format(source=source_name, target=target_name)
    chunks = _split_chunks(q, CHUNK_SIZE)
    srt_mode = _is_srt(q)

    log.info(
        "Translating: %s→%s | %d chunk(s) | SRT=%s | model=%s",
        source_name, target_name, len(chunks), srt_mode, TRANSLATE_MODEL,
    )

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        log.info("  [%d/%d] %d chars", i, len(chunks), len(chunk))
        response = await client.chat(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
        )
        parts.append(response.message.content.strip())

    separator = "\n\n" if srt_mode else "\n"
    return separator.join(parts)


async def _translate_and_cache(q: str, source_name: str, target_name: str, key: str) -> None:
    try:
        result = await _translate(q, source_name, target_name)
        async with aiofiles.open(_cache_path(key), "w", encoding="utf-8") as f:
            await f.write(result)
        log.info("Cached: %s...", key[:12])
    except Exception as exc:
        log.error("Translation failed (%s...): %s", key[:12], exc)
    finally:
        _pending_path(key).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/translate")
async def translate(
    background_tasks: BackgroundTasks,
    q: Annotated[str, Form()],
    source: Annotated[str, Form()] = "en",
    target: Annotated[str, Form()] = "pl",
) -> JSONResponse:
    source_name = resolve_lang(source)
    target_name = resolve_lang(target)

    key = _cache_key(q, source, target)
    cached = _cache_path(key)
    pending = _pending_path(key)

    # Cache hit — return immediately
    if cached.exists():
        log.info("Cache hit  %s...", key[:12])
        async with aiofiles.open(cached, "r", encoding="utf-8") as f:
            content = await f.read()
        return JSONResponse({"translatedText": content})

    # Already in progress — don't start a duplicate job
    if pending.exists():
        log.info("In progress %s...", key[:12])
        return JSONResponse(
            status_code=503,
            content={"message": "Translation in progress. Please retry in a moment."},
        )

    # New request — queue background translation
    log.info("Cache miss %s... — queuing job", key[:12])
    pending.touch()
    background_tasks.add_task(_translate_and_cache, q, source_name, target_name, key)
    return JSONResponse(
        status_code=503,
        content={"message": "Translation started. Please retry in a moment."},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve() -> None:
    uvicorn.run(
        "ollama_translator.web.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
