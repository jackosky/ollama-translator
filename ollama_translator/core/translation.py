import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL: str = "gemma4:31b"
BATCH_SIZE: int = 10


def build_prompts(source_lang: str, target_lang: str) -> str:
    return (
        "You are an expert in audiovisual translation. "
        f"You translate movie subtitles from {source_lang} into {target_lang}, "
        "ensuring natural-sounding dialogue while preserving the SRT format."
    )
