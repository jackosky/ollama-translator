import argparse
import os
import sys
from pathlib import Path

import ollama
import srt
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "gemma4:31b"
BATCH_SIZE = 10
SYSTEM_PROMPT = (
    "You are an expert in audiovisual translation. "
    "You translate movie subtitles from English to Polish, "
    "ensuring natural-sounding dialogue while preserving the SRT format."
)


def translate_batch(subtitles: list[srt.Subtitle], model: str) -> list[str]:
    # Flatten multi-line subtitle content using " | " as an intra-line separator
    # so every subtitle occupies exactly one numbered line in the prompt.
    flat = [sub.content.strip().replace("\n", " | ") for sub in subtitles]

    numbered_lines = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(flat))
    prompt = (
        "Translate the following movie subtitles from English to Polish. "
        "Each subtitle is on one line prefixed with its number. "
        "Multi-line subtitles are joined with \" | \" — preserve that separator in your translation. "
        "Return ONLY the translated lines in the same format (number. text), "
        "with no additional comments.\n\n"
        f"{numbered_lines}"
    )
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.message.content.strip()

    translations: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ". " in line:
            prefix, _, text = line.partition(". ")
            if prefix.isdigit():
                # Restore newlines from the " | " separator
                translations[int(prefix)] = text.strip().replace(" | ", "\n")

    return [translations.get(i + 1, subtitles[i].content) for i in range(len(subtitles))]


def translate_file(input_path: Path, output_path: Path, model: str) -> None:
    raw = input_path.read_text(encoding="utf-8-sig")
    subtitles = list(srt.parse(raw))

    if not subtitles:
        print("No subtitles found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subtitles)} subtitles. Translating in batches of {BATCH_SIZE} using model '{model}'...")

    batches = [subtitles[i : i + BATCH_SIZE] for i in range(0, len(subtitles), BATCH_SIZE)]

    with tqdm(total=len(subtitles), unit="sub") as progress:
        for batch in batches:
            translated_texts = translate_batch(batch, model)
            for sub, text in zip(batch, translated_texts):
                sub.content = text
            progress.update(len(batch))

    output_path.write_text(srt.compose(subtitles), encoding="utf-8")
    print(f"Done! Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRT subtitle translator (EN→PL) using a local Ollama model."
    )
    parser.add_argument("input", type=Path, help="Input .srt file")
    parser.add_argument("output", type=Path, help="Output .srt file")
    parser.add_argument(
        "--model", "-m",
        default=MODEL,
        help=f"Ollama model to use (default: {MODEL})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    translate_file(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
