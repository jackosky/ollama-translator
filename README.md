# ollama-translator

Translate SRT subtitle files between any languages using a locally running [Ollama](https://ollama.com) model.

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org)
- [Ollama](https://ollama.com) running locally or on your network

## Installation

```bash
git clone <repo>
cd ollama-translator
poetry install
```

Copy the example environment file and set your Ollama address:

```bash
cp .env.example .env
```

`.env`:
```
OLLAMA_HOST=http://192.168.1.100:11434
```

If `OLLAMA_HOST` is not set, it defaults to `http://localhost:11434`.

## Usage

```bash
poetry run translate <input.srt> <output.srt> [options]
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--source-lang` | `-s` | `en` | Source language (ISO 639-1 code) |
| `--target-lang` | `-t` | `pl` | Target language (ISO 639-1 code) |
| `--model` | `-m` | `gemma4:31b` | Ollama model to use |

### Examples

```bash
# English → Polish (defaults)
poetry run translate film.srt film.pl.srt

# English → German
poetry run translate film.srt film.de.srt -t de

# French → English
poetry run translate film.srt film.en.srt -s fr -t en

# Use a different model
poetry run translate film.srt film.pl.srt -m llama3.3:70b
```

## Language codes

Any [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) two-letter code is accepted. Common examples:

| Code | Language |
|------|----------|
| `en` | English |
| `pl` | Polish |
| `de` | German |
| `fr` | French |
| `es` | Spanish |
| `it` | Italian |
| `pt` | Portuguese |
| `ru` | Russian |
| `uk` | Ukrainian |
| `ja` | Japanese |
| `zh` | Chinese |
| `ko` | Korean |

## How it works

The script parses the input `.srt` file and sends subtitles to Ollama in batches of 10. Each batch is translated in a single request, which preserves conversational context across neighbouring lines. Multi-line subtitle entries are flattened with a ` | ` separator before translation and restored afterward, so the SRT structure is never broken.

## Running tests

```bash
poetry run pytest tests/ -v
```
