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

## Modes

### CLI mode

Translates a single SRT file on the command line and exits.

```bash
poetry run translate <input.srt> <output.srt> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--source-lang` | `-s` | `en` | Source language (ISO 639-1 code) |
| `--target-lang` | `-t` | `pl` | Target language (ISO 639-1 code) |
| `--model` | `-m` | `gemma4:31b` | Ollama model to use |

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

### Server mode

Runs a persistent HTTP server that exposes a LibreTranslate-compatible `POST /translate` endpoint. This is useful for integrating with tools that speak the LibreTranslate API (e.g. media players, browser extensions).

```bash
poetry run serve
```

The server defaults to `0.0.0.0:5000`. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `5000` | Bind port |
| `TRANSLATE_MODEL` | `llama3.2:1b` | Ollama model to use |
| `CHUNK_SIZE` | `30` | Subtitle blocks per translation request |
| `CACHE_DIR` | `.cache` | Directory for cached results |

**Endpoint:** `POST /translate`

| Form field | Default | Description |
|------------|---------|-------------|
| `q` | — | Text or SRT content to translate |
| `source` | `en` | Source language (ISO 639-1 code) |
| `target` | `pl` | Target language (ISO 639-1 code) |

Translations run in the background. The first request for a given input returns HTTP 503 and starts the job; subsequent requests for the same input return 503 until the job finishes, then return 200 with `{"translatedText": "..."}`. Results are cached on disk so repeated requests are instant.

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

## Docker

Build the image:

```bash
docker build -t ollama-translator .
```

Run the server:

```bash
docker run -p 5000:5000 -e OLLAMA_HOST=http://192.168.1.100:11434 ollama-translator
```

Override any server setting with `-e`:

```bash
docker run -p 8080:8080 \
  -e OLLAMA_HOST=http://192.168.1.100:11434 \
  -e SERVER_PORT=8080 \
  -e TRANSLATE_MODEL=llama3.3:70b \
  ollama-translator
```

> Never bind-mount a `.env` file into the container — pass secrets as `-e` flags or via Docker secrets.

## Running tests

```bash
poetry run pytest tests/ -v
```
