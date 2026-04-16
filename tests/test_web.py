from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ollama_translator.web.server as srv
from ollama_translator.web.server import _cache_key, _is_srt, _split_chunks, app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, mocker):
    """Redirect all cache I/O to a temp directory for every test."""
    mocker.patch.object(srv, "CACHE_DIR", tmp_path)


@pytest.fixture
def mock_ollama(mocker):
    fake = SimpleNamespace(message=SimpleNamespace(content="Translated"))
    client = AsyncMock()
    client.chat = AsyncMock(return_value=fake)
    mocker.patch("ollama_translator.web.server.AsyncClient", return_value=client)
    return client


# ---------------------------------------------------------------------------
# _is_srt
# ---------------------------------------------------------------------------

class TestIsSrt:
    def test_detects_srt_timestamps(self):
        assert _is_srt("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

    def test_rejects_plain_text(self):
        assert not _is_srt("Hello world\nThis is plain text.")


# ---------------------------------------------------------------------------
# _split_chunks
# ---------------------------------------------------------------------------

class TestSplitChunks:
    def test_plain_text_split_by_lines(self):
        text = "\n".join(f"Line {i}" for i in range(5))
        chunks = _split_chunks(text, 2)
        assert len(chunks) == 3
        assert chunks[0] == "Line 0\nLine 1"
        assert chunks[2] == "Line 4"

    def test_srt_split_by_blocks(self):
        blocks = [f"{i}\n00:00:0{i},000 --> 00:00:0{i+1},000\nText {i}" for i in range(1, 5)]
        text = "\n\n".join(blocks)
        chunks = _split_chunks(text, 2)
        assert len(chunks) == 2
        # Each chunk contains exactly 2 SRT blocks
        assert chunks[0].count("-->") == 2

    def test_single_chunk_when_content_fits(self):
        text = "Line 1\nLine 2"
        chunks = _split_chunks(text, 10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_srt_joined_with_double_newline(self):
        blocks = [f"{i}\n00:00:0{i},000 --> 00:00:0{i+1},000\nText" for i in range(1, 3)]
        text = "\n\n".join(blocks)
        chunks = _split_chunks(text, 10)
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_inputs_produce_same_key(self):
        assert _cache_key("hello", "en", "pl") == _cache_key("hello", "en", "pl")

    def test_different_target_produces_different_key(self):
        assert _cache_key("hello", "en", "pl") != _cache_key("hello", "en", "de")

    def test_different_source_produces_different_key(self):
        assert _cache_key("hello", "en", "pl") != _cache_key("hello", "fr", "pl")

    def test_different_text_produces_different_key(self):
        assert _cache_key("hello", "en", "pl") != _cache_key("world", "en", "pl")


# ---------------------------------------------------------------------------
# POST /translate — endpoint
# ---------------------------------------------------------------------------

class TestTranslateEndpoint:
    def test_cache_hit_returns_200_with_text(self, tmp_path):
        key = _cache_key("Hello", "en", "pl")
        (tmp_path / f"{key}.txt").write_text("Cześć", encoding="utf-8")

        with TestClient(app) as client:
            response = client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        assert response.status_code == 200
        assert response.json() == {"translatedText": "Cześć"}

    def test_pending_returns_503_in_progress(self, tmp_path):
        key = _cache_key("Hello", "en", "pl")
        (tmp_path / f"{key}.pending").touch()

        with TestClient(app) as client:
            response = client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        assert response.status_code == 503
        assert "progress" in response.json()["message"].lower()

    def test_new_request_returns_503_started(self, mock_ollama):
        with TestClient(app) as client:
            response = client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        assert response.status_code == 503
        assert "started" in response.json()["message"].lower()

    def test_background_task_writes_cache_file(self, tmp_path, mock_ollama):
        mock_ollama.chat = AsyncMock(
            return_value=SimpleNamespace(message=SimpleNamespace(content="Cześć"))
        )

        with TestClient(app) as client:
            client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        key = _cache_key("Hello", "en", "pl")
        cache_file = tmp_path / f"{key}.txt"
        assert cache_file.exists()
        assert cache_file.read_text(encoding="utf-8") == "Cześć"

    def test_background_task_removes_pending_marker(self, tmp_path, mock_ollama):
        with TestClient(app) as client:
            client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        key = _cache_key("Hello", "en", "pl")
        assert not (tmp_path / f"{key}.pending").exists()

    def test_second_request_returns_200_after_cache_populated(self, tmp_path, mock_ollama):
        mock_ollama.chat = AsyncMock(
            return_value=SimpleNamespace(message=SimpleNamespace(content="Cześć"))
        )

        with TestClient(app) as client:
            client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})
            response = client.post("/translate", data={"q": "Hello", "source": "en", "target": "pl"})

        assert response.status_code == 200
        assert response.json()["translatedText"] == "Cześć"

    def test_default_language_params(self, mock_ollama):
        """source and target default to en and pl when omitted."""
        with TestClient(app) as client:
            response = client.post("/translate", data={"q": "Hello"})

        # 503 means it was accepted (not a 422 validation error)
        assert response.status_code == 503
