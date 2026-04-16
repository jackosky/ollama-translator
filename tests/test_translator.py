import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import srt

import translator
from translator import translate_batch, translate_file


def make_subtitle(index: int, content: str) -> srt.Subtitle:
    start = datetime.timedelta(seconds=index)
    end = datetime.timedelta(seconds=index + 1)
    return srt.Subtitle(index=index, start=start, end=end, content=content)


def fake_response(text: str):
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(message=msg)


# ---------------------------------------------------------------------------
# translate_batch
# ---------------------------------------------------------------------------

class TestTranslateBatch:
    def test_returns_translated_text(self, mocker):
        subs = [make_subtitle(1, "Hello world")]
        mocker.patch("translator.ollama.Client").return_value.chat.return_value = (
            fake_response("1. Witaj świecie")
        )
        result = translate_batch(subs, "test-model")
        assert result == ["Witaj świecie"]

    def test_multiline_subtitle_separator_preserved(self, mocker):
        subs = [make_subtitle(1, "That was crazy, huh?\n- Tell me about it.")]
        mocker.patch("translator.ollama.Client").return_value.chat.return_value = (
            fake_response("1. To było szalone, prawda? | - No właśnie.")
        )
        result = translate_batch(subs, "test-model")
        assert result == ["To było szalone, prawda?\n- No właśnie."]

    def test_missing_translation_falls_back_to_original(self, mocker):
        subs = [make_subtitle(1, "Hello"), make_subtitle(2, "World")]
        # Model only returns line 1
        mocker.patch("translator.ollama.Client").return_value.chat.return_value = (
            fake_response("1. Cześć")
        )
        result = translate_batch(subs, "test-model")
        assert result[0] == "Cześć"
        assert result[1] == "World"  # original kept

    def test_batch_of_multiple_subtitles(self, mocker):
        subs = [make_subtitle(i, f"Line {i}") for i in range(1, 4)]
        mocker.patch("translator.ollama.Client").return_value.chat.return_value = (
            fake_response("1. Linia 1\n2. Linia 2\n3. Linia 3")
        )
        result = translate_batch(subs, "test-model")
        assert result == ["Linia 1", "Linia 2", "Linia 3"]

    def test_uses_correct_model_and_host(self, mocker):
        subs = [make_subtitle(1, "Hi")]
        mock_client_cls = mocker.patch("translator.ollama.Client")
        mock_client_cls.return_value.chat.return_value = fake_response("1. Cześć")

        translate_batch(subs, "llama3:8b")

        mock_client_cls.assert_called_once_with(host=translator.OLLAMA_HOST)
        mock_client_cls.return_value.chat.assert_called_once()
        call_kwargs = mock_client_cls.return_value.chat.call_args
        assert call_kwargs.kwargs["model"] == "llama3:8b"


# ---------------------------------------------------------------------------
# translate_file
# ---------------------------------------------------------------------------

SRT_CONTENT = """\
1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""


class TestTranslateFile:
    def test_writes_translated_output(self, tmp_path, mocker):
        input_file = tmp_path / "input.srt"
        output_file = tmp_path / "output.srt"
        input_file.write_text(SRT_CONTENT, encoding="utf-8")

        mocker.patch("translator.ollama.Client").return_value.chat.side_effect = [
            fake_response("1. Cześć\n2. Świecie"),
        ]

        translate_file(input_file, output_file, "test-model")

        result = list(srt.parse(output_file.read_text(encoding="utf-8")))
        assert result[0].content == "Cześć"
        assert result[1].content == "Świecie"

    def test_exits_on_empty_srt(self, tmp_path, mocker):
        input_file = tmp_path / "empty.srt"
        input_file.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit):
            translate_file(input_file, tmp_path / "out.srt", "test-model")

    def test_missing_input_file_exits_via_main(self, tmp_path, mocker):
        mocker.patch(
            "sys.argv",
            ["translate", str(tmp_path / "nonexistent.srt"), str(tmp_path / "out.srt")],
        )
        with pytest.raises(SystemExit):
            translator.main()
