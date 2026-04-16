import pytest

from ollama_translator.core.languages import resolve_lang


class TestResolveLang:
    def test_valid_code_returns_name(self):
        assert resolve_lang("en") == "English"
        assert resolve_lang("pl") == "Polish"
        assert resolve_lang("de") == "German"

    def test_invalid_code_exits(self):
        with pytest.raises(SystemExit):
            resolve_lang("xx_INVALID_CODE_zz")
