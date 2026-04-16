import sys

from langcodes import Language, LanguageTagError


def resolve_lang(code: str) -> str:
    try:
        return Language.get(code).display_name()
    except (LanguageTagError, ValueError):
        print(f"Error: unknown language code '{code}'.", file=sys.stderr)
        sys.exit(1)
