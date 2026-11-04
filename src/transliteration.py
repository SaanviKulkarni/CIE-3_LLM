"""Official AI4Bharat IndicXlit inference only."""
import re
from functools import lru_cache

@lru_cache(maxsize=1)
def _engine():
    try:
        from ai4bharat.transliteration import XlitEngine
        return XlitEngine('hi', beam_width=4, rescore=False)
    except Exception as exc:
        raise RuntimeError('IndicXlit could not initialize. Install ai4bharat-transliteration and check its official checkpoint download. Original error: ' + str(exc)) from exc

def transliterate(text: str) -> str:
    if not text or not text.strip():
        raise ValueError('Enter a non-empty Hinglish sentence.')
    clean = re.sub(r'\s+', ' ', text.strip())
    try:
        output = _engine().translit_sentence(clean, lang_code='hi')
        if isinstance(output, dict):
            output = output.get('hi', output)
        if isinstance(output, list):
            output = output[0] if output else ''
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError(f'IndicXlit returned an unexpected response: {output!r}')
        return output.strip()
    except Exception as exc:
        raise RuntimeError(f'IndicXlit transliteration failed: {exc}') from exc
