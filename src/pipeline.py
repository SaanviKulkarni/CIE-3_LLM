"""Composes the three approved models lazily and sequentially, without hidden substitutions.

A failing optional stage (Param1 or IndicTrans2) is recorded in Result.errors; earlier
stage outputs are still returned. Only a transliteration failure aborts the run.
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from . import phrasing, translation
from .transliteration import transliterate

MODES = {
    'Transliteration only (IndicXlit)': dict(improve=False, english=False),
    'Transliteration + English (IndicXlit → IndicTrans2)': dict(improve=False, english=True),
    'Full pipeline (IndicXlit → Param1 → IndicTrans2)': dict(improve=True, english=True),
}


@dataclass
class Result:
    original: str
    transliteration: Optional[str] = None
    improved_hindi: Optional[str] = None
    improved_hindi_raw: Optional[str] = None
    english_translation: Optional[str] = None
    translation_source: Optional[str] = None   # which Hindi text IndicTrans2 received
    errors: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    timings_s: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def process(text: str, improve: bool = False, english: bool = False, low_memory: bool = False,
            on_stage: Optional[Callable[[str], None]] = None) -> Result:
    say = on_stage or (lambda _msg: None)
    if not text or not text.strip():
        raise ValueError('Please enter a Hinglish sentence.')
    if len(text) > 300:
        raise ValueError('Keep demo inputs under 300 characters.')
    result = Result(original=text.strip())

    say('IndicXlit: transliterating to Devanagari…')
    t = time.perf_counter()
    result.transliteration = transliterate(text)          # raises: nothing to continue with
    result.timings_s['indicxlit'] = round(time.perf_counter() - t, 2)

    if improve:
        say('BharatGen Param1: ' + ('generating' if phrasing.is_loaded() else 'loading model (first use is slow), then generating') + '…')
        t = time.perf_counter()
        try:
            out = phrasing.improve_hindi(result.transliteration)
            result.improved_hindi, result.improved_hindi_raw = out['text'], out['raw']
            result.notes += [f"Param1 ran on {out['device']}"] + out['notes']
        except Exception as exc:
            result.errors['param1'] = str(exc)
        result.timings_s['param1'] = round(time.perf_counter() - t, 2)
        if low_memory:
            phrasing.unload()                              # sequential: free ~6 GB before IndicTrans2
            result.notes.append('Param1 unloaded after use (low-memory mode)')

    if english:
        if result.improved_hindi:
            source_text, result.translation_source = result.improved_hindi, 'Param1 improved Hindi'
        else:
            source_text = result.transliteration
            result.translation_source = ('IndicXlit output (Param1 failed — see error)'
                                         if improve else 'IndicXlit output')
        say('IndicTrans2: ' + ('translating' if translation.is_loaded() else 'loading model, then translating') + '…')
        t = time.perf_counter()
        try:
            result.english_translation = translation.translate_hindi_to_english(source_text)
        except Exception as exc:
            result.errors['indictrans2'] = str(exc)
        result.timings_s['indictrans2'] = round(time.perf_counter() - t, 2)
    return result


def loaded_models() -> list:
    from .transliteration import _engine
    names = []
    if _engine.cache_info().currsize:
        names.append('IndicXlit')
    if phrasing.is_loaded():
        names.append('Param1')
    if translation.is_loaded():
        names.append('IndicTrans2')
    return names
