"""Pipeline behaviour tests with stand-in functions: no model is downloaded or run."""
import unittest
from unittest import mock

from src import pipeline


def fake_xlit(text):
    return 'कल मेरा एग्जाम है'


class PipelineTests(unittest.TestCase):
    def test_translit_only_never_touches_other_models(self):
        with mock.patch.object(pipeline, 'transliterate', fake_xlit), \
             mock.patch.object(pipeline.phrasing, 'improve_hindi') as p1, \
             mock.patch.object(pipeline.translation, 'translate_hindi_to_english') as it2:
            r = pipeline.process('kal mera exam hai')
        self.assertEqual(r.transliteration, 'कल मेरा एग्जाम है')
        p1.assert_not_called()
        it2.assert_not_called()

    def test_param1_failure_is_reported_not_hidden(self):
        with mock.patch.object(pipeline, 'transliterate', fake_xlit), \
             mock.patch.object(pipeline.phrasing, 'improve_hindi', side_effect=RuntimeError('boom')), \
             mock.patch.object(pipeline.translation, 'translate_hindi_to_english', return_value='EN') as it2:
            r = pipeline.process('kal mera exam hai', improve=True, english=True)
        self.assertIsNone(r.improved_hindi)
        self.assertIn('param1', r.errors)
        it2.assert_called_once_with('कल मेरा एग्जाम है')
        self.assertIn('Param1 failed', r.translation_source)

    def test_param1_output_feeds_translation_and_low_memory_unloads(self):
        out = {'text': 'कल मेरी परीक्षा है।', 'raw': 'कल मेरी परीक्षा है।', 'device': 'mps', 'notes': []}
        with mock.patch.object(pipeline, 'transliterate', fake_xlit), \
             mock.patch.object(pipeline.phrasing, 'improve_hindi', return_value=out), \
             mock.patch.object(pipeline.phrasing, 'unload') as unload, \
             mock.patch.object(pipeline.translation, 'translate_hindi_to_english', return_value='EN') as it2:
            r = pipeline.process('kal mera exam hai', improve=True, english=True, low_memory=True)
        it2.assert_called_once_with('कल मेरी परीक्षा है।')
        unload.assert_called_once()
        self.assertEqual(r.translation_source, 'Param1 improved Hindi')


class Param1ConfigTests(unittest.TestCase):
    def test_rope_default_becomes_none(self):
        from src.phrasing import normalise_rope_config
        cfg = type('C', (), {'rope_scaling': {'rope_type': 'default'}})()
        self.assertIsNotNone(normalise_rope_config(cfg))
        self.assertIsNone(cfg.rope_scaling)

    def test_rope_other_type_is_preserved(self):
        from src.phrasing import normalise_rope_config
        cfg = type('C', (), {'rope_scaling': {'rope_type': 'linear', 'factor': 2.0}})()
        normalise_rope_config(cfg)
        self.assertEqual(cfg.rope_scaling, {'rope_type': 'linear', 'factor': 2.0, 'type': 'linear'})

    def test_clean_output(self):
        from src.phrasing import clean_output
        self.assertEqual(clean_output('\nसुधारा हुआ वाक्य: कल मेरी परीक्षा है।\nव्याख्या...'), 'कल मेरी परीक्षा है।')
