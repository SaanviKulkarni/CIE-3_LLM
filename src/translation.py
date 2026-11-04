"""AI4Bharat IndicTrans2 Hindi-to-English using the official IndicProcessor workflow."""
import gc
import os

MODEL_ID = 'ai4bharat/indictrans2-indic-en-dist-200M'
_state = {'tokenizer': None, 'model': None, 'processor': None, 'device': None}


def _pick_device() -> str:
    """HSWA_IT2_DEVICE = cpu (default) | mps. The 200M model is fast on CPU; MPS is opt-in
    because support for this remote-code model on MPS is not assumed."""
    import torch
    want = os.environ.get('HSWA_IT2_DEVICE', 'cpu').lower()
    if want == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    if want == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _use_cache() -> bool:
    # The official modeling_indictrans.py indexes legacy tuple caches (past_key_value[0].shape);
    # newer Transformers pass Cache objects, giving "'NoneType' object has no attribute 'shape'".
    # use_cache=False avoids that code path. Set HSWA_IT2_USE_CACHE=1 to test the cached path.
    return os.environ.get('HSWA_IT2_USE_CACHE', '0') == '1'


def _indic_processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor  # current toolkit layout
    except ImportError:
        from IndicTransToolkit import IndicProcessor  # older layout
    return IndicProcessor(inference=True)


def is_loaded() -> bool:
    return _state['model'] is not None


def load():
    if is_loaded():
        return
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
    _state.update(tokenizer=tokenizer, model=model, processor=_indic_processor(), device=device)


def unload():
    _state.update(tokenizer=None, model=None, processor=None, device=None)
    gc.collect()


def translate_hindi_to_english(hindi: str) -> str:
    if not hindi or not hindi.strip():
        raise ValueError('Hindi input is empty.')
    try:
        import torch
        load()
        tokenizer, model, processor, device = (_state[k] for k in ('tokenizer', 'model', 'processor', 'device'))
        batch = processor.preprocess_batch([hindi.strip()], src_lang='hin_Deva', tgt_lang='eng_Latn')
        encoded = tokenizer(batch, truncation=True, max_length=256, padding='longest',
                            return_tensors='pt', return_attention_mask=True).to(device)
        with torch.inference_mode():
            outputs = model.generate(**encoded, use_cache=_use_cache(), min_length=0,
                                     max_length=128, num_beams=3, num_return_sequences=1)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = processor.postprocess_batch(decoded, lang='eng_Latn')
        if not translations or not translations[0].strip():
            raise RuntimeError('IndicTrans2 returned no translation.')
        return translations[0].strip()
    except Exception as exc:
        # IndicProcessor keeps a queue of placeholder maps (one per preprocess call, consumed by
        # postprocess). A failure in between leaves a stale entry, so replace the processor.
        if _state['model'] is not None:
            _state['processor'] = _indic_processor()
        raise RuntimeError(f'IndicTrans2 failed: {type(exc).__name__}: {exc}') from exc
