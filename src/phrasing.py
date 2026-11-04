"""Hindi rewriting with the official BharatGen Param-1-2.9B-Instruct checkpoint.

Memory: weights load in bfloat16 (the checkpoint's own dtype, ~5.8 GB) instead of
float32 (~11.6 GB). The model loads only when first used and can be unloaded.
No other model is ever substituted; failures are raised to the caller.
"""
import gc
import os

MODEL_ID = 'bharatgenai/Param-1-2.9B-Instruct'
MAX_PROMPT_TOKENS = 256          # chat template + instruction + sentence
MAX_NEW_TOKENS_CAP = 96          # generation runs with use_cache=False, so keep it short

_state = {'tokenizer': None, 'model': None, 'device': None, 'notes': []}

SYSTEM_PROMPT = 'आप हिंदी लेखन सहायक हैं। केवल सुधारा हुआ हिंदी वाक्य लिखें। अर्थ न बदलें, कोई टिप्पणी न जोड़ें।'
USER_PROMPT = 'इस वाक्य को स्वाभाविक, स्पष्ट और व्याकरण की दृष्टि से सही हिंदी में लिखें। अर्थ, समय और नाम वही रखें:\n'


def _pick_device() -> str:
    """HSWA_PARAM1_DEVICE = mps (default) | cpu | cuda. Falls back to CPU only if unavailable."""
    import torch
    want = os.environ.get('HSWA_PARAM1_DEVICE', 'mps').lower()
    if want == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    if want == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def normalise_rope_config(config):
    """Fix the observed KeyError: 'type' without changing model behaviour.

    Newer Transformers writes rope_scaling as {'rope_type': ...}; the checkpoint's
    remote code reads the legacy key rope_scaling['type'].
    - rope_type 'default' means *no scaling*, so None is exactly equivalent.
    - any other rope_type is kept and mirrored under the legacy 'type' key.
    Returns a note describing what was changed (or None).
    """
    rs = getattr(config, 'rope_scaling', None)
    if not isinstance(rs, dict) or 'type' in rs or 'rope_type' not in rs:
        return None
    rope_type = rs['rope_type']
    if rope_type == 'default':
        config.rope_scaling = None
        return "config.rope_scaling {'rope_type': 'default'} set to None (equivalent: no RoPE scaling)"
    config.rope_scaling = {**rs, 'type': rope_type}
    return f"config.rope_scaling: mirrored rope_type={rope_type!r} under legacy 'type' key"


def is_loaded() -> bool:
    return _state['model'] is not None


def load():
    if is_loaded():
        return
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    notes = []
    # Model card loads the tokenizer WITHOUT remote code (plain tokenizer.json).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    note = normalise_rope_config(config)
    if note:
        notes.append(note)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if not hasattr(model, 'generate'):
        raise RuntimeError(
            f'{type(model).__name__} has no .generate() under transformers '
            f'{transformers.__version__}. From v4.50, PreTrainedModel no longer provides '
            'generation to remote-code classes; use transformers>=4.45,<4.50 (see README).')
    device = _pick_device()
    model.to(device).eval()
    _state.update(tokenizer=tokenizer, model=model, device=device, notes=notes)


def unload():
    """Release Param1 so other models have headroom (used in low-memory mode)."""
    _state.update(tokenizer=None, model=None, device=None, notes=[])
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def clean_output(raw: str) -> str:
    """Deterministic, documented clean-up: first non-empty line, common label prefixes removed.
    The raw generation is always returned alongside, so nothing is hidden."""
    for line in raw.splitlines():
        line = line.strip().strip('"“”\'')
        for prefix in ('सुधारा हुआ वाक्य:', 'सुधरा हुआ वाक्य:', 'उत्तर:', 'वाक्य:'):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            return line
    return ''


def improve_hindi(hindi: str) -> dict:
    """Returns {'text', 'raw', 'device', 'notes'}; raises RuntimeError on any failure."""
    if not hindi or not hindi.strip():
        raise ValueError('Hindi input is empty.')
    try:
        import torch
        load()
        tokenizer, model, device = _state['tokenizer'], _state['model'], _state['device']
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': USER_PROMPT + hindi.strip()}]
        ids = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True)
        if ids.shape[-1] > MAX_PROMPT_TOKENS:
            raise ValueError(f'Input too long for the Param1 stage ({ids.shape[-1]} > {MAX_PROMPT_TOKENS} tokens).')
        n_sentence = len(tokenizer(hindi, add_special_tokens=False)['input_ids'])
        budget = min(MAX_NEW_TOKENS_CAP, 2 * n_sentence + 16)
        ids = ids.to(device)
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=budget, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 use_cache=False)  # as in the official model card example
        raw = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        text = clean_output(raw)
        if not text:
            raise RuntimeError(f'Param1 returned empty output (raw={raw!r}).')
        return {'text': text, 'raw': raw, 'device': device, 'notes': list(_state['notes'])}
    except Exception as exc:
        raise RuntimeError(f'BharatGen Param1 failed: {type(exc).__name__}: {exc}') from exc
