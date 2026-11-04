"""
probe_models.py - diagnose each approved model in isolation (one model per process).

Usage (from the project root, inside the Python 3.10 venv):
    python tests/probe_models.py versions
    python tests/probe_models.py xlit
    python tests/probe_models.py it2 [mps|cpu]
    python tests/probe_models.py param1 [mps|cpu]

Nothing here patches a checkpoint or substitutes a model. Errors are printed in full.
"""
import importlib
import json
import platform
import resource
import sys
import time
import traceback


def peak_rss_gb():
    # macOS reports ru_maxrss in bytes (Linux reports KB)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def pick_device(arg):
    import torch
    if arg == "cpu":
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def versions():
    print("python", platform.python_version(), platform.machine(), platform.mac_ver()[0])
    pkgs = ["torch", "transformers", "tokenizers", "huggingface_hub", "safetensors",
            "sentencepiece", "numpy", "fairseq", "IndicTransToolkit", "streamlit", "accelerate"]
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            print(f"{p:20s} {getattr(m, '__version__', '(no __version__)')}")
        except Exception as e:
            print(f"{p:20s} NOT IMPORTABLE -> {type(e).__name__}: {e}")
    try:
        importlib.import_module("ai4bharat.transliteration")
        print(f"{'ai4bharat-xlit':20s} importable")
    except Exception as e:
        print(f"{'ai4bharat-xlit':20s} NOT IMPORTABLE -> {type(e).__name__}: {e}")
    import torch
    print("mps built/available:", torch.backends.mps.is_built(), torch.backends.mps.is_available())


def xlit():
    from ai4bharat.transliteration import XlitEngine
    t = time.time()
    engine = XlitEngine("hi", beam_width=4, rescore=True, src_script_type="en")
    print(f"load {time.time() - t:.1f}s")
    for s in ["kal mera exam hai", "mujhe kal library jana hai"]:
        out = engine.translit_sentence(s)
        print(repr(s), "->", repr(out))
    print(f"peak RSS {peak_rss_gb():.2f} GB")


def load_indic_processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor  # newer toolkit layout
    except ImportError:
        from IndicTransToolkit import IndicProcessor  # older layout
    return IndicProcessor(inference=True)


def it2(device_arg):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "ai4bharat/indictrans2-indic-en-dist-200M"
    device = pick_device(device_arg)
    t = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(device).eval()
    print(f"device={device} load {time.time() - t:.1f}s")

    for use_cache in (True, False):  # True reproduces the reported error if present
        # IndicProcessor queues one placeholder map per preprocess call and postprocess
        # blocks until one is available, so every attempt needs its own preprocess call
        # and a fresh processor (a failed attempt would leave a stale entry behind).
        ip = load_indic_processor()
        batch = ip.preprocess_batch(["कल मेरा एग्जाम है"], src_lang="hin_Deva", tgt_lang="eng_Latn")
        inputs = tok(batch, truncation=True, padding="longest",
                     return_tensors="pt", return_attention_mask=True).to(device)
        try:
            t = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, use_cache=use_cache, min_length=0,
                                     max_length=128, num_beams=2, num_return_sequences=1)
            dec = tok.batch_decode(gen, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
            print(f"use_cache={use_cache}: {ip.postprocess_batch(dec, lang='eng_Latn')} "
                  f"({time.time() - t:.1f}s)")
        except Exception:
            print(f"use_cache={use_cache}: FAILED")
            traceback.print_exc()
    print(f"peak RSS {peak_rss_gb():.2f} GB")


def param1(device_arg):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    name = "bharatgenai/Param-1-2.9B-Instruct"
    device = pick_device(device_arg)

    raw = json.load(open(hf_hub_download(name, "config.json")))
    print("raw config rope_scaling:", raw.get("rope_scaling"),
          "| rope_theta:", raw.get("rope_theta"),
          "| torch_dtype:", raw.get("torch_dtype"),
          "| auto_map:", raw.get("auto_map"))

    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)  # as in model card
        print("tokenizer OK:", type(tok).__name__)
    except Exception:
        print("TOKENIZER FAILED (ModelWrapper error => installed `tokenizers` too old)")
        traceback.print_exc()
        return

    try:
        config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        print("parsed config.rope_scaling:", config.rope_scaling)
        t = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        print(f"class {type(model).__name__} device={device} load {time.time() - t:.1f}s "
              f"| has generate: {hasattr(model, 'generate')}")
    except Exception:
        print("MODEL LOAD FAILED")
        traceback.print_exc()
        return

    if not hasattr(model, "generate"):
        print("No .generate(): this Transformers version no longer gives GenerationMixin "
              "to remote-code classes. Report the version above.")
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant that corrects Hindi grammar."},
        {"role": "user", "content": "निम्नलिखित हिंदी वाक्य का व्याकरण और शब्द-चयन सुधारें। "
                                    "केवल सुधरा हुआ वाक्य लिखें।\nवाक्य: कल मेरा एग्जाम है"},
    ]
    ids = tok.apply_chat_template(messages, return_tensors="pt",
                                  add_generation_prompt=True).to(device)
    try:
        t = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 eos_token_id=tok.eos_token_id, use_cache=False)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        print(f"OUTPUT ({time.time() - t:.1f}s): {text!r}")
    except Exception:
        print("GENERATE FAILED")
        traceback.print_exc()
    if device == "mps":
        print(f"MPS allocated {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    print(f"peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "versions"
    dev = sys.argv[2] if len(sys.argv) > 2 else "mps"
    {"versions": versions, "xlit": xlit,
     "it2": lambda: it2(dev), "param1": lambda: param1(dev)}[stage]()
