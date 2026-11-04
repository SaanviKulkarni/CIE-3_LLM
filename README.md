# Hinglish Smart Writing Assistant

Roman-script Hinglish → **AI4Bharat IndicXlit** (Devanagari Hindi) → **BharatGen Param1** (improved Hindi) → **AI4Bharat IndicTrans2** (English). Streamlit front end, fully local inference on an Apple Silicon Mac. Only assignment-approved models are used; no external inference APIs.

## 1. Problem and objective
Roman-script Hindi–English writing is informal and inconsistent. The goal is native-script Hindi, improved phrasing that keeps the meaning, and an optional English translation.

## 2. Models
| Component | Official artifact | Role |
|---|---|---|
| IndicXlit | `ai4bharat-transliteration` package (official IndicXlit v1.0 model) | Roman → Devanagari |
| BharatGen Param1 | `bharatgenai/Param-1-2.9B-Instruct` | Hindi rewriting |
| IndicTrans2 | `ai4bharat/indictrans2-indic-en-dist-200M` (gated) | Hindi → English |
| Aksharantar | `ai4bharat/Aksharantar` (Hindi word pairs) | transliteration evaluation |

Licence note: the Param1 page's metadata says Apache-2.0 while its card text says MIT; confirm before redistributing.

## 3. Architecture (lightweight by design)
- **Lazy loading.** No model loads when Streamlit starts; each loads on first use and stays in memory.
- **Three modes.** Transliteration only · Transliteration + English · Full pipeline.
- **Sequential, fault-tolerant execution.** If Param1 fails, the error is shown and IndicTrans2 translates the *raw IndicXlit output*, labelled as such. Nothing is substituted silently.
- **Memory.** Param1 loads in bfloat16 (~5.8 GB, the checkpoint's own dtype) rather than float32 (~11.6 GB). Optional low-memory mode unloads Param1 after each run. `torch.inference_mode()`, bounded output lengths, beam size 3 for IndicTrans2, greedy decoding for Param1.
- **Devices** (environment variables):

| Variable | Default | Values |
|---|---|---|
| `HSWA_PARAM1_DEVICE` | `mps` | `mps`, `cpu` |
| `HSWA_IT2_DEVICE` | `cpu` | `cpu`, `mps` |
| `HSWA_IT2_USE_CACHE` | `0` | `1` re-enables KV cache (triggers the known `NoneType … shape` error on newer Transformers) |

### Compatibility fixes applied (and why)
| Symptom | Cause | Fix in code |
|---|---|---|
| Param1 tokenizer: `data did not match any variant of untagged enum ModelWrapper` | `tokenizers` too old for the checkpoint's `tokenizer.json` | `transformers>=4.45` (pulls `tokenizers` 0.20+) |
| `ParamBharatGenForCausalLM … has no attribute 'generate'` | Transformers ≥4.50 no longer gives remote-code classes `.generate()` | `transformers<4.50`; clear error if violated |
| Param1 `KeyError: 'type'` (rope_scaling) | remote code reads legacy `rope_scaling['type']`; config uses `rope_type` | `normalise_rope_config`: `default` → `None` (equivalent), other types mirrored under `type` |
| IndicTrans2 `'NoneType' object has no attribute 'shape'` | older Transformers + legacy tuple-cache code | not reproduced with transformers 4.49.0 (both cache settings work); app keeps `use_cache=False` |
| IndicTrans2 probe froze in `postprocess_batch` | `IndicProcessor` queues one placeholder map per preprocess call and postprocess blocks until one exists | one preprocess per postprocess; processor replaced after a failed call |
| `tokenizers` build failure | project had been run with **Python 3.13** (no prebuilt wheels) | use the Python 3.10 venv, also inside VS Code |

## 4. Installation (macOS, Apple Silicon, Python 3.10)
**Quick path:** `brew install python@3.10` (if needed), then `bash setup_mac.sh`. It creates `.venv` with Python 3.10 (replacing a broken one), installs torch, fairseq (cloned to `~/fairseq`), the requirements and IndicTransToolkit. The manual steps it performs are:

```bash
cd hinglish-smart-writing-assistant
python3.10 -m venv .venv
source .venv/bin/activate
python --version                     # must print Python 3.10.x
python -m pip install "pip<24.1"     # pip ≥24.1 rejects fairseq's omegaconf<2.1
python -m pip install torch          # macOS arm64 wheel includes MPS support

# fairseq (needed by IndicXlit): install from the official source, cloned OUTSIDE this folder.
# Replace with the exact command/commit that worked for you and record it here.
cd .. && git clone https://github.com/facebookresearch/fairseq.git && cd fairseq && python -m pip install --editable ./ && cd ../hinglish-smart-writing-assistant

python -m pip install -r requirements.txt
python -m pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
huggingface-cli login                # after accepting the IndicTrans2 gate on its model page
```
**VS Code:** `.vscode/settings.json` points VS Code at `.venv/bin/python`. Reload the window (Cmd+Shift+P → *Developer: Reload Window*) and check that the status bar shows Python 3.10.x. If not: Cmd+Shift+P → *Python: Select Interpreter* → `.venv/bin/python`. Open a new terminal afterwards.

Models download once to `~/.cache/huggingface` and are reused.

## 5. Verify each model in isolation
```bash
python tests/probe_models.py versions
python tests/probe_models.py xlit
python tests/probe_models.py it2          # tries use_cache=True then False
python tests/probe_models.py param1       # bf16 on MPS; add "cpu" to force CPU
```

## 6. Run
```bash
python run_stage.py "kal mera exam hai" --stage 2   # IndicXlit
python run_stage.py "kal mera exam hai" --stage 3   # + IndicTrans2
python run_stage.py "kal mera exam hai" --stage 4   # full pipeline
streamlit run app.py
```

## 7. Evaluation
```bash
python -m unittest discover -s tests                 # metric + pipeline logic (no models)
python -m evaluation.evaluate --stage 2 --limit 5     # smoke test
python -m evaluation.evaluate --stage 3               # CER, word accuracy, BLEU on raw transliteration
python -m evaluation.evaluate --stage 4 --low-memory  # full pipeline
```
Outputs: `evaluation/results.csv`, `summary.json`, `failure_analysis.csv`.

**Aksharantar (word level):** download the Hindi split from the official dataset (check the file names on the dataset page first), then:
```bash
huggingface-cli download ai4bharat/Aksharantar --repo-type dataset --include "hin*" --local-dir data/aksharantar
python -m evaluation.evaluate --aksharantar data/aksharantar/<hindi test file> --n 500 --seed 13
```

**Metrics**
- **CER** = (S + D + I) / reference characters (Unicode code points, spaces included); reported as mean sentence CER and corpus CER.
- **Word accuracy** — alignment policy: whitespace tokens compared position by position (IndicXlit is word-for-word); exact matches ÷ reference words. Rows with differing word counts are flagged.
- **Aksharantar**: top-1 exact-match word accuracy and CER on a fixed seeded sample.
- **English**: corpus BLEU with local `sacrebleu`; `translation_source` records whether IndicTrans2 received Param1 or raw IndicXlit text.
- **Param1** output is never scored against transliteration references; judge it manually for grammar and meaning preservation.
- `failure_analysis.csv` labels each mismatch by *observed difference* (anusvara/chandrabindu, nukta, vowel length, halant, ऑ, segmentation, other). Fill `manual_cause` yourself.

The 40 sentences in `test_cases.csv` are hand-written references, not drawn from Aksharantar. Valid alternative spellings (e.g. हूँ/हूं) count as errors under exact matching; report that.

## 8. Example (target, not a verified output)
`kal mera exam hai` → `कल मेरा एग्जाम है` → `कल मेरी परीक्षा है।` → `My exam is tomorrow.` (`kal` is ambiguous between yesterday and tomorrow).

## 9. Limitations
- IndicXlit converts script only; no context, so ambiguity remains.
- Param1 is an early SFT checkpoint and can change meaning; it runs with `use_cache=False`, so it is slow.
- The 40-sentence set is small and has single references.

## 10. Tested environment
All three models ran successfully with this configuration (probe runs and the stage 3/4 evaluations, November 2026). Full package list: `requirements-lock.txt`.

| Item | Version / value |
|---|---|
| Machine | MacBook Air, Apple Silicon, 16 GB unified memory |
| macOS | 26.6.2 (arm64) |
| Python | 3.10.22 (Homebrew) |
| torch | 2.14.1 (MPS available) |
| transformers / tokenizers | 4.49.0 / 0.21.4 |
| huggingface_hub / safetensors | 0.36.2 / 0.8.0 |
| sentencepiece / numpy | 0.2.2 / 2.2.6 |
| fairseq | 0.12.2 (installed from source; pip < 24.1 needed for its omegaconf dependency) |
| IndicTransToolkit | installed from GitHub (no version string) |
| streamlit | 1.65.0 |
| IndicXlit settings | `beam_width=4, rescore=False` |
| Param1 | bf16 on MPS; load 8.1 s; one generation 6.3 s; 5.87 GB MPS memory |
| IndicTrans2 | `use_cache=True` and `use_cache=False` both worked with transformers 4.49.0 |
