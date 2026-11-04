# Mini-Project Report — Hinglish Smart Writing Assistant

> **Status:** all three models run locally on a 16 GB MacBook Air (Apple Silicon). Sentence-level results below are from real runs on 3–4 Nov 2026. All figures below come from these runs; full per-case outputs are in `evaluation/`.

## 1. Abstract
A locally running writing assistant that converts Roman-script Hinglish into Devanagari Hindi (AI4Bharat IndicXlit), improves its phrasing (BharatGen Param-1-2.9B-Instruct) and translates it into English (AI4Bharat IndicTrans2), with a Streamlit interface, designed to fit a 16 GB Apple Silicon laptop.

## 2. Problem and objective
Indian users often type Hindi in Roman script, mixed with English words and with inconsistent spellings. The objective is to (a) produce correct native-script Hindi, (b) improve its grammar and phrasing without changing meaning, and (c) optionally translate it into English, using only the assignment-approved models.

## 3. Models and data
| Component | Artifact | Task |
|---|---|---|
| AI4Bharat IndicXlit | `ai4bharat-transliteration` (IndicXlit v1.0) | Roman → Devanagari |
| BharatGen Param1 | `bharatgenai/Param-1-2.9B-Instruct` | Hindi rewriting via instruction prompt |
| AI4Bharat IndicTrans2 | `ai4bharat/indictrans2-indic-en-dist-200M` | Hindi → English |
| AI4Bharat Aksharantar | Hindi word pairs | transliteration evaluation |

Param1 is an early SFT checkpoint; its card warns outputs may be incoherent. Its page metadata says Apache-2.0 while the card text says MIT.

## 4. System design
Pipeline: Roman Hinglish → whitespace normalisation → IndicXlit → Param1 → IndicTrans2.

- **Three modes:** transliteration only; transliteration + English; full pipeline.
- **Lazy, sequential loading:** models load on first use; optional low-memory mode unloads Param1 before IndicTrans2.
- **Precision/devices:** Param1 in bfloat16 on MPS (~5.8 GB instead of ~11.6 GB in float32); IndicTrans2 and IndicXlit on CPU.
- **Fault tolerance:** a failed stage is reported with its error; IndicTrans2 then translates the raw IndicXlit output and the interface labels the source. No model is ever substituted.

## 5. Engineering problems and solutions
| Problem | Root cause | Resolution (verified) |
|---|---|---|
| `tokenizers` build failure | project had been run with Homebrew Python 3.13/3.14 (no prebuilt wheels) | dedicated Python 3.10.22 venv; VS Code pinned to it |
| fairseq install: "no matching distributions … omegaconf" | fairseq needs `omegaconf<2.1`; pip ≥24.1 rejects its metadata | install with `pip<24.1`; fairseq 0.12.2 from source |
| Param1 tokenizer "ModelWrapper" error | `tokenizers` too old for the checkpoint's `tokenizer.json` | transformers 4.49.0 / tokenizers 0.21.4: tokenizer loads |
| Param1 has no `.generate()` | Transformers ≥4.50 dropped generation from remote-code classes | transformers 4.49.0: `has generate: True` |
| Param1 `KeyError: 'type'` (rope_scaling) | earlier Transformers/config mismatch | not reproduced: parsed `rope_scaling` is `None`, so no change to the checkpoint config was needed |
| Param1 memory | float32 load ≈ 11.6 GB | bfloat16 on MPS: 5.87 GB MPS memory |
| IndicTrans2 `NoneType … shape` | older Transformers + legacy tuple-cache code | not reproduced on 4.49.0; both cache settings work; `use_cache=False` kept |
| Probe froze in `IndicProcessor.postprocess_batch` | toolkit queues one placeholder map per preprocess call; postprocess blocks when the queue is empty | one preprocess per postprocess; processor replaced after a failed call |

**Measured footprint:** IndicTrans2 0.57 GB peak RSS, 0.4–0.7 s per sentence on MPS; Param1 8.1 s load, about 6 s per short sentence, 5.87 GB MPS memory. All three models fit in memory together.

## 6. Evaluation design
**Sentence set:** 40 hand-written cases in 8 categories (simple, Hindi-heavy, English-heavy, spelling variation, informal, ambiguous, student, long), each with a Devanagari transliteration reference and an English reference.

**Aksharantar:** a fixed random sample (seed 13, n = 500) of Hindi word pairs, for word-level evaluation independent of our own references.

**Configuration:** IndicXlit `beam_width=4, rescore=False`; IndicTrans2 beam 3, `use_cache=False`; Param1 greedy decoding, `use_cache=False`, at most 96 new tokens.

**Metrics**
- CER = (S + D + I) / reference characters; mean sentence CER and corpus CER.
- Word accuracy: position-aligned exact word matches / reference words (IndicXlit is word-for-word; count mismatches are flagged).
- Aksharantar: top-1 exact-match accuracy and CER.
- English: corpus BLEU (local `sacrebleu`).
- Param1: manual judgement of grammaticality and meaning preservation (not scored against transliteration references).

## 7. Results
**Sentence set (40 cases, hand-written references)**

| Metric | Stage 3 (IndicXlit → IndicTrans2) | Stage 4 (IndicXlit → Param1 → IndicTrans2) |
|---|---|---|
| Transliterations succeeded | 40 / 40 | 40 / 40 |
| Exact sentence matches | 10 / 40 (25%) | 10 / 40 (25%) |
| Corpus CER | 8.86% | 8.86% |
| Mean sentence CER | 9.93% | 9.93% |
| Mean word accuracy | 73.6% | 73.6% |
| Word-count mismatches | 0 | 0 |
| Param1 failures | – | 0 |
| IndicTrans2 failures | 0 | 0 |
| English corpus BLEU (n = 40) | 43.67 | 43.51 |
| IndicTrans2 input | raw IndicXlit output | Param1 output |
| Wall time for 40 cases (incl. model loading) | 33.4 s | 230.6 s |

Transliteration metrics are identical across stages because they always score raw IndicXlit output. The BLEU difference (0.16) on 40 single-reference sentences is too small to show that Param1 changes translation quality either way.

Observed character-difference labels in the 30 imperfect sentences (one sentence can carry several): other spelling 28, ऑ vs ओ/आ 3, vowel length 2.

**Param1 change analysis** (stage 4 outputs; per-sentence labels in `evaluation/param1_review.csv`)

| Measure | Count |
|---|---|
| Output differs from input at all | 39 / 40 |
| Differs beyond adding a full stop (।) | 30 / 40 |

Manual categorisation of all 40 outputs:

| Category | Count | Example |
|---|---|---|
| Punctuation only (।, ?) | 11 | हम सब साथ चलेंगे → हम सब साथ चलेंगे। |
| Improved, meaning preserved | 13 | मुजे पानी चाहिए → मुझे पानी चाहिए। |
| Mixed (some fixes, some new errors) | 4 | …क्वेस्शन कैसे सोल्व कारू → …क्वेस्ट कैसे हल किया जाए। |
| Meaning changed or made worse | 11 | वोह पास हो गया → वह गुज़र गया। ("passed" → "passed away") |
| Unchanged | 1 | घर पर बात करना |

Observations:
- Most genuine improvements correct IndicXlit's own errors (मैन → मैं, नही → नहीं, केई लिये → के लिए) or add missing case markers (की, का).
- A recurring failure is a change of grammatical person or gender (हूँ → है in cases 33 and 37; गया → गई in case 25; neutral sentences rewritten in the masculine in cases 2 and 6).
- When the transliteration is ambiguous or wrong (अचा, एपी, एएम), Param1 guesses a different word instead of the intended one, e.g. अचा (good) → अजीब (strange).
- Improvements and harmful changes roughly cancel out, which is consistent with the almost identical BLEU in stages 3 and 4 (43.67 vs 43.51).
- Against the transliteration references, Param1 output is closer in 7 sentences, further in 21 and equal in 12 (CER 8.86% → 22.09%). This is not a quality score for Param1, because legitimate rewording (रेडी → तैयार) counts as error; it shows that Param1 often goes beyond spelling correction.

**Aksharantar word-level evaluation** (official Hindi test split `hin_test.json`, 10,112 pairs; fixed random sample n = 500, seed 13; same IndicXlit settings as the app)

| Metric | Value |
|---|---|
| Words evaluated / errors | 500 / 0 |
| Top-1 exact-match word accuracy | 60.8% (304 / 500) |
| Mean word CER | 10.45% |
| Corpus CER | 9.09% |

Observed difference labels in the 196 incorrect words: other spelling 158, vowel length (इ/ई, उ/ऊ) 14, nukta 13, halant/schwa 7, ऑ vs ओ/आ 3, chandrabindu/anusvara 1.

The corpus CER on Aksharantar (9.09%) is close to that on the hand-written sentence set (8.86%), which suggests the custom references are not unusually strict or lenient. Exact-match accuracy is much lower than 1 − CER because a single differing character makes a word wrong, and Aksharantar provides one reference per word, so valid alternative spellings (e.g. nukta or vowel-length variants) are counted as errors.

## 8. Failure analysis
Thirty of the 40 sentences contain at least one transliteration mismatch, 50 mismatched words in total. Each word was assigned a cause after manual inspection (per-row causes in `evaluation/failure_analysis.csv`, column `manual_cause`).

**Transliteration error causes (50 words)**

| Cause | Words | Examples (reference → IndicXlit) |
|---|---|---|
| Short Roman word read as English letters or with English phonetics | 13 | आप → एपी (*ap*), अब → एबी (*ab*), आ → एए (*aa*), के → केई (*ke*), तुम → टम (*tum*) |
| Nasalisation not written in the Roman input | 12 | मैं → मैन (*main*), नहीं → नही (*nahi*), कहाँ → कहन (*kahan*) |
| English loanword spelling convention | 9 | एग्जाम → एक्साम, मेल → मैल, ऑनलाइन → ओनलाइन |
| Vowel length / schwa | 9 | अभी → अभि, चलो → चालो, सुबह → सुबाह |
| Non-standard Roman input (user's spelling) | 4 | अच्छा → अचा (*acha*), मुझे → मुजे (*muje*), बहुत → भट (*bht*) |
| Valid alternative spelling (reference strictness) | 3 | लिए → लिये, आएगा → आयेगा |

The two largest groups are systematic. IndicXlit transliterates word by word without sentence context, so very short words that look like English (*ap*, *ab*, *aa*, *ke*) are spelt as English letter names, and *main* (मैं, "I") becomes मैन because the Roman spelling does not mark nasalisation. Only 7 of the 50 errors (non-standard input and valid alternatives) are arguably not the model's fault.

**Representative end-to-end failures**

| # | Input | IndicXlit | Param1 | English (IndicTrans2) | Expected English | Stage and cause |
|---|---|---|---|---|---|---|
| 26 | kal main aaya tha | कल मैन आया था | (full stop only) | The man came yesterday. | I came yesterday. | Transliteration: *main* → मैन ("man"); error propagates |
| 34 | professor ne mail bheja hai | …मैल भेजा है | (full stop only) | The professor has sent the dirt. | …has sent an email. | Transliteration: मेल spelt मैल, a Hindi word meaning "dirt" |
| 18 | ap kaha ja rahe ho | एपी कहा जा रहे हो | एपी कहा जा रहा है। | It's called AP. | Where are you going? | Transliteration (*ap* → एपी), then Param1 changes person |
| 5 | tum kahan ho | टम कहन हो | (full stop only) | Tell me. | Where are you? | Transliteration: two errors make the sentence unreadable |
| 36 | agar tum jaldi aaoge… | अगर टम जल्दी आओगे… | …आ जाए… | If Tom comes early… | If you come early… | Transliteration: तुम → टम read as the name "Tom" |
| 3 | aaj mausam acha hai | …अचा है | आज मौसम अजीब है। | The weather is strange today. | The weather is good today. | Input spelling *acha*; Param1 guesses the wrong word |
| 29 | woh pass ho gaya | वोह पास हो गया | वह गुज़र गया। | He went through. | He passed. | Param1 changes meaning ("passed" → "passed away") |
| 37 | main kal subah train se delhi ja raha hoon | मैन … जा रहा हूँ | … जा रहा है। | Man is going to Delhi by Subah train tomorrow. | I am going to Delhi by train tomorrow morning. | Transliteration (मैन, सुबाह) plus Param1 changes हूँ → है |
| 27 | usne aam khaya | उसने एएम खाया | उसने सुबह का खाना खाया। | He had breakfast. | He ate a mango. | Ambiguity: *aam* read as "AM", Param1 interprets it as morning |
| 1 | kal mera exam hai | कल मेरा एक्साम है | (full stop only) | Tomorrow is my exam. | My exam is tomorrow. | Harmless loanword variant: counted as a CER error, translation still correct |

**Patterns across stages**
- Most English-side failures start in transliteration, and IndicTrans2 then translates the wrong word faithfully (मैन → "man", मैल → "dirt", टम → "Tom").
- Param1 repairs some of these errors (मैन → मैं in cases 4 and 14) but introduces others, most often a change of grammatical person or gender.
- IndicTrans2 itself produced fluent English in every case; its errors trace back to its input.

## 9. Limitations
- IndicXlit transliterates each word independently, so short words that look like English (*ap*, *ab*, *aa*) and words whose Roman spelling hides nasalisation (*main*, *nahi*) are systematically mis-spelt.
- Transliteration is context-free: `kal` (yesterday/tomorrow) and `aam` (mango/AM) stay ambiguous.
- Single-reference exact matching penalises valid alternative spellings (लिए/लिये, nukta and vowel-length variants), so word accuracy understates usable quality.
- Param1 (an early SFT checkpoint) improves about a third of sentences but changes meaning, person or gender in a similar number; its output should be shown next to the IndicXlit text rather than replacing it. It runs without a KV cache and is the slowest stage (~5 s per sentence).
- The 40-sentence set is small and single-reference; the stage 3 vs stage 4 BLEU difference (0.16) is not meaningful at this size.

## 10. Conclusion
The complete pipeline (IndicXlit → BharatGen Param1 → IndicTrans2) runs locally on a 16 GB MacBook Air using only the assignment-approved models: Param1 in bfloat16 on MPS needs 5.87 GB, IndicTrans2-200M about 0.6 GB. Reaching a working setup required diagnosing several version incompatibilities (Python version, Transformers 4.45–4.49 window, pip and fairseq metadata, an IndicTransToolkit queue issue) rather than replacing any model.

Measured quality: IndicXlit reaches a corpus CER of 8.86% and word accuracy of 73.6% on 40 hand-written sentences, consistent with 9.09% CER and 60.8% top-1 word accuracy on 500 Aksharantar test words. IndicTrans2 reaches a corpus BLEU of 43.7 on the same sentences. Param1 edits 30 of 40 sentences; its improvements and its meaning-changing errors roughly balance, so it does not measurably change translation quality.

The main bottleneck is context-free transliteration: errors made at this first stage propagate to the English output. Natural next steps are sentence-level context or IndicXlit's top-k candidates for short ambiguous words, a more constrained Param1 prompt evaluated on a separate development set, and a larger multi-reference test set.

## References
- https://github.com/AI4Bharat/IndicXlit
- https://github.com/AI4Bharat/IndicTrans2
- https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
- https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct
- https://huggingface.co/datasets/ai4bharat/Aksharantar
