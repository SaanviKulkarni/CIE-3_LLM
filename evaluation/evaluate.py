"""Real evaluation only: every number written here comes from an actual model run.

Sentence mode (default) : evaluation/test_cases.csv -> results.csv, summary.json, failure_analysis.csv
Aksharantar mode        : --aksharantar <file> -> aksharantar_results.csv, aksharantar_summary.json
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'evaluation' / 'test_cases.csv'
OUTPUT = ROOT / 'evaluation' / 'results.csv'
SUMMARY = ROOT / 'evaluation' / 'summary.json'
FAILURES = ROOT / 'evaluation' / 'failure_analysis.csv'
AKS_OUTPUT = ROOT / 'evaluation' / 'aksharantar_results.csv'
AKS_SUMMARY = ROOT / 'evaluation' / 'aksharantar_summary.json'


# ---------------------------------------------------------------- metrics
def edit_distance(a, b) -> int:
    """Levenshtein distance; works on strings (characters) or lists (words)."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    """(S + D + I) / reference characters. Unicode code points, spaces included."""
    return edit_distance(reference, prediction) / max(1, len(reference))


def word_accuracy(reference: str, prediction: str) -> float:
    """Alignment policy: whitespace tokens compared position by position (IndicXlit transliterates
    word-for-word, so token counts normally match); exact matches / reference word count.
    Rows where counts differ are flagged in results.csv (word_count_match = False)."""
    ref_words, pred_words = reference.split(), prediction.split()
    if not ref_words:
        return float(not pred_words)
    return sum(a == b for a, b in zip(ref_words, pred_words)) / len(ref_words)


# ------------------------------------------------- deterministic diff labels
_VARIANTS = [  # (label, transformation applied to both strings before comparing)
    ('chandrabindu/anusvara (ँ vs ं)', lambda s: s.replace('ँ', 'ं')),
    ('nukta (़)', lambda s: s.replace('़', '')),
    ('vowel length (इ/ई, उ/ऊ)', lambda s: s.translate(str.maketrans({'ी': 'ि', 'ू': 'ु', 'ई': 'इ', 'ऊ': 'उ'}))),
    ('halant/schwa (्)', lambda s: s.replace('्', '')),
    ('ऑ vs ओ/आ (English vowel)', lambda s: s.translate(str.maketrans({'ॉ': 'ो', 'ऑ': 'ओ'}))),
]


def diff_type(ref_word: str, pred_word: str) -> str:
    for label, f in _VARIANTS:
        if f(ref_word) == f(pred_word):
            return label
    return 'other spelling'


def describe_errors(reference: str, prediction: str):
    """Returns (error_types, mismatches). Labels describe the *observed difference*, not its cause."""
    ref_w, pred_w = reference.split(), prediction.split()
    if len(ref_w) != len(pred_w):
        return 'word segmentation/count', f'{reference} → {prediction}'
    pairs = [(r, p) for r, p in zip(ref_w, pred_w) if r != p]
    types = sorted({diff_type(r, p) for r, p in pairs})
    return '; '.join(types), '; '.join(f'{r}→{p}' for r, p in pairs)


def environment():
    info = {'python': platform.python_version(), 'machine': platform.machine()}
    for pkg in ('torch', 'transformers', 'tokenizers'):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    return info


# ------------------------------------------------------- sentence evaluation
def evaluate(stage: int, limit=None, low_memory=False):
    from src.pipeline import process
    with INPUT.open(encoding='utf-8', newline='') as f:
        cases = list(csv.DictReader(f))[:limit]
    results = []
    started = time.time()
    for item in cases:
        print(f"Case {item['id']}: {item['roman_hinglish']}", flush=True)
        row = dict(item)
        row.update(predicted_hindi='', predicted_improved_hindi='', predicted_english='',
                   translation_source='', cer='', word_accuracy='', word_count_match='',
                   error_type='', mismatched_words='', model_errors='', manual_cause='')
        try:
            out = process(item['roman_hinglish'], improve=stage >= 4,
                          english=stage >= 3 and bool(item['expected_english']), low_memory=low_memory)
            row['predicted_hindi'] = out.transliteration
            row['predicted_improved_hindi'] = out.improved_hindi or ''
            row['predicted_english'] = out.english_translation or ''
            row['translation_source'] = out.translation_source or ''
            row['model_errors'] = json.dumps(out.errors, ensure_ascii=False) if out.errors else ''
            row['cer'] = round(cer(item['expected_hindi'], out.transliteration), 6)
            row['word_accuracy'] = round(word_accuracy(item['expected_hindi'], out.transliteration), 6)
            row['word_count_match'] = len(item['expected_hindi'].split()) == len(out.transliteration.split())
            if row['cer'] > 0:
                row['error_type'], row['mismatched_words'] = describe_errors(item['expected_hindi'], out.transliteration)
        except Exception as exc:
            row['model_errors'] = json.dumps({'indicxlit': str(exc)}, ensure_ascii=False)
            print('  ERROR: ' + str(exc), flush=True)
        results.append(row)

    fields = list(results[0]) if results else []
    failed = [r for r in results if r['model_errors'] or (r['cer'] != '' and r['cer'] > 0)]
    for path, data in ((OUTPUT, results), (FAILURES, failed)):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    good = [r for r in results if r['cer'] != '']
    pairs = [(r['predicted_english'], r['expected_english']) for r in good
             if r['expected_english'] and r['predicted_english']]
    bleu = None
    if pairs:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True).corpus_score([p for p, _ in pairs], [[r for _, r in pairs]]).score
    error_counts = {}
    for r in failed:
        for t in filter(None, (r['error_type'] or '').split('; ')):
            error_counts[t] = error_counts.get(t, 0) + 1
    summary = {
        'stage': stage, 'requested_cases': len(cases),
        'successful_transliterations': len(good), 'failed_transliterations': len(cases) - len(good),
        'exact_sentence_matches': sum(r['cer'] == 0 for r in good),
        'mean_sentence_cer': sum(r['cer'] for r in good) / len(good) if good else None,
        'corpus_cer': (sum(edit_distance(r['expected_hindi'], r['predicted_hindi']) for r in good)
                       / sum(max(1, len(r['expected_hindi'])) for r in good)) if good else None,
        'mean_word_accuracy': sum(r['word_accuracy'] for r in good) / len(good) if good else None,
        'word_count_mismatches': sum(r['word_count_match'] is False for r in good),
        'param1_failures': sum('param1' in r['model_errors'] for r in results),
        'indictrans2_failures': sum('indictrans2' in r['model_errors'] for r in results),
        'english_evaluated_count': len(pairs), 'corpus_bleu': bleu,
        'translation_sources': sorted({r['translation_source'] for r in good if r['translation_source']}),
        'observed_error_types': error_counts,
        'wall_time_s': round(time.time() - started, 1), 'environment': environment(),
        'metric_notes': ('CER/word accuracy compare raw IndicXlit output with hand-written references '
                         '(Param1 output is never scored against them). BLEU compares IndicTrans2 output '
                         'with hand-written English references; see translation_sources for its input. '
                         'error_type labels describe observed character differences, not verified causes.'),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Fill the manual_cause column in {FAILURES.name} after inspecting each row.')


# ---------------------------------------------------- Aksharantar evaluation
def load_aksharantar(path: Path):
    """Reads an Aksharantar Hindi split (.json list, .jsonl, or .csv) with
    'english word' / 'native word' fields."""
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.csv':
        rows = list(csv.DictReader(text.splitlines()))
    elif text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = [(r['english word'].strip(), r['native word'].strip()) for r in rows
             if r.get('english word') and r.get('native word')]
    if not pairs:
        raise ValueError(f'No "english word"/"native word" pairs found in {path}')
    return pairs


def evaluate_aksharantar(path: Path, n: int, seed: int):
    from src.transliteration import transliterate
    pairs = load_aksharantar(path)
    sample = random.Random(seed).sample(pairs, min(n, len(pairs)))
    rows = []
    for i, (roman, native) in enumerate(sample, 1):
        try:
            pred = transliterate(roman)
            rows.append(dict(roman=roman, reference=native, predicted=pred, exact=pred == native,
                             cer=round(cer(native, pred), 6),
                             error_type='' if pred == native else diff_type(native, pred), error=''))
        except Exception as exc:
            rows.append(dict(roman=roman, reference=native, predicted='', exact='', cer='', error_type='', error=str(exc)))
        if i % 50 == 0:
            print(f'{i}/{len(sample)}', flush=True)
    with AKS_OUTPUT.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = [r for r in rows if r['cer'] != '']
    types = {}
    for r in ok:
        if r['error_type']:
            types[r['error_type']] = types.get(r['error_type'], 0) + 1
    summary = {
        'source_file': path.name, 'pool_size': len(pairs), 'sampled': len(sample), 'seed': seed,
        'evaluated': len(ok), 'errors': len(rows) - len(ok),
        'top1_word_accuracy': sum(r['exact'] for r in ok) / len(ok) if ok else None,
        'mean_word_cer': sum(r['cer'] for r in ok) / len(ok) if ok else None,
        'corpus_cer': (sum(edit_distance(r['reference'], r['predicted']) for r in ok)
                       / sum(len(r['reference']) for r in ok)) if ok else None,
        'observed_error_types': types, 'environment': environment(),
        'notes': 'Word-level: one Aksharantar reference per word, so valid alternative spellings count as errors.',
    }
    AKS_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, choices=[2, 3, 4], default=2)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--low-memory', action='store_true')
    p.add_argument('--aksharantar', type=Path, help='path to an Aksharantar Hindi split file')
    p.add_argument('--n', type=int, default=500, help='Aksharantar sample size')
    p.add_argument('--seed', type=int, default=13)
    args = p.parse_args()
    if args.aksharantar:
        evaluate_aksharantar(args.aksharantar, args.n, args.seed)
    else:
        evaluate(args.stage, args.limit, args.low_memory)
