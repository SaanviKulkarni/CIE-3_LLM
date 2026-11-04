"""CLI to test each stage independently: 2=IndicXlit, 3=IndicXlit+IndicTrans2, 4=full pipeline."""
import argparse
import json

from src.pipeline import process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', help='Hinglish sentence')
    parser.add_argument('--stage', choices=['2', '3', '4'], default='2')
    parser.add_argument('--low-memory', action='store_true', help='unload Param1 after use')
    args = parser.parse_args()
    result = process(args.text, improve=args.stage == '4', english=args.stage in ('3', '4'),
                     low_memory=args.low_memory, on_stage=print)
    print(json.dumps({k: v for k, v in result.to_dict().items() if v not in (None, {}, [])},
                     ensure_ascii=False, indent=2))
