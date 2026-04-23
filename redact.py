#!/usr/bin/env python3
"""CLI tool that redacts PII from text files using openai/privacy-filter."""

import argparse
import sys
from pathlib import Path

from transformers import pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "model"


def main():
    parser = argparse.ArgumentParser(description="Redact PII from a text file.")
    parser.add_argument("file", help="Path to the input text file (use - for stdin)")
    parser.add_argument("--model", default=None,
                        help="Model name or path (default: ./model or openai/privacy-filter)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum confidence score to redact (default: 0.5)")
    args = parser.parse_args()

    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file) as f:
            text = f.read()

    model_path = args.model
    if model_path is None:
        model_path = str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "openai/privacy-filter"

    print(f"Loading model from {model_path}...", file=sys.stderr)
    classifier = pipeline(
        task="token-classification",
        model=model_path,
    )
    print("Model loaded. Redacting...", file=sys.stderr)

    def redact_with_threshold(text: str) -> str:
        entities = classifier(text, aggregation_strategy="simple")
        entities = [e for e in entities if e["score"] >= args.threshold]
        entities.sort(key=lambda e: e["start"])

        # Merge overlapping/adjacent spans
        merged = []
        for ent in entities:
            if merged and ent["start"] <= merged[-1]["end"]:
                prev = merged[-1]
                prev["end"] = max(prev["end"], ent["end"])
                if ent["score"] > prev["score"]:
                    prev["entity_group"] = ent["entity_group"]
                    prev["score"] = ent["score"]
            else:
                merged.append(dict(ent))

        # Apply replacements in reverse order
        result = list(text)
        for ent in reversed(merged):
            label = ent["entity_group"]
            result[ent["start"]:ent["end"]] = list(f"[REDACTED:{label}]")
        return "".join(result)

    print(redact_with_threshold(text))


if __name__ == "__main__":
    main()
