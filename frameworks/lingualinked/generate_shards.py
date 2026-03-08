#!/usr/bin/env python3
"""
Generate ONNX model shards for LinguaLinked.

Usage:
  python generate_shards.py --model opt125m --split 2
  python generate_shards.py --model tinyllama --split 2
  python generate_shards.py --model llama-2-7b --split 4

Output goes to: onnx_model/backup/<model>_quantized_int8_res/
root.py --model_dir should point to that directory.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from util.model_card import ModelCard, available_models


def main():
    parser = argparse.ArgumentParser(description="Generate ONNX shards for LinguaLinked")
    parser.add_argument("--model", required=True,
                        help=f"Model key. Available: {list(available_models.keys())}")
    parser.add_argument("--split", type=int, default=None,
                        help="Number of shards (devices). Default: auto from model size.")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip int8 quantization (larger shards, faster generation)")
    parser.add_argument("--no-residual", action="store_true",
                        help="Use sequential (non-residual) split")
    args = parser.parse_args()

    if args.model not in available_models:
        print(f"ERROR: Unknown model '{args.model}'. Available: {list(available_models.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" Generating shards for: {args.model}")
    print(f" Split size: {args.split or 'auto'}")
    print(f" Quantization: {not args.no_quantize}")
    print(f" Residual connections: {not args.no_residual}")
    print(f"{'='*60}\n")

    card = ModelCard(
        model_name=args.model,
        quantization_option=not args.no_quantize,
        residual_connection=not args.no_residual,
        split_size=args.split,
    )
    card.prepare_model_split()

    out = card.onnx_module_to_split_path
    print(f"\n{'='*60}")
    print(f" Shards saved to: {out}")
    print(f" Pass this to root.py: --model_dir {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
