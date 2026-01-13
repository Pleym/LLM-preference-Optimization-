#!/usr/bin/env python3
"""
Data preparation and exploration script.

This script demonstrates the data pipeline for DPO vs SFT experiments.
It loads the Anthropic HH-RLHF dataset, prepares it for both training
paradigms, and displays statistics.

Usage:
    python scripts/prepare_data.py --split train --max_samples 1000
    python scripts/prepare_data.py --format dpo --save_path ./data/dpo_train.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_anthropic_hh,
    prepare_for_sft,
    prepare_for_dpo,
)
from src.data.dataset import get_dataset_statistics, create_train_val_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Anthropic HH-RLHF dataset for training"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to load",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to load (for debugging)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["sft", "dpo", "both"],
        help="Output format to prepare",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model name for tokenizer (used for chat template)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save prepared dataset (optional)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=3,
        help="Number of examples to display",
    )
    return parser.parse_args()


def print_separator(title: str = ""):
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)


def display_example(example: dict, idx: int, format_type: str):
    """Pretty print a single example."""
    print(f"\n--- Example {idx + 1} ---")

    if format_type == "sft":
        print(f"[Text] (first 500 chars):")
        print(
            example["text"][:500] + "..."
            if len(example["text"]) > 500
            else example["text"]
        )
    else:  # DPO
        print(f"[Prompt] (first 300 chars):")
        print(
            example["prompt"][:300] + "..."
            if len(example["prompt"]) > 300
            else example["prompt"]
        )
        print(f"\n[Chosen Response]:")
        print(
            example["chosen"][:200] + "..."
            if len(example["chosen"]) > 200
            else example["chosen"]
        )
        print(f"\n[Rejected Response]:")
        print(
            example["rejected"][:200] + "..."
            if len(example["rejected"]) > 200
            else example["rejected"]
        )


def main():
    args = parse_args()

    print_separator("Loading Anthropic HH-RLHF Dataset")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples or 'All'}")

    # Load raw dataset
    dataset = load_anthropic_hh(
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataset)} examples")

    # Show raw example
    print_separator("Raw Data Sample")
    raw_example = dataset[0]
    print("[Chosen conversation]:")
    print(raw_example["chosen"][:500] + "...")

    # Initialize variables
    sft_data = None
    dpo_data = None

    # Prepare for SFT
    if args.format in ["sft", "both"]:
        print_separator("SFT Dataset Preparation")
        sft_data = prepare_for_sft(dataset)
        stats = get_dataset_statistics(sft_data)

        print(f"Prepared {stats['num_examples']} examples for SFT")
        print(f"Columns: {stats['columns']}")
        if "text_length" in stats:
            print(
                f"Text length - Mean: {stats['text_length']['mean']:.0f}, "
                f"Min: {stats['text_length']['min']}, Max: {stats['text_length']['max']}"
            )

        print("\n--- SFT Examples ---")
        for i in range(min(args.show_examples, len(sft_data))):
            display_example(sft_data[i], i, "sft")

    # Prepare for DPO
    if args.format in ["dpo", "both"]:
        print_separator("DPO Dataset Preparation")
        dpo_data = prepare_for_dpo(dataset)
        stats = get_dataset_statistics(dpo_data)

        print(f"Prepared {stats['num_examples']} examples for DPO")
        print(f"Columns: {stats['columns']}")
        if "prompt_length" in stats:
            print(f"Prompt length - Mean: {stats['prompt_length']['mean']:.0f}")
            print(f"Chosen length - Mean: {stats['chosen_length']['mean']:.0f}")
            print(f"Rejected length - Mean: {stats['rejected_length']['mean']:.0f}")

        print("\n--- DPO Examples ---")
        for i in range(min(args.show_examples, len(dpo_data))):
            display_example(dpo_data[i], i, "dpo")

    # Save if requested
    if args.save_path:
        print_separator("Saving Dataset")
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "sft" and sft_data is not None:
            sft_data.to_json(save_path)
            print(f"Saved SFT dataset to {save_path}")
        elif args.format == "dpo" and dpo_data is not None:
            dpo_data.to_json(save_path)
            print(f"Saved DPO dataset to {save_path}")
        elif sft_data is not None and dpo_data is not None:
            # Save both
            sft_path = save_path.with_suffix(".sft.json")
            dpo_path = save_path.with_suffix(".dpo.json")
            sft_data.to_json(sft_path)
            dpo_data.to_json(dpo_path)
            print(f"Saved SFT dataset to {sft_path}")
            print(f"Saved DPO dataset to {dpo_path}")

    print_separator("Done")


if __name__ == "__main__":
    main()
