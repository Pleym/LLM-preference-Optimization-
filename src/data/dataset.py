"""
Dataset loading and preparation for DPO and SFT training.

This module provides utilities for loading the Anthropic HH-RLHF dataset
and preparing it for both SFT (Supervised Fine-Tuning) and DPO (Direct
Preference Optimization) training paradigms.

Dataset: Anthropic/hh-rlhf
- Contains ~170k preference pairs
- Each example has 'chosen' and 'rejected' conversations
- Used to train helpful and harmless AI assistants
"""

from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict
from .formatting import (
    format_conversation,
    extract_prompt_and_responses,
    parse_conversation,
    apply_chat_template,
)


def load_anthropic_hh(
    split: Optional[str] = None,
    subset: Optional[str] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> Dataset | DatasetDict:
    """
    Load the Anthropic HH-RLHF dataset from HuggingFace Hub.

    Args:
        split: Dataset split to load ('train', 'test', or None for all)
        subset: 'helpful-base', 'helpful-online', 'helpful-rejection-sampled',
                'harmless-base', or None for all subsets
        streaming: Whether to stream the dataset (memory efficient for large data)
        max_samples: Maximum number of samples to load (useful for debugging)

    Returns:
        HuggingFace Dataset or DatasetDict

    Example:
        >>> dataset = load_anthropic_hh(split="train", max_samples=1000)
        >>> print(dataset[0].keys())
        dict_keys(['chosen', 'rejected'])
    """
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir=subset,
        split=split,
        streaming=streaming,
    )

    if max_samples is not None and not streaming:
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict(
                {
                    k: v.select(range(min(max_samples, len(v))))
                    for k, v in dataset.items()
                }
            )
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def prepare_for_sft(
    dataset: Dataset,
    tokenizer=None,
    max_length: Optional[int] = 2048,
    use_chat_template: bool = True,
) -> Dataset:
    """
    Prepare dataset for Supervised Fine-Tuning (SFT).

    SFT trains on the 'chosen' responses only, teaching the model
    to generate preferred outputs given the conversation context.

    The output format follows TRL's SFTTrainer expectations:
    - 'text': The full formatted conversation (prompt + response)
    - 'messages': List of message dicts (for chat template application)

    Args:
        dataset: HuggingFace Dataset with 'chosen' column
        tokenizer: Optional tokenizer for applying chat template
        max_length: Maximum sequence length (for filtering)
        use_chat_template: Whether to apply model's chat template

    Returns:
        Dataset ready for SFTTrainer

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> sft_data = prepare_for_sft(dataset, tokenizer)
    """

    def process_example(example):
        # Parse the chosen conversation
        messages = parse_conversation(example["chosen"])

        if use_chat_template and tokenizer is not None:
            text = apply_chat_template(messages, tokenizer)
        else:
            # Fallback: reconstruct as plain text
            parsed = format_conversation(example["chosen"])
            text = f"{parsed['prompt']}\n\nAssistant: {parsed['response']}"

        return {
            "text": text,
            "messages": messages,
        }

    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Preparing SFT dataset",
    )

    # Filter by length if tokenizer provided
    if tokenizer is not None and max_length is not None:

        def filter_by_length(example):
            tokens = tokenizer(example["text"], truncation=False)
            return len(tokens["input_ids"]) <= max_length

        processed = processed.filter(
            filter_by_length,
            desc="Filtering by length",
        )

    return processed


def prepare_for_dpo(
    dataset: Dataset,
    tokenizer=None,
    max_length: Optional[int] = 2048,
    max_prompt_length: Optional[int] = 1024,
) -> Dataset:
    """
    Prepare dataset for Direct Preference Optimization (DPO).

    DPO requires preference pairs: (prompt, chosen_response, rejected_response).
    The model learns to prefer the chosen response over the rejected one.

    The output format follows TRL's DPOTrainer expectations:
    - 'prompt': The conversation context (all messages before final response)
    - 'chosen': The preferred response
    - 'rejected': The rejected response

    Args:
        dataset: HuggingFace Dataset with 'chosen' and 'rejected' columns
        tokenizer: Optional tokenizer for length filtering
        max_length: Maximum total sequence length
        max_prompt_length: Maximum prompt length

    Returns:
        Dataset ready for DPOTrainer

    Example:
        >>> dpo_data = prepare_for_dpo(dataset)
        >>> print(dpo_data[0].keys())
        dict_keys(['prompt', 'chosen', 'rejected'])
    """

    def process_example(example):
        extracted = extract_prompt_and_responses(example["chosen"], example["rejected"])

        return {
            "prompt": extracted["prompt"],
            "chosen": extracted["chosen_response"],
            "rejected": extracted["rejected_response"],
        }

    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Preparing DPO dataset",
    )

    # Filter out empty responses
    processed = processed.filter(
        lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0,
        desc="Filtering empty responses",
    )

    # Filter by length if tokenizer provided
    if tokenizer is not None and max_length is not None:

        def filter_by_length(example):
            prompt_tokens = tokenizer(example["prompt"], truncation=False)
            chosen_tokens = tokenizer(example["chosen"], truncation=False)
            rejected_tokens = tokenizer(example["rejected"], truncation=False)

            prompt_len = len(prompt_tokens["input_ids"])
            chosen_len = len(chosen_tokens["input_ids"])
            rejected_len = len(rejected_tokens["input_ids"])

            # Check both combinations fit within max_length
            chosen_total = prompt_len + chosen_len
            rejected_total = prompt_len + rejected_len

            prompt_ok = max_prompt_length is None or prompt_len <= max_prompt_length
            length_ok = chosen_total <= max_length and rejected_total <= max_length

            return prompt_ok and length_ok

        processed = processed.filter(
            filter_by_length,
            desc="Filtering by length",
        )

    return processed


def get_dataset_statistics(dataset: Dataset) -> dict:
    """
    Compute statistics about the dataset for analysis and reporting.

    Args:
        dataset: Prepared dataset (either SFT or DPO format)

    Returns:
        Dict with statistics (counts, lengths, etc.)
    """
    stats = {
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
    }

    # Compute length statistics for text columns
    if "text" in dataset.column_names:
        # SFT format
        lengths = [len(ex["text"]) for ex in dataset]
        stats["text_length"] = {
            "mean": sum(lengths) / len(lengths),
            "min": min(lengths),
            "max": max(lengths),
        }

    if "prompt" in dataset.column_names:
        # DPO format
        prompt_lens = [len(ex["prompt"]) for ex in dataset]
        chosen_lens = [len(ex["chosen"]) for ex in dataset]
        rejected_lens = [len(ex["rejected"]) for ex in dataset]

        stats["prompt_length"] = {
            "mean": sum(prompt_lens) / len(prompt_lens),
            "min": min(prompt_lens),
            "max": max(prompt_lens),
        }
        stats["chosen_length"] = {
            "mean": sum(chosen_lens) / len(chosen_lens),
            "min": min(chosen_lens),
            "max": max(chosen_lens),
        }
        stats["rejected_length"] = {
            "mean": sum(rejected_lens) / len(rejected_lens),
            "min": min(rejected_lens),
            "max": max(rejected_lens),
        }

    return stats


def create_train_val_split(
    dataset: Dataset,
    val_size: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """
    Create train/validation split from a dataset.

    Args:
        dataset: Input dataset
        val_size: Fraction to use for validation (default 5%)
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    split = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )
