"""
Direct Preference Optimization (DPO) Trainer for Mistral-7B.

This script trains the model using preference pairs (chosen vs rejected).
DPO directly optimizes the policy to prefer chosen responses over rejected ones,
without requiring a separate reward model.

Key components:
- QLoRA for memory-efficient training on consumer GPUs
- TRL's DPOTrainer for preference optimization
- Reference model (frozen) for KL-divergence constraint
- Wandb integration for experiment tracking
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "data/train.json"
OUTPUT_DIR = "outputs/dpo-mistral-7b"

# QLoRA Configuration (same as SFT for fair comparison)
QLORA_CONFIG = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,  # LoRA alpha scaling
    "lora_dropout": 0.05,  # Dropout for LoRA layers
    "target_modules": [  # Modules to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# DPO Training Configuration
DPO_TRAINING_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,  # Smaller batch size (DPO needs more memory)
    "gradient_accumulation_steps": 8,  # Compensate with more accumulation
    "learning_rate": 5e-5,  # Lower LR than SFT (DPO is more sensitive)
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "bf16": True,
    "max_length": 2048,  # Max total sequence length
    "max_prompt_length": 1024,  # Max prompt length
    "beta": 0.1,  # DPO beta (KL penalty coefficient)
    "loss_type": "sigmoid",  # DPO loss type
    "report_to": "wandb",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_preference_data(data_path: str):
    """
    Load the preference dataset for DPO training.

    DPO expects: prompt, chosen, rejected
    The dataset is already in this format.
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Verify required columns
    required_cols = {"prompt", "chosen", "rejected"}
    if not required_cols.issubset(set(dataset.column_names)):
        raise ValueError(f"Dataset must have columns: {required_cols}")

    print(f"Loaded {len(dataset)} preference pairs for DPO training")

    # Show a sample
    print("\n--- Sample preference pair ---")
    sample = dataset[0]
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Chosen: {sample['chosen'][:50]}...")
    print(f"Rejected: {sample['rejected'][:50]}...")

    return dataset


# =============================================================================
# Model Setup
# =============================================================================


def setup_model_and_tokenizer(model_name: str):
    """
    Load model with 4-bit quantization (QLoRA) and prepare for DPO training.

    Note: DPO uses a reference model internally. With PEFT, the reference
    model shares weights with the policy model, saving memory.
    """
    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(**QLORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================


def train():
    """Main DPO training function."""
    print("=" * 60)
    print("DPO Training: Mistral-7B with QLoRA")
    print("=" * 60)

    # Load preference data
    dataset = load_preference_data(DATA_PATH)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # DPO training arguments
    training_args = DPOConfig(**DPO_TRAINING_CONFIG)

    # Initialize DPO trainer
    # Note: ref_model=None means TRL will use the base model (before LoRA)
    # as the reference, which is memory efficient with PEFT
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference with PEFT
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    print(f"Beta (KL penalty): {training_args.beta}")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("DPO Training complete!")


if __name__ == "__main__":
    train()
