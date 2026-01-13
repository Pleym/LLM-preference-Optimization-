"""
Supervised Fine-Tuning (SFT) Trainer for Mistral-7B.

This script trains the model on the 'chosen' responses from the preference dataset.
SFT serves as the baseline for comparing against DPO.

Key components:
- QLoRA for memory-efficient training on consumer GPUs
- TRL's SFTTrainer for standardized training loop
- Wandb integration for experiment tracking
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "data/train.json"
OUTPUT_DIR = "outputs/sft-mistral-7b"

# QLoRA Configuration
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

# Training Configuration
TRAINING_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "bf16": True,  # Use bfloat16 mixed precision
    "max_seq_length": 2048,
    "packing": False,  # Don't pack sequences for cleaner training
    "report_to": "wandb",  # Enable Wandb logging
}


# =============================================================================
# Data Loading
# =============================================================================


def load_training_data(data_path: str):
    """
    Load and prepare the dataset for SFT.

    For SFT, we only use the 'chosen' responses - concatenating
    the prompt with the chosen response to create the training text.
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_for_sft(example):
        """Format as: prompt + chosen response"""
        # Reconstruct the full conversation with the chosen response
        text = f"{example['prompt']}\n\nAssistant: {example['chosen']}"
        return {"text": text}

    dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)
    print(f"Loaded {len(dataset)} examples for SFT training")

    return dataset


# =============================================================================
# Model Setup
# =============================================================================


def setup_model_and_tokenizer(model_name: str):
    """
    Load model with 4-bit quantization (QLoRA) and prepare for training.
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
    tokenizer.padding_side = "right"

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
    """Main training function."""
    print("=" * 60)
    print("SFT Training: Mistral-7B with QLoRA")
    print("=" * 60)

    # Load data
    dataset = load_training_data(DATA_PATH)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # Training arguments
    training_args = SFTConfig(**TRAINING_CONFIG)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")


if __name__ == "__main__":
    train()
