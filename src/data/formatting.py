"""
Conversation formatting utilities for Anthropic HH-RLHF dataset.

The Anthropic HH dataset uses a specific format with "Human:" and "Assistant:" prefixes.
This module converts that format into standard chat message structures compatible
with modern LLM chat templates.
"""

import re
from typing import Optional


def parse_conversation(text: str) -> list[dict[str, str]]:
    """
    Parse Anthropic HH-RLHF conversation format into message list.

    The HH dataset format looks like:
        Human: Hello, can you help me?
        Assistant: Of course! How can I assist you today?
        Human: I need advice on...

    Args:
        text: Raw conversation string from the dataset

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    # Split on Human:/Assistant: markers while keeping the marker
    pattern = r"\n\n(?=Human:|Assistant:)"
    turns = re.split(pattern, text.strip())

    messages = []
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue

        if turn.startswith("Human:"):
            content = turn[len("Human:") :].strip()
            messages.append({"role": "user", "content": content})
        elif turn.startswith("Assistant:"):
            content = turn[len("Assistant:") :].strip()
            messages.append({"role": "assistant", "content": content})

    return messages


def format_conversation(text: str) -> dict[str, str]:
    """
    Convert raw HH conversation into structured prompt/response format.

    This extracts:
    - prompt: All messages up to (not including) the final assistant response
    - response: The final assistant response

    Args:
        text: Raw conversation string from the dataset

    Returns:
        Dict with 'prompt' (full conversation context) and 'response' (final answer)
    """
    messages = parse_conversation(text)

    if not messages:
        return {"prompt": "", "response": ""}

    # Find the last assistant response
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        # No assistant response found
        return {"prompt": reconstruct_conversation(messages), "response": ""}

    prompt_messages = messages[:last_assistant_idx]
    response = messages[last_assistant_idx]["content"]

    return {"prompt": reconstruct_conversation(prompt_messages), "response": response}


def reconstruct_conversation(messages: list[dict[str, str]]) -> str:
    """
    Reconstruct conversation string from message list.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Formatted conversation string
    """
    parts = []
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    return "\n\n".join(parts)


def apply_chat_template(
    messages: list[dict[str, str]], tokenizer, add_generation_prompt: bool = False
) -> str:
    """
    Apply the tokenizer's chat template to format messages.

    This uses the model's native chat template (e.g., Mistral's [INST] format)
    to ensure proper formatting during training and inference.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer with chat_template
        add_generation_prompt: Whether to add the generation prompt suffix

    Returns:
        Formatted string ready for tokenization
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def extract_prompt_and_responses(chosen: str, rejected: str) -> dict[str, str]:
    """
    Extract the common prompt and differing responses from chosen/rejected pairs.

    In the HH dataset, chosen and rejected share the same conversation context
    but differ in the final assistant response. This function extracts:
    - prompt: The shared conversation context (including the human query)
    - chosen_response: The preferred assistant response
    - rejected_response: The rejected assistant response

    Args:
        chosen: Full conversation with the preferred response
        rejected: Full conversation with the rejected response

    Returns:
        Dict with 'prompt', 'chosen_response', and 'rejected_response'
    """
    chosen_parsed = format_conversation(chosen)
    rejected_parsed = format_conversation(rejected)

    # The prompts should be identical; use chosen's prompt
    return {
        "prompt": chosen_parsed["prompt"],
        "chosen_response": chosen_parsed["response"],
        "rejected_response": rejected_parsed["response"],
    }
