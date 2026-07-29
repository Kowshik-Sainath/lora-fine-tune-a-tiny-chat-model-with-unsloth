"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""

    for module in model.modules():
        # Check by class name or module name to safely detect bitsandbytes 4-bit layers
        class_name = module.__class__.__name__
        module_path = module.__class__.__module__
        
        if "Linear4bit" in class_name or "bitsandbytes" in module_path and "4bit" in class_name.lower():
            return True

    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel

def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """
    Wrap the base model with LoRA adapters and return the PEFT model.

    Args:
        model: The 4-bit base model.
        r (int): LoRA rank.
        lora_alpha (int): LoRA scaling factor.
        target_modules (list[str], optional): Modules to apply LoRA to.

    Returns:
        The PEFT-wrapped model with LoRA adapters attached.
    """
    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "What is the capital of Australia?",
            "response": "The capital of Australia is Canberra."
        },
        {
            "instruction": "Translate 'Good morning' into Spanish.",
            "response": "Buenos días."
        },
        {
            "instruction": "Summarize: 'Machine learning enables computers to learn from data without being explicitly programmed.'",
            "response": "Machine learning allows computers to learn from data."
        },
        {
            "instruction": "Write a Python function to check whether a number is even.",
            "response": "def is_even(n):\n    return n % 2 == 0"
        },
        {
            "instruction": "Determine the sentiment: 'The service was excellent and fast.'",
            "response": "Positive"
        },
        {
            "instruction": "Explain recursion in one sentence.",
            "response": "Recursion is a technique where a function calls itself to solve smaller instances of a problem."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text (not yet solved)
# TODO: implement

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

