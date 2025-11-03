import os
import json
import time
from datetime import datetime

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)

from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from reward_function import compute_reward, format_prompt

import numpy as np


def load_gsm8k_dataset(split="train", max_samples=None):
    """Load and format the GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_example(example):
        prompt = format_prompt(example["question"])
        completion = example["answer"]
        return {
            "text": prompt + " " + completion,
            "prompt": prompt,
            "completion": completion,  # Changed from "answer" to "completion" for SFTTrainer
            "answer": completion,  # Keep "answer" for GRPO compatibility
            "question": example["question"]
        }

    formatted_dataset = dataset.map(format_example)
    return formatted_dataset


def setup_model_and_tokenizer(model_name="Qwen/Qwen3-4B", use_4bit=False):
    """
    Setup the model and tokenizer with optional 4-bit quantization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer


def setup_lora_config():
    """Configure LoRA for parameter-efficient fine-tuning."""
    return LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


class GRPOLoggingCallback(TrainerCallback):
    """Custom callback for detailed GRPO logging."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.step_start_time = None
        self.generation_time = 0
        self.backward_time = 0
        self.first_log = True

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step_end_time = time.time()
        time_per_step = step_end_time - self.step_start_time if self.step_start_time else 0

        # Extract metrics
        step = state.global_step
        loss = logs.get("loss", 0.0)
        reward_mean = logs.get("reward/default/mean", logs.get("reward_mean", 0.0))
        reward_std = logs.get("reward/default/std", logs.get("reward_std", 0.0))
        kl = logs.get("kl", 0.0)

        # Calculate average completion length if tokens_generated is available
        tokens_gen = logs.get("tokens_generated", 0)
        # Assuming num_generations=4 from config
        avg_completion_length = tokens_gen / 4 if tokens_gen > 0 else 0

        # Print header on first log
        if self.first_log:
            print("\n" + "=" * 100)
            print(f"{'Step':<8} | {'Loss':<12} | {'Reward':<12} | {'Reward Std':<12} | {'Completion Len':<15} | {'KL':<12}")
            print("=" * 100)
            self.first_log = False

        # Print compact table row
        print(f"{step:<8} | {loss:<12.6f} | {reward_mean:<12.6f} | {reward_std:<12.6f} | {avg_completion_length:<15.2f} | {kl:<12.6f}")

        # Full log entry for JSONL file
        log_entry = {
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "epoch": state.epoch if state.epoch is not None else 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
            "grpo_loss": loss,
            "policy_loss": logs.get("policy_loss", loss),
            "kl_loss": kl,
            "reward": logs.get("reward"),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "adv_mean": logs.get("advantages/mean", logs.get("adv_mean", None)),
            "adv_std": logs.get("advantages/std", logs.get("adv_std", None)),
            "grad_norm": logs.get("grad_norm", 0.0),
            "learning_rate": logs.get("learning_rate", 0.0),
            "time_per_step_s": time_per_step,
            "generation_time_s": logs.get("generation_time", self.generation_time),
            "backward_time_s": logs.get("backward_time", self.backward_time),
            "other_time_s": time_per_step - logs.get("generation_time", 0) - logs.get("backward_time", 0),
            "tokens_generated": tokens_gen,
            "avg_completion_length": avg_completion_length,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def train_with_sft(
    model_path="Qwen/Qwen3-4B",
    output_dir="./qwen_gsm8k_sft",
    num_epochs=3,
    batch_size=4,
    max_samples=None,
    use_4bit=False,
):
    """
    Supervised Fine-Tuning (SFT) to teach the model GSM8K answer format.

    This pre-training phase teaches the base model:
    - How to structure mathematical reasoning
    - When to output the #### marker
    - When to stop generating (EOS token usage)
    """
    print("=" * 60)
    print("STARTING SFT TRAINING")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_path, use_4bit=use_4bit)

    # Apply LoRA for parameter-efficient training
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    print("\nLoRA adapter applied:")
    model.print_trainable_parameters()

    train_dataset = load_gsm8k_dataset("train", max_samples=max_samples)

    sft_trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=500,
            bf16=True,
            gradient_accumulation_steps=4,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting SFT training...")
    sft_trainer.train()

    sft_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSFT model saved to {output_dir}")
    print("=" * 60)

    return model, tokenizer


def train_with_grpo(
    model_path="Qwen/Qwen3-4B",
    output_dir="./qwen_gsm8k_grpo",
    max_samples=None,
    use_4bit=False,
):
    """
    GRPO training with custom reward function
    """
    log_file = os.path.join(output_dir, "grpo_training_logs.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    train_dataset = load_gsm8k_dataset("train", max_samples=max_samples)

    # Create a mapping from prompts to ground truth answers
    prompt_to_answer = {}
    for example in train_dataset:
        prompt_to_answer[example["prompt"]] = example["answer"]

    # Debug counter
    debug_count = [0]

    def reward_function(prompts, completions, **kwargs):
        """
        GRPO reward function that compares generated completions with ground truth.

        Args:
            prompts: List of prompt strings (batch_size × num_generations items)
            completions: List of completion strings

        Returns:
            List of rewards (0.0 or 1.0 for each completion)
        """
        rewards = []

        # Debug: print first completion every 10 calls
        if debug_count[0] % 10 == 0:
            print("\n" + "="*80)
            print("DEBUG: Sample prompt and completion")
            print("="*80)
            if len(prompts) > 0:
                print(f"Prompt: {prompts[0][:150]}...")
            if len(completions) > 0:
                print(f"Completion: {completions[0][:200]}...")
                print(f"Has ####: {'####' in completions[0]}")
            print("="*80 + "\n")
        debug_count[0] += 1

        # Process each completion
        for i, completion in enumerate(completions):
            # Get corresponding prompt (GRPO repeats prompts for each generation)
            # With batch_size=4 and num_generations=4, we have 16 prompts/completions
            # prompts = [p1, p1, p1, p1, p2, p2, p2, p2, p3, p3, p3, p3, p4, p4, p4, p4]
            prompt = prompts[i] if i < len(prompts) else prompts[0]

            # Get ground truth answer for this prompt
            gt_answer = prompt_to_answer.get(prompt, "#### 0")

            # Compute reward
            reward = compute_reward([completion], [gt_answer])[0]
            rewards.append(reward)

        return rewards
    
    

    grpo_config = GRPOConfig(
        output_dir=output_dir,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=4,

        learning_rate=1e-6,
        num_train_epochs=1,

        max_steps=500,
        logging_steps=1,
        save_steps=500,

        max_prompt_length=128,
        max_completion_length=128,

        num_generations=4,
        generation_batch_size=16,

        temperature=0.7,
        beta=0.01,
        epsilon=0.2,

        dataloader_num_workers=2,
        bf16=True,
        seed=42,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=lambda prompts, completions, **kwargs: reward_function(prompts, completions, dataset=train_dataset),
        args=grpo_config,
        train_dataset=train_dataset,

        processing_class=tokenizer,

        callbacks=[GRPOLoggingCallback(log_file)],
    )

    grpo_trainer.train()

    grpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO model saved to {output_dir}")
    print(f"Training logs saved to {log_file}")

    return model, tokenizer


def evaluate_model(model_path, num_samples=100):
    """Evaluate the model on GSM8K test set."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    test_dataset = load_gsm8k_dataset("test", max_samples=num_samples)

    correct = 0
    total = 0

    for example in test_dataset:
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):] 
        
        reward = compute_reward([generated], [example["answer"]])[0]
        if reward > 0.5:
            correct += 1
        total += 1

        if total % 10 == 0:
            print(f"Evaluated {total}/{num_samples}\nAccuracy: {correct/total:.2%}")

    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen on GSM8K with SFT and GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name or path")
    parser.add_argument("--mode", type=str, choices=["sft", "grpo", "both", "eval"], default="both", help="Training mode")
    parser.add_argument("--sft_output", type=str, default="./qwen_gsm8k_sft", help="SFT output directory")
    parser.add_argument("--grpo_output", type=str, default="./qwen_gsm8k_grpo", help="GRPO output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="SFT training batch size")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="Use 4-bit quantization")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (for testing)")

    args = parser.parse_args()

    # Evaluation mode
    if args.mode == "eval":
        if os.path.exists(args.grpo_output):
            print(f"Evaluating GRPO model from {args.grpo_output}")
            evaluate_model(args.grpo_output)
        elif os.path.exists(args.sft_output):
            print(f"Evaluating SFT model from {args.sft_output}")
            evaluate_model(args.sft_output)
        else:
            print(f"Error: No trained model found at {args.grpo_output} or {args.sft_output}")

    # SFT training mode
    elif args.mode == "sft":
        print("Mode: SFT only")
        model, tokenizer = train_with_sft(
            model_path=args.model_name,
            output_dir=args.sft_output,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
        )

    # GRPO training mode (assumes SFT model exists, or uses base model)
    elif args.mode == "grpo":
        print("Mode: GRPO only")
        # Use SFT model if it exists, otherwise use base model
        if os.path.exists(args.sft_output):
            model_path = args.sft_output
            print(f"Using SFT-trained model from {model_path}")
        else:
            model_path = args.model_name
            print(f"Using base model {model_path} (no SFT model found)")

        model, tokenizer = train_with_grpo(
            model_path=model_path,
            output_dir=args.grpo_output,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
        )

    # Both SFT + GRPO mode
    elif args.mode == "both":
        print("Mode: SFT + GRPO")

        # Step 1: SFT training
        print("\n" + "=" * 80)
        print("PHASE 1: SUPERVISED FINE-TUNING (SFT)")
        print("=" * 80)
        sft_model, sft_tokenizer = train_with_sft(
            model_path=args.model_name,
            output_dir=args.sft_output,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
        )

        # Step 2: GRPO training on SFT model
        print("\n" + "=" * 80)
        print("PHASE 2: GRPO TRAINING")
        print("=" * 80)
        grpo_model, grpo_tokenizer = train_with_grpo(
            model_path=args.sft_output,  # Use SFT model as starting point
            output_dir=args.grpo_output,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"SFT model: {args.sft_output}")
        print(f"GRPO model: {args.grpo_output}")
        print("=" * 80)