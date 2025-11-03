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
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from reward_function import compute_reward, format_prompt, extract_numeric_answer
import numpy as np


def load_gsm8k_dataset(split="train", max_samples=None):
    """Load and format the GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Format the dataset for training
    def format_example(example):
        prompt = format_prompt(example["question"])
        # For SFT, we want the full answer
        completion = example["answer"]
        return {
            "text": prompt + " " + completion,
            "prompt": prompt,
            "answer": completion,
            "question": example["question"]
        }

    formatted_dataset = dataset.map(format_example)
    return formatted_dataset


def setup_model_and_tokenizer(model_name="Qwen/Qwen3-0.6B", use_4bit=False):
    """
    Setup the model and tokenizer with optional 4-bit quantization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization for memory efficiency on g5.2xlarge
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
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
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
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

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step_end_time = time.time()
        time_per_step = step_end_time - self.step_start_time if self.step_start_time else 0

        log_entry = {
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            "step": state.global_step,
            "epoch": state.epoch if state.epoch is not None else 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
            "grpo_loss": logs.get("loss", 0.0),
            "policy_loss": logs.get("policy_loss", logs.get("loss", 0.0)),
            "kl_loss": logs.get("kl", 0.0),
            "reward": logs.get("reward"),

            "reward_mean": logs.get("reward/default/mean", logs.get("reward_mean", None)),
            "reward_std": logs.get("reward/default/std", logs.get("reward_std", None)),
            
            "adv_mean": logs.get("advantages/mean", logs.get("adv_mean", None)),
            "adv_std": logs.get("advantages/std", logs.get("adv_std", None)),
            
            "grad_norm": logs.get("grad_norm", 0.0),
            "learning_rate": logs.get("learning_rate", 0.0),
            "time_per_step_s": time_per_step,
            "generation_time_s": logs.get("generation_time", self.generation_time),
            "backward_time_s": logs.get("backward_time", self.backward_time),
            "other_time_s": time_per_step - logs.get("generation_time", 0) - logs.get("backward_time", 0),
            "tokens_generated": logs.get("tokens_generated", 0),
        }

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also print to console for visibility
        print(json.dumps(log_entry, indent=2))


def train_with_grpo(
    model_path="Qwen/Qwen3-0.6B",
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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    train_dataset = load_gsm8k_dataset("train", max_samples=max_samples)

    def reward_function(prompts, completions, **kwargs):
        """
        GRPO reward function that compares generated completions with ground truth.
        Returns a list of rewards.
        """
        # # random reward for testing
        # return [np.random.uniform(0, 1) for _ in completions]

        rewards = []

        for i, completion in enumerate(completions):
            gt_answer = dataset[i]["answer"]
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

        max_steps=100,
        logging_steps=5,
        save_steps=50,
        
        max_prompt_length=128,
        max_completion_length=400,

        num_generations=4,
        generation_batch_size=4,
        
        temperature=1.0,
        beta=0.1,
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

    print(f"Starting GRPO training... Logs will be saved to {log_file}")
    grpo_trainer.train()

    # Save the final model
    grpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO model saved to {output_dir}")
    print(f"Training logs saved to {log_file}")

    return model, tokenizer


def evaluate_model(model_path, num_samples=100):
    """Evaluate the model on GSM8K test set."""
    print("=" * 50)
    print("Evaluating Model")
    print("=" * 50)

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
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):]  # Remove prompt

        # Check if answer is correct
        reward = compute_reward([generated], [example["answer"]])[0]
        if reward > 0.5:
            correct += 1
        total += 1

        if total % 10 == 0:
            print(f"Evaluated {total}/{num_samples}, Accuracy: {correct/total:.2%}")

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen on GSM8K with GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--mode", type=str, choices=["sft", "grpo", "both", "eval"], default="both", help="Training mode")
    parser.add_argument("--sft_output", type=str, default="./qwen_gsm8k_sft", help="SFT output directory")
    parser.add_argument("--grpo_output", type=str, default="./qwen_gsm8k_grpo", help="GRPO output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="SFT training batch size")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="Use 4-bit quantization")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (for testing)")

    args = parser.parse_args()

    if args.mode == "sft" or args.mode == "both":
        model, tokenizer = train_sft(
            model_name=args.model_name,
            output_dir=args.sft_output,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            use_4bit=args.use_4bit,
            max_samples=args.max_samples,
        )

    if args.mode == "grpo" or args.mode == "both":
        # Use SFT output if training both, otherwise use base model
        model_path = args.sft_output if args.mode == "both" else args.model_name
        model, tokenizer = train_with_grpo(
            model_path=model_path,
            output_dir=args.grpo_output,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
        )

    if args.mode == "eval":
        # Determine which model to evaluate
        if os.path.exists(args.grpo_output):
            evaluate_model(args.grpo_output)
        elif os.path.exists(args.sft_output):
            evaluate_model(args.sft_output)
        else:
            print(f"Error: No trained model found at {args.grpo_output} or {args.sft_output}")
            print("Please train a model first with --mode sft, --mode grpo, or --mode both")
