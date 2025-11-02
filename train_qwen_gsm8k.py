import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from reward_function import compute_reward, format_prompt, extract_numeric_answer


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


def train_sft(
    model_name="Qwen/Qwen3-0.6B",
    output_dir="./qwen_gsm8k_sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=512,
    use_4bit=True,
    max_samples=None,
):
    """
    Supervised Fine-Tuning (SFT) phase.
    This trains the model on the full reasoning chains.
    """
    print("=" * 50)
    print("Starting Supervised Fine-Tuning (SFT)")
    print("=" * 50)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit=use_4bit)

    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_dataset = load_gsm8k_dataset("train", max_samples=max_samples)

    # Training arguments optimized for g5.2xlarge (8 vCPUs, 1 GPU)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        warmup_steps=100,
        dataloader_num_workers=4,  # Utilize 4 of 8 vCPUs for data loading
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=1.0,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer


def train_with_custom_reward(
    sft_model_path="./qwen_gsm8k_sft",
    output_dir="./qwen_gsm8k_rl",
    num_train_epochs=1,
    batch_size=8,
    mini_batch_size=2,
    learning_rate=1e-5,
    max_samples=None,
):
    """
    Reinforcement Learning phase using PPO with custom reward function.
    This fine-tunes the model based on exact numeric answer matching.
    """
    print("=" * 50)
    print("Starting RL Fine-Tuning with Custom Reward")
    print("=" * 50)

    # Load the SFT model
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load dataset
    dataset = load_gsm8k_dataset("train", max_samples=max_samples)

    # PPO Configuration optimized for g5.2xlarge
    ppo_config = PPOConfig(
        model_name=sft_model_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None,
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    print("Starting PPO training...")
    for epoch in range(num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{num_train_epochs}")

        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Get prompts
            prompts = [format_prompt(q) for q in batch["question"]]
            prompt_tensors = [tokenizer.encode(p, return_tensors="pt")[0].to(model.device) for p in prompts]

            # Generate responses
            response_tensors = ppo_trainer.generate(
                prompt_tensors,
                return_prompt=False,
                **generation_kwargs
            )

            # Decode responses
            responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

            # Compute rewards using custom reward function
            rewards = compute_reward(responses, batch["answer"])
            rewards_tensors = [torch.tensor(r, dtype=torch.float32).to(model.device) for r in rewards]

            # PPO update step
            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards_tensors)

            # Logging
            if batch_idx % 10 == 0:
                avg_reward = sum(rewards) / len(rewards)
                print(f"Batch {batch_idx}: avg_reward={avg_reward:.3f}")
                ppo_trainer.log_stats(stats, batch, rewards_tensors)

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"RL model saved to {output_dir}")

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

    parser = argparse.ArgumentParser(description="Fine-tune Qwen on GSM8K with custom reward")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "both", "eval"], default="both", help="Training mode")
    parser.add_argument("--sft_output", type=str, default="./qwen_gsm8k_sft", help="SFT output directory")
    parser.add_argument("--rl_output", type=str, default="./qwen_gsm8k_rl", help="RL output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization")
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

    if args.mode == "rl" or args.mode == "both":
        model, tokenizer = train_with_custom_reward(
            sft_model_path=args.sft_output,
            output_dir=args.rl_output,
            num_train_epochs=1,
            max_samples=args.max_samples,
        )

    if args.mode == "eval":
        evaluate_model(args.rl_output if args.mode == "both" else args.sft_output)
