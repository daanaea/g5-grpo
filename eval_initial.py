#!/usr/bin/env python3
"""
Evaluate initial (untrained) model accuracy on GSM8K dataset.

Usage:
    python3 eval_initial.py --model_name Qwen/Qwen3-0.6B --num_samples 100
    python3 eval_initial.py --model_name Qwen/Qwen3-0.6B  # Full test set
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from reward_function import compute_reward, format_prompt


def evaluate_initial_accuracy(
    model_name="Qwen/Qwen3-4B",
    num_samples=None,
    split="test"
):
    """
    Evaluate initial model accuracy before any training.

    Args:
        model_name: HuggingFace model name or path
        num_samples: Number of samples to evaluate (None = all)
        split: Dataset split to use ("test" or "train")
    """
    print("=" * 60)
    print(f"INITIAL MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Samples: {num_samples if num_samples else 'All'}")
    print("=" * 60)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Load dataset
    print(f"Loading GSM8K {split} dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples...")

    # Evaluation
    correct = 0
    total = 0
    results = []

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]

        # Format prompt
        prompt = format_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and extract answer
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):]  # Remove prompt

        # Check correctness
        reward = compute_reward([completion], [ground_truth])[0]
        is_correct = reward > 0.5

        if is_correct:
            correct += 1

        total += 1

        # Store result
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "completion": completion,
            "correct": is_correct
        })

        # Progress update
        if (i + 1) % 10 == 0:
            current_acc = correct / total
            print(f"Progress: {i+1}/{len(dataset)} | Accuracy: {current_acc:.2%} ({correct}/{total})")

    # Final results
    accuracy = correct / total

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Examples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 60)

    # Show some examples
    print("\nSample Incorrect Predictions:")
    print("-" * 60)
    incorrect_samples = [r for r in results if not r["correct"]][:3]

    for i, sample in enumerate(incorrect_samples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Ground Truth: {sample['ground_truth'][:100]}...")
        print(f"Model Output: {sample['completion'][:100]}...")

    print("\nSample Correct Predictions:")
    print("-" * 60)
    correct_samples = [r for r in results if r["correct"]][:3]

    for i, sample in enumerate(correct_samples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Ground Truth: {sample['ground_truth'][:100]}...")
        print(f"Model Output: {sample['completion'][:100]}...")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate initial model accuracy on GSM8K"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model name or path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test on 50 samples"
    )

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.num_samples = 50
        print("Quick mode: evaluating on 50 samples")

    # Run evaluation
    accuracy, results = evaluate_initial_accuracy(
        model_name=args.model_name,
        num_samples=args.num_samples,
        split=args.split
    )

    return accuracy


if __name__ == "__main__":
    main()
