import re
from typing import List, Union


def extract_numeric_answer(text: str) -> Union[float, None]:
    """
    Extract the numeric answer from GSM8K format.
    Answers are marked with #### followed by the number.

    Args:
        text: The generated text or ground truth answer

    Returns:
        The extracted numeric value or None if not found
    """
    # Look for the #### marker followed by a number
    pattern = r'####\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # Fallback: try to extract last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None

    return None


def compute_reward(
    generated_answers: List[str],
    ground_truth_answers: List[str],
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0
) -> List[float]:
    """
    Compute rewards based on exact numeric match.

    Args:
        generated_answers: List of model-generated answers
        ground_truth_answers: List of ground truth answers from dataset
        correct_reward: Reward value for correct answers (default: 1.0)
        incorrect_reward: Reward value for incorrect answers (default: 0.0)

    Returns:
        List of reward values
    """
    rewards = []

    for gen_answer, gt_answer in zip(generated_answers, ground_truth_answers):
        gen_num = extract_numeric_answer(gen_answer)
        gt_num = extract_numeric_answer(gt_answer)

        # Check for exact match
        if gen_num is not None and gt_num is not None:
            if abs(gen_num - gt_num) < 1e-6:  # Handle floating point comparison
                rewards.append(correct_reward)
            else:
                rewards.append(incorrect_reward)
        else:
            # If we can't extract a number, give incorrect reward
            rewards.append(incorrect_reward)

    return rewards


def format_prompt(question: str) -> str:
    """
    Format a GSM8K question into a prompt for the model.

    Args:
        question: The math problem from the dataset

    Returns:
        Formatted prompt string
    """
    return f"""Solve the following math problem step by step. Show your work and provide the final answer after ####.

Question: {question}

Answer:"""


if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        {
            "generated": "She sold 48/2 = 24 clips in May. Total: 48+24 = 72 clips. #### 72",
            "ground_truth": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
            "expected_reward": 1.0
        },
        {
            "generated": "She sold 24 clips in May. Total: 70 clips. #### 70",
            "ground_truth": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
            "expected_reward": 0.0
        },
        {
            "generated": "I don't know the answer.",
            "ground_truth": "#### 72",
            "expected_reward": 0.0
        }
    ]

    print("Testing reward function:")
    for i, test in enumerate(test_cases):
        reward = compute_reward([test["generated"]], [test["ground_truth"]])[0]
        status = "✓" if reward == test["expected_reward"] else "✗"
        print(f"{status} Test {i+1}: reward={reward}, expected={test['expected_reward']}")
        print(f"  Generated: {extract_numeric_answer(test['generated'])}")
        print(f"  Ground truth: {extract_numeric_answer(test['ground_truth'])}")
        print()
