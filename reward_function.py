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
    pattern = r'####\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
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

        if gen_num is not None and gt_num is not None:
            if abs(gen_num - gt_num) < 1e-6:
                rewards.append(correct_reward)
            else:
                rewards.append(incorrect_reward)
        else:
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
    return f"""
        Solve the following math problem step by step. Show your work and provide the final answer after ####.

        Question: {question}

        Answer:"""
