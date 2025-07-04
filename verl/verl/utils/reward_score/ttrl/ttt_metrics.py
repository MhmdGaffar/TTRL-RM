from collections import Counter
from typing import List

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify

import zlib

def get_compression_ratio(text, encoding='utf-8', level=6):
    """
    Computes the compression ratio (original_size / compressed_size) for a given text.
    
    Args:
        text (str): Input text to compress.
        encoding (str): Text encoding (default: 'utf-8').
        level (int): zlib compression level (0-9, default=6).
    
    Returns:
        float: Compression ratio. Returns 0.0 for empty input.
    """
    # Encode text to bytes and get original size
    original_bytes = text.encode(encoding)
    original_size = len(original_bytes)
    
    # Handle empty input
    if original_size == 0:
        return 0.0
    
    # Compress data and get compressed size
    compressed_bytes = zlib.compress(original_bytes, level=level)
    compressed_size = len(compressed_bytes)
    
    # Compute compression ratio
    return original_size / compressed_size


def test_time_train_metrics(
        solutions: List[str],
        ground_truth: List[str],
        task="math", 
        extra_info=None, 
        extended_info=None,
        return_majority_rewards=False
    ):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0]["accuracy"] else 0.0
    majority_ratio = majority_count / len(solutions)
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    calculated_rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info, extended_info=extended_info)
    rewards = [sum([v for k,v in i.items()]) for i in calculated_rewards]
    # convert to a dictionary of lists
    calculated_rewards = {k: [d[k] for d in calculated_rewards] for k in calculated_rewards[0].keys()}
    majority_rewards = calculated_rewards["accuracy"]
    # get the avg for calculated_rewards
    calculated_rewards_avg = {k: sum(v) / len(v) for k, v in calculated_rewards.items()}
    
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    true_rewards = [i["accuracy"] for i in true_rewards]

    reward_avg = sum(rewards) / len(rewards)
    majority_reward_avg = sum(majority_rewards) / len(majority_rewards)
    extra_reward_avg = reward_avg - majority_reward_avg

    rewards_hit_rate = 0
    for reward, true_reward in zip(majority_rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "reward": reward_avg,
        "majority_voting_reward": majority_reward_avg,
        "extra_rewards": extra_reward_avg,
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
        **calculated_rewards_avg,
    }
    if return_majority_rewards:
        ttrl_metrics["majority_rewards"] = majority_rewards
    return rewards, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    # true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    true_rewards = [i["accuracy"] for i in true_rewards]

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics