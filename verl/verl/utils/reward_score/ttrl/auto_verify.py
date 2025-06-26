from collections import defaultdict

from tqdm import tqdm

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.qwen.qwen_eval import (qwen_reward_fn,
                                                         qwen_reward_fn_gpqa,
                                                         simplerl_reward_fn)


def auto_verify(task, all_outputs, all_labels, extra_info=None, extended_info=None):

    task2verify = {
        "math": qwen_reward_fn,
        "simplerl_math": simplerl_reward_fn,
        "gpqa": qwen_reward_fn_gpqa,
    }
    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"
    verify_fn = task2verify[task]
    verify_extra_info = defaultdict(list)

    if extended_info is not None:
        rewards = [verify_fn(output, label, extended_info=extended) for output, label, extended in zip(all_outputs, all_labels, extended_info)]
    else:
        rewards = [verify_fn(output, label) for output, label in zip(all_outputs, all_labels)]
    
    verify_extra_info["acc"] = rewards

    verify_extra_info["pred"] = auto_extract(task, all_outputs, extra_info=extra_info)
        
    return rewards, verify_extra_info