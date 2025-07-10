import concurrent.futures
from collections import Counter
from typing import List

from verl.utils.reward_score.ttrl.qwen.grader import \
    math_equal as qwen_math_equal
from verl.utils.reward_score.ttrl.qwen.math_grade import grade_answer
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer
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

def qwen_reward_fn(generated_text, golden_answer, task="math"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy


def qwen_reward_fn_gpqa(generated_text, golden_answer, task="gpqa"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

# ------------------------------------------------------------
# import math

# def match_answer(response):
#     is_matched = False
#     ans_marker = 'The answer is: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer:\n'
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     # Find boxed
#     ans_boxed = _last_boxed_only_string(response)
#     if ans_boxed:
#         is_matched = True
#         response = ans_boxed

#     # Grade
#     return is_matched, response


# def _last_boxed_only_string(string):
#     idx = string.rfind("\\boxed")
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     left_brace_idx = None
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#             if left_brace_idx is None:
#                 left_brace_idx = i
#         elif string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break

#         i += 1

#     if left_brace_idx is None or right_brace_idx is None:
#         return None

#     return string[left_brace_idx + 1: right_brace_idx].strip()

# def qwen_reward_fn(generated_text, golden_answer, task="math"):
#     if golden_answer in ["A", "B", "C", "D"]:
#         task = "gpqa"
#         model_answer = extract_answer(generated_text, task)
#         accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
#         # if "boxed" not in generated_text:
#         #     accuracy = -1.0
#         return accuracy

#     answer = golden_answer.lstrip('0') 
#     is_matched, model_output = match_answer(generated_text)
#     model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
#     try:
#         if "\pi" in model_output or "\pi" in golden_answer:
#             equivs = []
#             for pi in [math.pi, 3.14]:
#                 equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
#             equiv = any(equivs)
#         else:
#             equiv = math_equal(model_output, answer, timeout=True)
#     except:
#         equiv = False

#     if equiv:
#         return 1.0
#     else:
#         return 0.0

# ------------------------------------------------------------

def majority_vote(
    solutions: List[str],
    ground_truth: str,
    task="math"
):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]

    if len(model_answers) == 0:
        return 0
    
    counter = Counter(model_answers)
    
    majority_answer, _ = counter.most_common(1)[0]
    accuracy = 1.0 if grade_answer(majority_answer, ground_truth) else 0
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(qwen_math_equal, prediction=prediction, reference=reference, timeout=False)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False

def simplerl_reward_fn(generated_text, golden_answer):
    model_answer = extract_answer(generated_text, "math")
    accuracy = 1.0 if qwen_math_equal_subprocess(prediction=model_answer, reference=golden_answer) else -0.5
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy

def qwen_reward_fn(generated_text, golden_answer, extended_info=None, task="math"):
    results = {}
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    results["accuracy"] = accuracy

    if extended_info is not None:
        if "boxed" not in generated_text:
            results["boxed"] = -1
            # accuracy -= 1
        else:
            results["boxed"] = 0

        # if "```python" in generated_text:
        #     accuracy -= 0.5
        #     results["python"] = -0.5
        # else:
        #     results["python"] = 0

        results["compression_ratio"] = (3.5 - get_compression_ratio(generated_text))/3.5 # least on AIME was 1

        if "prompt_length" in extended_info and "response_length" in extended_info:
            response_prompt_length_ratio = ((extended_info["response_length"] / extended_info["prompt_length"]) - 5) / 5
            response_prompt_length_ratio = -max(0, response_prompt_length_ratio)
            # accuracy -= response_prompt_length_ratio
            results["response_prompt_length_ratio"] = response_prompt_length_ratio
    
    # pass the accuracy as a dictionary and sum it up later
    # that will also help with minimizing the number of calls to the grading function
    # return accuracy
    return results