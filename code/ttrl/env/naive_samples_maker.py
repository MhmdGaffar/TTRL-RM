import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import ray
import torch
from vllm import SamplingParams

from ttrl.models.model_utils import process_sequences
from ttrl.helper.logging_utils import init_logger
from ttrl.verifier.auto_verify import auto_verify

from ttrl.helper.utils import to

logger = init_logger(__name__)


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    labels: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        if self.attention_mask is not None:
            self.attention_mask = to(self.attention_mask, device)
        if self.action_mask is not None:
            self.action_mask = to(self.action_mask, device)
        if self.num_actions is not None and isinstance(self.num_actions, torch.Tensor):
            self.num_actions = to(self.num_actions, device)
        if self.packed_seq_lens is not None and isinstance(self.packed_seq_lens, torch.Tensor):
            self.packed_seq_lens = to(self.packed_seq_lens, device)
        if self.response_length is not None and isinstance(self.response_length, torch.Tensor):
            self.response_length = to(self.response_length, device)
        if self.total_length is not None and isinstance(self.total_length, torch.Tensor):
            self.total_length = to(self.total_length, device)
        if self.labels is not None:
            self.labels = to(self.labels, device)
        if self.rewards is not None:
            self.rewards = to(self.rewards, device)


class NaiveSamplesMaker(ABC):

    def __init__(self,
                 strategy,
                 tokenizer,
                 vllm_engines=None,):
        super().__init__()
        args = strategy.args
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines
        self.prompt_max_len = args.prompt_max_len
        self.generate_max_len = args.generate_max_len
        self.packing_samples = getattr(args, "packing_samples", False)

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate_samples(self, eval_data: Union[List[str], dict], **kwargs):
        args = self.strategy.args
        # sampling_params = SamplingParams(
        #     temperature=kwargs.get("eval_temperature", 0.6),
        #     top_p=kwargs.get("top_p", 0.95),
        #     top_k=kwargs.get("top_k", -1),
        #     max_tokens=kwargs.get("max_new_tokens", 3072),
        #     min_tokens=kwargs.get("min_new_tokens", 16),
        #     skip_special_tokens=kwargs.get("skip_special_tokens", False),
        # )
        
        sampling_params = SamplingParams(
            temperature=kwargs.get("eval_temperature", 0.0),
            top_p=kwargs.get("eval_top_p", 1.0),
            max_tokens=kwargs.get("max_new_tokens", 3072)
        )
        print("Debug Max Tokens:", sampling_params.max_tokens)
        print("Debug Temperature:", sampling_params.temperature)

        all_prompts, all_labels, all_indices = eval_data[
            "prompt"], eval_data["label"], eval_data["indice"]

        all_output_refs = []
        # we generate multiple outputs for each prompt for stable evaluation
        for llm in self.vllm_engines:
            all_output_ref = llm.generate.remote(
                sampling_params=sampling_params, prompts=all_prompts)
            all_output_refs.append(all_output_ref)
        
        all_outputs = ray.get(all_output_refs)

        all_accuracies = []
        verify_task = getattr(args, "verify_task_eval", args.verify_task)
        print(f"Using verification task: {verify_task}")
        print(len(all_outputs))
        for outputs in all_outputs:
            all_accuracies.append(auto_verify(verify_task, 1, outputs, all_labels))

        # print(all_accuracies)
        accuracy = np.mean([np.mean(acc) for acc in all_accuracies])
        # accuracy = np.mean(all_accuracies[0])

        metadata = []
        for prompt, label, indice in zip(all_prompts, all_labels, all_indices):
            metadata.append({"prompt": prompt, "label": label,
                            "indice": indice, "outputs": []})

        for outputs in all_outputs:
            for idx, output in enumerate(outputs):
                metadata[idx]["outputs"].append(output.outputs[0].text)

        return {"accuracy": accuracy, "metadata": metadata}

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8, **kwargs) -> List[Samples]:
        """
        Generate samples and return a list of Samples.
        """
        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        all_prompts, all_labels = all_prompts["prompt"], all_prompts["label"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i *
                                                    batch_size: (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(
                        sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])
        samples_list = []

        for i in random.sample(list(range(len(all_prompts))), k=min(3, len(all_prompts))):
            print(f"Question {i+1}:", all_prompts[i])
            print(f"Answer {i+1}:", all_outputs[i].outputs[0].text)
            print("\n\n")

        all_pred_labels = auto_verify(
            args.verify_task, args.n_samples_per_prompt, all_outputs, all_labels)
        all_pred_outputs = [output.outputs[0].text for output in all_outputs]

        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i: i +
                                  self.strategy.args.micro_rollout_batch_size]
            pred_labels = all_pred_labels[i: i +
                                          self.strategy.args.micro_rollout_batch_size]
            pred_labels = torch.tensor(
                pred_labels, device="cpu", dtype=torch.float)

            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(
                        max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(
                        output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [
                        pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(
                        output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(
                            output_ids) - 1)] = eos_token_id

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cpu")
                attention_mask = attention_mask.to("cpu")
                action_mask = action_mask.to("cpu")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        labels=pred_labels,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids +
                                     list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))
                sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
                attention_mask = torch.tensor(
                    attention_mask, device="cpu").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(
                    num_actions, device="cpu", dtype=torch.float)
                total_length = torch.tensor(
                    packed_seq_lens, device="cpu", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        labels=pred_labels,
                    )
                )
        return samples_list, all_prompts, all_pred_outputs, all_pred_labels
