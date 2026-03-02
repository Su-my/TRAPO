# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List, Union, Optional, Any
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
import traceback
from torch import nn
import numpy as np

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))

# from pprint import pprint

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    else:
        token_ids = prompt_token_ids[:non_pad_index[-1][0]+1].tolist()
    return token_ids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


from verl.workers.rollout.vllm_rollout import vLLMRollout

class MIXvLLMRollout(vLLMRollout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vocab = self.tokenizer.get_vocab()
        cc_token_ids = [token_id for token, token_id in vocab.items() if "ĊĊ" in token]
        self.cc_set = set(cc_token_ids)
        # self.guidance_end_token_id = self.tokenizer.convert_tokens_to_ids("<|guidance_end|>")

        self.prefix_strategy = self.config.get('prefix_strategy', 'random')
        
        self.prefix_steps = self.config.get('prefix_steps', 300)
        self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
        if self.prefix_strategy == 'linear':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            pass
        elif self.prefix_strategy == 'linear_max':
            self.prefix_ratio_windows = [(0, i*self.prefix_linear_max_ratio/10) for i in range(10, 0, -1)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'linear_variance':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            self.prefix_lienar_max_var = self.config.get('prefix_lienar_max_var', 0.1)
        elif self.prefix_strategy == 'reverse_linear':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            self.prefix_ratio_windows = [(0, (i+1)*self.prefix_linear_max_ratio/10) for i in range(10)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'fixed':
            assert self.config.prefix_share_across_samples == False, "Fixed strategy could not work with prefix_share_across_samples=True ! "
            # self.prefix_fixed_num = self.config.get('prefix_fixed_num', 2)
            n_prefix = self.config.n_prefix if self.config.n_prefix != -1 else self.config.n
            ratio_step = (self.config.max_prefix_ratio - self.config.min_prefix_ratio) / (n_prefix-1)
            self.prefix_fix_ratios = [self.config.min_prefix_ratio + i*ratio_step for i in range(n_prefix)]
    
    def get_prefix_end_idx(self, hint_ids, ratio = 0.2) -> Optional[int]:
        matching_indice = [i for i, token_id in enumerate(hint_ids) if token_id in self.cc_set]
        threshold = int(len(hint_ids) * ratio)
        target_idx = None
        for idx in matching_indice:
            if idx > threshold:
                target_idx = idx
                break
        
        if target_idx is not None and target_idx < len(hint_ids) * (ratio+0.3):
            return target_idx + 1
        else:
            return threshold
        
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        max_retries = int(max_retries)
        for attempt in range(max_retries):
            try:
                # Rebuild vLLM cache engine if configured
                if self.config.free_cache_engine:
                    self.inference_engine.init_cache_engine()
                    
                # Extract input tensors from prompt batch
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                hint_levels = prompts.batch.get('hint_levels', None)
                eos_token_id = prompts.meta_info['eos_token_id']
                rollout_n = prompts.meta_info.get("rollout_n", None)
                batch_id = prompts.meta_info.get("batch_id", None)

                batch_size = idx.size(0)

                idx_list = [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ]

                do_sample = prompts.meta_info.get("do_sample", True)
                is_validate = prompts.meta_info.get("validate", False)
                if do_sample and is_validate == False:
                    idx_list = sum([[idx_list[i]] * rollout_n for i in range(len(idx_list))], [])

                if is_validate == False:
                    if batch_id == 1 or batch_id == 2 or batch_id == 3:
                        hint_input_ids = prompts.batch["hint_input_ids_0"]
                    else:
                        hint_input_ids = prompts.batch["hint_input_ids_1"]

                    hint_list = [
                        _pre_process_inputs_right_pad(self.pad_token_id, hint_input_ids[i]) for i in range(batch_size)
                    ]

                    hint_list = [
                        hint_list[i] + [self.tokenizer.eos_token_id,] if len(hint_list[i]) > 0 else hint_list[i]
                        for i in range(batch_size)
                    ]

                    # hint_ratio = []
                    hint_index = []
                    # we only calculate off loss for the first hint of each group
                    off_seq_loss_mask = []
                    for i in range(batch_size):
                        hint_ids = hint_list[i]
                        hint_level = hint_levels[i]
                        if hint_level == 0:
                            hint_index.extend([0] * rollout_n)
                            off_seq_loss_mask.extend([0] * rollout_n)
                        elif hint_level == 1:
                            prefix_end_idx = self.get_prefix_end_idx(hint_ids, ratio=0.25)
                            assert prefix_end_idx is not None, f"prefix_end_idx is None for hint_ids: {hint_ids}"
                            hint_index.extend([prefix_end_idx] * rollout_n)
                            assert rollout_n == 2, f"rollout_n should be 2 when hint_level is 1, but got {rollout_n}"
                            off_seq_loss_mask.extend([0, 1])
                            # if batch_id == 4:
                            #     off_seq_loss_mask.extend([1] * rollout_n)
                            # else:
                            #     off_seq_loss_mask.extend([0] * rollout_n)
                        elif hint_level == 2:
                            prefix_end_idx = self.get_prefix_end_idx(hint_ids, ratio=0.5)
                            assert prefix_end_idx is not None, f"prefix_end_idx is None for hint_ids: {hint_ids}"
                            hint_index.extend([prefix_end_idx] * rollout_n)
                            off_seq_loss_mask.extend([1] * rollout_n)
                            # if batch_id == 4:
                            #     off_seq_loss_mask.extend([1] * rollout_n)
                            # else:
                            #     off_seq_loss_mask.extend([0] * rollout_n)
                        elif hint_level == 3:
                            # prefix_end_idx = self.get_prefix_end_idx(hint_ids, ratio=0.75)
                            prefix_end_idx = len(hint_ids)
                            assert prefix_end_idx is not None, f"prefix_end_idx is None for hint_ids: {hint_ids}"
                            hint_index.extend([prefix_end_idx] * rollout_n)
                            off_seq_loss_mask.extend([1] * rollout_n)
                            # if batch_id == 4:
                            #     off_seq_loss_mask.extend([1] * rollout_n)
                            # else:
                            #     off_seq_loss_mask.extend([0] * rollout_n)
                        else:
                            assert hint_level <= 3, f"hint_level {hint_level} is not supported, should be in [0, 1, 2, 3]"


                    off_seq_loss_mask = torch.tensor(off_seq_loss_mask, dtype=torch.bool).to(idx.device).unsqueeze(1)
                    hint_list = [item for item in hint_list for _ in range(rollout_n)]

                    assert len(hint_list) == len(hint_index)

                    # prefix_list = [hint_list[i][:int(len(hint_list[i]) * hint_ratio[i])] for i in range(len(hint_list))]
                    prefix_list = []
                    for i in range(len(hint_list)):
                        prefix_end_idx = hint_index[i]
                        prefix_list.append(hint_list[i][:prefix_end_idx])
                        # if prefix_end_idx > 0:
                        #     prefix_list[i].append(self.guidance_end_token_id)

                    idx_list = [idx_list[i] + prefix_list[i] for i in range(len(idx_list))]


                    # assert self.config.n == 8, "now only support n = 8"
                    # group_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size * 2)], dtype=object)
                    # group_uid = np.repeat(group_uid, 4)
                    # group_index = np.tile([1,2], batch_size)
                    # group_index = np.repeat(group_index, 4)
                    # group_index = torch.from_numpy(group_index).to(idx.device)
                    if rollout_n == 4:
                        group_index = np.repeat(1, batch_size * rollout_n)
                        group_index = torch.from_numpy(group_index).to(idx.device)
                    else:
                        group_index = np.repeat(2, batch_size * rollout_n)
                        group_index = torch.from_numpy(group_index).to(idx.device)

                    # last_hint_levels = np.repeat(hint_levels, rollout_n)
                    # last_hint_levels = torch.from_numpy(last_hint_levels).to(idx.device)
                    last_hint_levels = hint_levels.repeat_interleave(rollout_n).to(idx.device)
        
                if not do_sample:
                    kwargs = {
                        "best_of": 1,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "temperature": 0,
                        "n": 1,  # if greedy, only 1 response
                    }
                elif is_validate:
                    # TODO: try **
                    kwargs = {
                        "top_k": self.config.val_kwargs.top_k,
                        "top_p": self.config.val_kwargs.top_p,
                        "temperature": self.config.val_kwargs.temperature,
                        "n": 1,  # if validate, already repeat in ray_trainer
                    }

                lora_requests = None
                kwargs['n'] = 1
                # users can customize different sampling_params at different run
                with self.update_sampling_params(**kwargs):
                    if batch_id == 4 and is_validate == False:
                        params_list = []
                        for i in range(len(idx_list)):
                            assert last_hint_levels.shape[0] == len(idx_list), f"last_hint_levels shape {last_hint_levels.shape} does not match idx_list {len(idx_list)}"
                            hint_level = last_hint_levels[i]
                            if hint_level == 3:
                                new_params = self.sampling_params.clone()
                                new_params.max_tokens = 1
                            else:
                                # hold the same
                                new_params = self.sampling_params.clone()
                            params_list.append(new_params)

                        output = self.inference_engine.generate(
                            prompts=None,  # because we have already convert it to prompt token id
                            sampling_params=params_list,
                            prompt_token_ids=idx_list,
                            lora_request=lora_requests,
                            use_tqdm=False,
                        )

                    else:
                        output = self.inference_engine.generate(
                            prompts=None,  # because we have already convert it to prompt token id
                            sampling_params=self.sampling_params,
                            prompt_token_ids=idx_list,
                            lora_request=lora_requests,
                            use_tqdm=False,
                        )

                    # TODO(sgm): disable logprob when recompute_log_prob is enable
                    # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

                    response = output[0].to(idx.device)

                    if is_validate == False:
                        resp_list = [
                            _pre_process_inputs_right_pad(self.pad_token_id, resp) if last_hint_levels[i] != 3 else []
                            for i, resp in enumerate(response)
                        ]

                        concat_resp_list = []
                        hint_end_index = [min(len(prefix_list[i]), self.config.response_length) for i in range(len(hint_list))]
                        hint_start_index = [0 for _ in range(len(hint_end_index))]
                        # for i in range(len(hint_end_index)):
                        #     if i % 8 < 4:
                        #         hint_start_index.append(hint_end_index[i])
                        #     else:
                        #         hint_start_index.append(hint_end_index[i-4])


                        prefix_mask = torch.zeros([len(resp_list), self.config.response_length], dtype=torch.bool).to(idx.device)
                        off_loss_mask = torch.zeros([len(resp_list), self.config.response_length], dtype=torch.bool).to(idx.device)
                        
                        for i in range(len(resp_list)):
                            concat_resp_list.append(torch.tensor(prefix_list[i] + resp_list[i]))
                            prefix_len = min(len(prefix_list[i]), self.config.response_length)
                            prefix_mask[i, :prefix_len] = True
                            off_loss_mask[i, hint_start_index[i]:hint_end_index[i]] = True

                        off_loss_mask = off_seq_loss_mask * off_loss_mask

                        resp_max_len = max([len(resp) for resp in concat_resp_list])
                        tt = torch.ones(len(concat_resp_list), resp_max_len).fill_(self.pad_token_id)
                        for i in range(len(concat_resp_list)):
                            tt[i][:len(concat_resp_list[i])] = concat_resp_list[i].clone().detach()
                        response = tt.to(idx.device)[:, :self.config.response_length].to(response.dtype)

                    if response.shape[1] < self.config.response_length:
                        response = pad_sequence_to_length(
                            response, self.config.response_length, self.pad_token_id
                        )


                    if (not is_validate) and rollout_n > 1:
                        idx = _repeat_interleave(idx, rollout_n)
                        if not is_validate:
                            hint_input_ids = _repeat_interleave(hint_input_ids, rollout_n)
                        attention_mask = _repeat_interleave(attention_mask, rollout_n)
                        position_ids = _repeat_interleave(position_ids, rollout_n)
                        batch_size = batch_size * rollout_n
                        assert last_hint_levels.shape[0] == batch_size, f"last_hint_levels shape {last_hint_levels.shape} does not match batch_size {batch_size}"
                        
                    seq = torch.cat([idx, response], dim=-1)

                # Create position IDs and attention mask for full sequence
                response_length = response.size(1)
                delta_position_id = torch.arange(
                    1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(
                    batch_size, 1)

                response_position_ids = position_ids[:, -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids],
                                       dim=-1)
                response_attention_mask = get_eos_mask(
                    response_id=response,
                    eos_token=eos_token_id,
                    dtype=attention_mask.dtype)
                attention_mask = torch.cat(
                    (attention_mask, response_attention_mask), dim=-1)

                # Construct output batch
                batch = TensorDict(
                    {
                        'prompts': idx,
                        'responses': response,
                        'input_ids': seq,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids,
                    },
                    batch_size=batch_size)
                
                if not is_validate:
                    batch["hint_input_ids"] = hint_input_ids
                    batch["prefix_mask"] = prefix_mask
                    batch["off_loss_mask"] = off_loss_mask
                    batch["group_index"] = group_index
                    batch["hint_levels"] = last_hint_levels

                # Free cache if configured
                if self.config.free_cache_engine:
                    self.inference_engine.free_cache_engine()

                return DataProto(batch=batch)

            except Exception as e:
                traceback.print_exc()
                print("Restarting vLLM due to error: ", e)
                print("Retrying...")

                # Clean up and restart engine
                torch.cuda.empty_cache()
                if hasattr(self.inference_engine, 'free_cache_engine'):
                    self.inference_engine.free_cache_engine()
                del self.inference_engine

                # Reinitialize engine with same parameters
                self.inference_engine = LLM(
                    self.actor_module,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.model_hf_config,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype=self.config.dtype,
                    enforce_eager=self.config.enforce_eager,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    skip_tokenizer_init=False,
                    max_model_len=self.config.prompt_length +
                    self.config.response_length,
                    load_format=self.config.load_format)
                print("vLLM is ready to roll!")

                if attempt < max_retries - 1:
                    continue

        raise RuntimeError(
            f"Failed to generate sequences after {max_retries} attempts")

def unit_test():
    batch = DataProto.from_single_dict({
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'tgt_input_ids': torch.tensor([[1, 2, 3, 4, 5]])
    })
    idx = batch.batch['input_ids']
    tgt_input_ids = batch.batch['tgt_input_ids']
    
    batch_size = tgt_input_ids.size(0)
    
    # idx_list = [1, 2, 3, 4, 5]
    idx_list = [
        _pre_process_inputs(1, idx[i])
        for i in range(batch_size)
    ]

    idx_list = sum([[idx_list[i]] * 2 for i in range(len(idx_list))], [])

    tgt_input_ids = batch.batch['tgt_input_ids']  # [bsz, tgt_len]

    tgt_list = [
        _pre_process_inputs(1, tgt_input_ids[i])
        for i in range(batch_size)
    ]
    
    tgt_list = sum([[tgt_list[i]] * 2 for i in range(len(tgt_list))], [])
    
    import random
    prefix_ratios = [random.randint(0, 100)/100 for _ in range(len(tgt_list))]
    prefix_list = [tgt_list[i][:int(len(tgt_list[i]) * prefix_ratios[i])] for i in range(len(tgt_list))]
    idx_list = [idx_list[i] + prefix_list[i] for i in range(len(idx_list))]
    print(idx_list)
    print(tgt_list)
    print(prefix_list)

if __name__ == "__main__":
    unit_test()