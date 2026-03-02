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

from omegaconf import ListConfig
import os
import re
from typing import List, Union, Optional

import pandas as pd
import copy 

import torch
import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

from verl.utils.dataset.rl_dataset import RLHFDataset

class RLHFDatasetWithTargetTrain(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 max_response_length=8192,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 target_key='target',
                 max_target_length=8192,
                 filter_targets=False,
                 sample_target_ratio=1.0,
                 target_list_key='target_lst',
                 max_num_targets=5,
                 target_probs_key='target_ds_qwen_7b_probs',
        ):
        super().__init__(parquet_files, tokenizer, prompt_key, max_prompt_length, filter_prompts, cache_dir, chat_template_func, return_raw_chat, truncation)
        
        self.max_target_length = max_target_length
        self.filter_targets = filter_targets
        self.target_key = target_key
        self.sample_target_ratio = sample_target_ratio
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.max_num_targets = max_num_targets
        self.max_response_length = max_response_length

    
    def our_get_prompt_text(self, problem: str) -> str:
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
        )

        prompt_text_version1 = (
            f"<|im_start|>system\n"
            f"Your task is to solve the following question by providing a systematic and thorough reasoning process.\n"
            f"First, present the initial steps of your thought process within the <|guidance_start|> and <|guidance_end|> tags. "
            f"This may include restating the problem, careful analysis, reflection, and reconsideration of different aspects. "
            f"Then continue your reasoning based on the guidance section and efficiently work towards the solution. "
            f"Please enclose your final answer within \\boxed{{}}.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{problem}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt_text
    
    def _read_files_and_tokenize(self):
        self.num_workers = max(1, os.cpu_count() // 4)
        self.num_workers = min(self.num_workers, os.cpu_count())
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if True:
        # if False:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            def filter_fn(doc):
                prompt = self.our_get_prompt_text(doc[prompt_key][-1]["content"])
                prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

                model_output = doc['extra_info']['answer_list'][0]
                match = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL | re.MULTILINE)
                valid_thought0 = bool(match and match.group(1).strip())

                model_output = doc['extra_info']['answer_list'][1]
                match = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL | re.MULTILINE)
                valid_thought1 = bool(match and match.group(1).strip())

                return prompt_token_len <= self.max_prompt_length and valid_thought0 and valid_thought1
            
            self.dataframe = self.dataframe.filter(
                filter_fn,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)

        problem = chat[-1]["content"]
        prompt_with_hint = self.our_get_prompt_text(problem=problem)
        raw_prompt = prompt_with_hint
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
            # truncation="right",
        )
        for i in range(2):
            model_output = row_dict['extra_info']['answer_list'][i]
            match = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL | re.MULTILINE)
            if match:
                thought = match.group(1).strip()
            else:
                thought = model_output
            hint_model_inputs = self.tokenizer(thought, return_tensors="pt", add_special_tokens=False)
            hint_input_ids = hint_model_inputs.pop("input_ids")
            hint_attention_mask = hint_model_inputs.pop("attention_mask")
            hint_input_ids, hint_attention_mask = verl_F.postprocess_data(
                input_ids=hint_input_ids,
                attention_mask=hint_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                # truncation=self.truncation,
                truncation="right",
            )

            row_dict[f"hint_input_ids_{i}"] = hint_input_ids[0]

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        uid = row_dict.get("extra_info", {}).get("uuid", "")
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["index"] = index
        row_dict["uid"] = uid
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict
    

class RLHFDatasetWithTargetTest(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 max_response_length=8192,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 target_key='target',
                 max_target_length=8192,
                 filter_targets=False,
                 sample_target_ratio=1.0,
                 target_list_key='target_lst',
                 max_num_targets=5,
                 target_probs_key='target_ds_qwen_7b_probs',
        ):
        super().__init__(parquet_files, tokenizer, prompt_key, max_prompt_length, filter_prompts, cache_dir, chat_template_func, return_raw_chat, truncation)
        
        self.max_target_length = max_target_length
        self.filter_targets = filter_targets
        self.target_key = target_key
        self.sample_target_ratio = sample_target_ratio
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.max_num_targets = max_num_targets
        self.max_response_length = max_response_length

    
    def _read_files_and_tokenize(self):
        self.num_workers = max(1, os.cpu_count() // 4)
        self.num_workers = min(self.num_workers, os.cpu_count())
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)


        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if True:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            def filter_fn(doc):
                prompt = self.our_get_prompt_text(doc[prompt_key][-1]["content"])
                prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

                return prompt_token_len <= self.max_prompt_length
            
            self.dataframe = self.dataframe.filter(
                filter_fn,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def our_get_prompt_text(self, problem: str) -> str:
        prompt_text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
        )

        prompt_text_version1 = (
            f"<|im_start|>system\n"
            f"Your task is to solve the following question by providing a systematic and thorough reasoning process.\n"
            f"First, present the initial steps of your thought process within the <|guidance_start|> and <|guidance_end|> tags. "
            f"This may include restating the problem, careful analysis, reflection, and reconsideration of different aspects. "
            f"Then continue your reasoning based on the guidance section and efficiently work towards the solution. "
            f"Please enclose your final answer within \\boxed{{}}.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{problem}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt_text

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)

        problem = chat[-1]["content"]

        raw_prompt = self.our_get_prompt_text(problem=problem)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

from verl import DataProto
class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)

import torch

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1].item() # the output index should be int

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    token_ids = prompt_token_ids[:non_pad_index[-1][0]].tolist()
    return token_ids
