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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from .base import BaseRollout
from .stop_criteria import MatchingTokensStoppingCriteria, AnyEosStoppingCriteria, get_matching_seq_mask, get_matching_eos_mask

from transformers import GenerationConfig, AutoTokenizer
from typing import Dict

__all__ = ['HFRollout']


def budget_forcing_gen(
    model: nn.Module, 
    tokenizer: AutoTokenizer, 
    input_ids: torch.Tensor, 
    input_attention_masks: torch.Tensor,
    generation_config: GenerationConfig, 
    eos_token_id: int,
    pad_token_id: int,
    min_budget: int, 
    max_budget: int,
    replace_tokens: Dict[str, int],
) -> torch.Tensor:
    """
    Perform budget forcing during generation on a batch of sequences to ensure that the number of newly generated tokens for each sequence is between min_budget and max_budget.
    
    Specifically:
        1. While generated output is less than m tokens, keep generating. Stop whenever:
            - a “</think>” token is generated. In this case, for any sequence in the batch that ends with “</think>”, replace “</think>” with “\n Wait ,”
            - an EOS token (in the case of Qwen, this is <im_end>) is generated. In this case, replace “</im_end>” with the new line character.
        2. Once each sequence has at least min_budget number of newly generated tokens, do generation once more so that the total number of new tokens for each sequence is at most max_budget.
    
    Return the end result of generation along with the initial prompt.
    """
    assert max_budget - min_budget > 128
    max_budget_cut = max_budget - 128
    
    input_idx = input_ids
    attention_mask = input_attention_masks
    device = input_ids.device
    matching_seq_stop_criteria = MatchingTokensStoppingCriteria(tokenizer, device)
    eos_stop_criteria = AnyEosStoppingCriteria(eos_token_id=eos_token_id) # This is <im_end> for the instruct Qwen models
    
    replacement_seq = torch.Tensor([replace_tokens["\n"], replace_tokens["Wait"], replace_tokens[","]]).long().to(device)
    
    init_prompt_len = input_idx.shape[1]
    prompt_length = input_idx.shape[1]
    
    # 1. This is the first round of generation to kick off the budget forcing process.
    # We stop the entire batch whenever any sequence produces "EOS" or "</think>".    
    output_seqs = model.generate(
        input_ids=input_idx,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=max_budget_cut,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id, # This should be 151643 <end_of_text> for Qwen
        generation_config=generation_config,
        output_scores=False,  # this is potentially very large
        stopping_criteria=[matching_seq_stop_criteria, eos_stop_criteria],
        use_cache=True)
    
    # 2. generate until every sequence has produced at least min_budget number of new tokens.
    # Stop whenever any sequence produces "</think>" and EOS, and do the appropriate replacement.
    current_used_budget = output_seqs.shape[1] - init_prompt_len
    while current_used_budget < min_budget:
        matching_seq_mask = get_matching_seq_mask(output_seqs, matching_seq_stop_criteria.think_token_combo)
        matching_eos_mask = get_matching_eos_mask(output_seqs, tokenizer.eos_token_id)
        matching_seq_idx = torch.nonzero(matching_seq_mask, as_tuple=True)[0]
        
        output_seqs[matching_seq_idx, -3:] = replacement_seq
        output_seqs[matching_eos_mask, -1] = replace_tokens["\n"]
        
        new_input_attention_masks = torch.cat([attention_mask, torch.ones_like(output_seqs[:, prompt_length:])], dim=-1)
        input_idx = output_seqs
        
        output_seqs = model.generate(
            input_ids=input_idx,
            attention_mask=new_input_attention_masks,
            do_sample=True,
            max_new_tokens=max_budget_cut-current_used_budget,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=generation_config,
            output_scores=False,  # this is potentially very large
            stopping_criteria=[matching_seq_stop_criteria, eos_stop_criteria],
            use_cache=True)
        
        prompt_length = input_idx.shape[1]
        attention_mask = new_input_attention_masks
        print(f"Current generation length: {current_used_budget}, minimum budget: {min_budget}, maximum budget: {max_budget}")
        
    # 3. After every sequence has generated at least min_budget new tokens, do a final round of generation without any stop criteria to finish off dangling generations.
    new_input_attention_masks = torch.cat([attention_mask, torch.ones_like(output_seqs[:, prompt_length:])], dim=-1)
    
    output_seqs = model.generate(
        input_ids=output_seqs,
        attention_mask=new_input_attention_masks,
        do_sample=True,
        max_new_tokens=max_budget-current_used_budget,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        generation_config=generation_config,
        output_scores=False,  # this is potentially very large
        use_cache=True)
    
    return output_seqs


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, tokenizer: AutoTokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.module = module
        wait_ids = self.tokenizer("Wait").input_ids
        assert len(wait_ids) == 1
        wait_token_id = wait_ids[0] # This should be 14190 for Qwen
        new_line_id = self.tokenizer("/n").input_ids[0] # This should be 198 for Qwen
        comma_id = self.tokenizer(",").input_ids[0]
        self.replacement_tokens = {
            "\n": new_line_id,
            "Wait": wait_token_id,
            ",": comma_id,
        }

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))
        num_return_sequences = self.config.n if do_sample else 1

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get('temperature', self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)
        
        validate = prompts.meta_info.get("validate", False)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Only do budget forcing during training and if do_budget_forcing is set to True in rollout config.
                if (not validate) and self.config.do_budget_forcing:
                    budget_forcing_gen(
                        model=self.module,
                        tokenizer=self.tokenizer,
                        input_ids=idx,
                        input_attention_masks=attention_mask,
                        generation_config=generation_config,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        min_budget=self.config.min_budget,
                        max_budget=self.max_budget,
                        replace_tokens=self.replacement_tokens,
                    )
                else:
                    output_seqs = self.module.generate(
                        input_ids=idx,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        num_return_sequences=num_return_sequences,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        # return_dict_in_generate=True,
                        use_cache=True)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output_seqs
        
        # repeat attention_mask and position_ids so batch dimensions match
        if num_return_sequences > 1:
            attention_mask = attention_mask.repeat_interleave(repeats=num_return_sequences, dim=0) # (b * n, prompt_length)
            position_ids = position_ids.repeat_interleave(repeats=num_return_sequences, dim=0) # (b * n, prompt_length)
            batch_size = batch_size * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)
