import torch
import itertools

from transformers import StoppingCriteria, PreTrainedTokenizerBase

def get_matching_seq_mask(input_ids: torch.LongTensor, pattern: torch.LongTensor) -> torch.BoolTensor:
    # 1. Get the last three tokens of each sequence
    last_three = input_ids[:, -3:]
    
    # 2. Compare each row's last three tokens to each pattern
    #    Expand dimensions to allow broadcasting:
    #    - last_three becomes (B, 1, 3)
    #    - patterns becomes (1, M, 3)
    matches = (last_three.unsqueeze(1) == pattern.unsqueeze(0))  # shape (B, M, 3)

    # 3. For a complete match, all three elements must match
    pattern_match = matches.all(dim=-1)  # shape (B, M)

    # 4. For each row, check if any of the patterns matched
    rows_matching = pattern_match.any(dim=-1)  # shape (B,)
    
    return rows_matching


class MatchingTokensStoppingCriteria(StoppingCriteria):
    """
    This class stores a set where each element is a sequence of 3 integers. If any sequence in
    the batch has generated a length-3 subsequence that matches with any element in the set,
    then stop generation for the entire batch.
    
    Each element in the set represents the "</think>" token (the end-of-thinking token for GRPO), and the 2nd integer always
    corresponds to "think". The first and third integers corresponds to all possible tokens in the
    vocabulary that ends with "</" and starts with ">", respectively.
    
    Args:
        tokenizer: PreTrainedTokenizerBase
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device
        
        think_token_id = 26865 # this is the "think" token for Qwen2.5-1.5B-Instruct
        # think_token_id = torch.Tensor(tokenizer("think").input_ids).long().to(self.device)
        
        vocab = tokenizer.get_vocab()        
        left_matching_token_ids = [token_id for token, token_id in vocab.items() if token.endswith("</")]
        right_matching_token_ids = [token_id for token, token_id in vocab.items() if token.startswith(">")]
        
        self.think_token_combo = torch.stack([torch.Tensor([left_token_id, think_token_id, right_token_id]) for left_token_id, right_token_id in itertools.product(left_matching_token_ids, right_matching_token_ids)], dim=0).long().to(self.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        Check if the last three tokens of *any* sequence in the batch matches one of the three-token sequences in self.think_token_combo.
        If any has, return True for all sequences to stop generation for the entire batch.
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be scores for each vocabulary token 
                before SoftMax or scores for each vocabulary token after SoftMax.
                
        Returns:
            `torch.BoolTensor` of shape `(batch_size,)`: A tensor of booleans indicating whether to stop 
            generation for each sequence in the batch. Here, either all values are True or all are False.
        """
        
        seq_len = input_ids.shape[1]
        
        if seq_len <= 3:
            return torch.zeros_like(input_ids[:, 0], dtype=torch.bool)
        else:
            # For each row, check if any of the patterns matched
            rows_matching = get_matching_seq_mask(input_ids, self.think_token_combo)
            
            matched_row = rows_matching.any().item()
            
            if matched_row:
                stop_tensor = torch.ones_like(input_ids[:, 0], dtype=torch.bool)
                eos_reached = input_ids[:, -1] == self.tokenizer.eos_token_id
                stop_tensor[eos_reached] = 0
                return stop_tensor
            else:
                return torch.zeros_like(input_ids[:, 0], dtype=torch.bool)