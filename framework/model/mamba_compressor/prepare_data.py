from typing import List
import torch
from torch.utils.data import Dataset
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum, auto
from dataclasses import dataclass
from .conversations import Conversation, SeparatorStyle

def prepare_training_data_mulitple(conversations: List[Conversation]) -> List[str]:
    return [conv.process_all() for conv in conversations]

def prepare_training_data_single(conversations: List[Conversation]) -> List[str]:
    all_messages = []
    for conv in conversations:
        all_messages.extend(conv.process_each())

    return all_messages


class ConversationDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return {
            'input_text': text,
            'target_text': text
        }

def prepare_input(
    mamba_model,
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    system_prompt: str,
    input_texts: List[str],
    device: str = 'cuda',
    end_sym: str = '\n'
):
    # print(f'Input texts: {input_texts}')
    # Get Mamba memory features
    input_ids = llm_tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    ).to(device)

    memory_features = mamba_model(input_ids).to(torch.float16)
    atts_memory = torch.ones(
        (memory_features.size(0), memory_features.size(1)),
        dtype=torch.long,
    ).to(device)
    
    # Combine system prompt with memory features
    system_encodings = llm_tokenizer(
        [system_prompt] * len(input_texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    ).to(device)
    system_embeds = llm_model.model.embed_tokens(system_encodings['input_ids']) # (batch, seq, hidden)
    # print(f'System encodings: {system_embeds.shape}')
    # print(f'Memory features: {memory_features.shape}')
    
    memory_features = torch.cat([system_embeds, memory_features], dim=1)
    atts_memory = atts_memory[:, :1].expand(-1, memory_features.size(1))

    # Prepare target texts
    target_texts = [t + end_sym for t in input_texts]
    to_regress_tokens = llm_tokenizer(
        target_texts,
        truncation=True,
        return_tensors="pt",
        padding="longest",
        max_length=512,
    ).to(device)
    targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == llm_tokenizer.pad_token_id, -100
            )
    empty_targets = (
                torch.ones([memory_features.shape[0], memory_features.shape[1]],
                        dtype=torch.long).to(device).fill_(-100)
            )
    targets = torch.cat([empty_targets, targets], dim=1)

    batch_size = memory_features.shape[0]

    to_regress_embeds = llm_model.model.embed_tokens(to_regress_tokens.input_ids)
    input_embeds = torch.cat([memory_features, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_memory, to_regress_tokens.attention_mask], dim=1)

    # print(f'input: {len(input_texts)}')
            
    
    # print(f'Target: {targets.shape}')
    # print(f'Input embeds: {input_embeds.shape}')

    return {
        'input_embeds': input_embeds,
        'attention_mask': attention_mask,
        'labels': targets
    }