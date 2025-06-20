import pandas as pd
import torch
import torch.distributed as dist
import transformers
from transformers import TrainerCallback
from typing import Dict, Optional, Sequence, List
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoConfig, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dataloader import get_dataset
import argparse
import re
from trl import GRPOConfig, GRPOTrainer
import deepspeed
import shutil
from FlagEmbedding import BGEM3FlagModel
import logging

from reward_service_bgem3 import get_reward_similarity_with_bgem3

# Use in case you have only one GPU
# from reward import get_reward_similarity_with_bgem3

def setup_rank_aware_logging(local_rank: int):
    """
    Configure logging to avoid duplicates in a distributed setting.
    Only rank 0 logs to console; other ranks suppress console logging.
    """
    # Initialize distributed process group if not already initialized
    if local_rank != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    
    # Determine the rank of the current process
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Get loggers for relevant libraries
    loggers = [
        logging.getLogger("accelerate"),
        logging.getLogger("transformers"),
        logging.getLogger("FlagEmbedding"),
        logging.getLogger("deepspeed"),
        logging.getLogger() 
    ]

    # Configure logging based on rank
    if rank != 0:
        # For non-rank-0 processes, suppress console logging
        for logger in loggers:
            # Set a high logging level to suppress most messages
            logger.setLevel(logging.CRITICAL)
            # Remove any handlers that output to console
            for handler in logger.handlers[:]:  # Create a copy of the list
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)
    else:
        # For rank 0, set a reasonable logging level (e.g., WARNING)
        for logger in loggers:
            logger.setLevel(logging.WARNING)

    # Optionally, reduce DeepSpeed verbosity
    os.environ["DS_VERBOSE"] = "0"
    os.environ["DEEPSPEED_LOG_LEVEL"] = "WARNING"

class BGERewardModel:
    def __init__(self, model_name='BAAI/bge-m3'):
        print(f'\nLoading BGE M3 Model ({model_name})...')
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        self.__name__ = "BGERewardModel"
        

    def compute_bge_score(self, sentences_1, sentences_2):
        sentence_pairs = [[sentences_1, sentences_2]]
        bge_score = self.model.compute_score(
            sentence_pairs,
            max_passage_length=128,
            weights_for_different_modes=[1, 0.3, 1]
        )
        return bge_score['colbert+sparse+dense'][0]

    def __call__(self, prompts, completions, **kwargs):
        print("completions len:", len(completions))
        print('label:', kwargs['ground_truths'][0][0]['content'])
        print("completions: ")
        for completion in completions:
            print("predict:", completion[0]['content'])
        print("-----\n")

        reward_lst = []
        for i in range(len(completions)):
            score = self.compute_bge_score(
                completions[i][0]['content'],
                kwargs['ground_truths'][i][0]['content']
            )
            reward_lst.append(score)
        print("reward_lst...:", reward_lst)
        return reward_lst

class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def train(model_id: str, dataset: Dataset, lora_opt: bool = False, local_rank: int = -1):
    # Setup rank-aware logging
    setup_rank_aware_logging(local_rank)

    # Only rank 0 should print this
    if local_rank in [-1, 0]:
        print(f"Running on GPU {local_rank}...")

    shutil.rmtree(f"{model_id}_grpo", ignore_errors=True)

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    # Only rank 0 should print this
    if local_rank in [-1, 0]:
        print("\nLoad model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )

    if lora_opt:
        if local_rank in [-1, 0]:
            print(f'\nConfig LoRA on rank {local_rank}...')
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        if local_rank in [-1, 0]:
            model.print_trainable_parameters()

    if local_rank in [-1, 0]:
        print(f'\nGRPO config on rank {local_rank}...')
    training_args = GRPOConfig(
        output_dir=f"{model_id}_grpo",
        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        bf16=True,
        max_completion_length=128,
        num_generations=4,
        max_prompt_length=2048,
        logging_steps=2,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10,
        deepspeed="./RL_GRPO_test/ds_config.json",
        beta=1.12
    )

    if local_rank in [-1, 0]:
        print(f'\nTrainer on rank {local_rank}...')
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[get_reward_similarity_with_bgem3],
        args=training_args,
        train_dataset=dataset,
        callbacks=[ClearCacheCallback()]
    )

    if local_rank in [-1, 0]:
        print(f'\nTrain on rank {local_rank}...')
    trainer.train()

def main(args):
    # Setup rank-aware logging as early as possible
    setup_rank_aware_logging(args.local_rank) 

    if args.local_rank in [-1, 0]:
        print('Training: ...')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=False
    )
    dataset_train = get_dataset(data_path=args.dataset_path, tokenizer=tokenizer, max_prompt_ids_length=1024)
    model_id = args.model_id
    train(
        model_id=model_id,
        dataset=dataset_train,
        local_rank=args.local_rank
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  

    parser.add_argument('--dataset-path', 
                        help='jsonl file path', 
                        default='/workspace/SMES_Therapy_framework/RL_GRPO_test/data/test.jsonl')
    parser.add_argument('--model-id', 
                        help='', 
                        required=False, 
                        default='Qwen/Qwen2-7B-Instruct')
    parser.add_argument('--local_rank', 
                        type=int, 
                        default=-1, 
                        help='Local rank for distributed training')
    args = parser.parse_args()

    main(args)