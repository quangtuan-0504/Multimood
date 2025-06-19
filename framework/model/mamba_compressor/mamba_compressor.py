from transformers import AutoModelForCausalLM, AutoTokenizer, MambaModel
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from dataclasses import dataclass
import logging
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
from .prepare_data import prepare_training_data_single, prepare_training_data_mulitple, prepare_input
from .conversations import ConversationDataset, csv_to_conversations

class MambaCompressor(nn.Module):
    def __init__(self, llm_input_size: int, tokenizer_len, mem_token_id, mamba_path: str = "state-spaces/mamba-370m-hf", torch_dtype=None):
        """
        Initialize MambaCompressor with a frozen Qwen LLM for reconstruction training
        Args:
            mamba_path: Path to pretrained Mamba model
            llm_path: Path to Qwen LLM
            torch_dtype: Data type for model loading (helps with DeepSpeed compatibility)
        """
        super().__init__()
        
        # Set default dtype if none provided
        self.dtype = torch_dtype if torch_dtype is not None else torch.float32

        # Load the model with the specified dtype
        self.mamba = MambaModel.from_pretrained(mamba_path, torch_dtype=self.dtype)
        
        # Move from meta device to CPU, allocating empty tensors
        self.mamba.to_empty(device="cpu")
        
        # Ensure the model is on the correct device and dtype before resizing
        self.mamba = self.mamba.to(dtype=self.dtype, device="cpu")
        
        self._custom_resize_token_embeddings(tokenizer_len)
        # self.mamba.resize_token_embeddings(tokenizer_len)
        self.mem_token_id = mem_token_id
        
        mamba_hidden = self.mamba.config.hidden_size
        self.memory_projection = nn.Linear(mamba_hidden, llm_input_size, dtype=self.dtype)
    
    
    def _custom_resize_token_embeddings(self, new_num_tokens):
        """
        Custom method to resize token embeddings with consistent dtype and device.
        """
        # Get the current embedding layer
        old_embeddings = self.mamba.get_input_embeddings()
        old_num_tokens, embedding_dim = old_embeddings.weight.shape

        if old_num_tokens == new_num_tokens:
            return

        # Create a new embedding layer with the same dtype and device
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim, dtype=self.dtype, device="cpu")
        
        # Copy the old weights to the new embeddings (up to the minimum size)
        min_num_tokens = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:min_num_tokens] = old_embeddings.weight.data[:min_num_tokens]

        # Set the new embeddings in the model
        self.mamba.set_input_embeddings(new_embeddings)
        

    def forward(self, input_ids):
        """
        Forward pass: compress and reconstruct
        Args:
            input_ids: Input token ids [batch_size, sequence_length]
        Returns:
            Dict containing loss and logits
        """
        outputs = self.mamba(input_ids.to(self.mamba.device)).last_hidden_state
        mem_token_mask = input_ids == self.mem_token_id
      
        
        
        
        batch_indices = torch.arange(outputs.size(0))[:, None]
        mem_positions = mem_token_mask.nonzero()
        batch_nums = mem_positions[:, 0]
        seq_positions = mem_positions[:, 1]
        
        # Group memory features by batch
        memory_features = []
        for batch_idx in range(outputs.size(0)):
            batch_mask = batch_nums == batch_idx
            batch_positions = seq_positions[batch_mask]
            batch_features = outputs[batch_idx, batch_positions]
            memory_features.append(batch_features)
        
        memory_features = torch.stack(memory_features)
        # print(f'Memory features: {memory_features.shape}')
        outputs = self.memory_projection(memory_features.to(dtype= self.dtype))
        # print(f'Final outputs: {outputs.shape}')
        
        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        # Save Mamba model
        self.mamba.save_pretrained(os.path.join(path, "mamba"))
        
        # Save memory projection layer
        torch.save(self.memory_projection.state_dict(), 
                os.path.join(path, "memory_projection.pt"))
        
        # Save config including model dimensions
        config = {
            "llm_input_size": self.memory_projection.out_features,
            "mamba_hidden_size": self.mamba.config.hidden_size,
            "device": self.device,
            "mem_token_id": self.mem_token_id
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str, device: str, tokenizer_len: int, mem_token_id: int, torch_dtype=None):
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Initialize model with config
        model = cls(
            llm_input_size=config["llm_input_size"],
            tokenizer_len=tokenizer_len,
            mem_token_id=mem_token_id,
            mamba_path=os.path.join(path, "mamba"),
            torch_dtype=torch_dtype
        )
        
        # Load memory projection
        memory_projection_path = os.path.join(path, "memory_projection.pt")
        model.memory_projection.load_state_dict(
            torch.load(memory_projection_path)
        )
        
        return model

def internal_train(
    mamba_model: MambaCompressor,
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    system_prompt: str,
    train_data: List[str],
    val_data: Optional[List[str]] = None,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    First stage: Train on single utterances
    The Mamba outputs at <MEM> tokens are fed to LLM for reconstruction
    """
    if device == 'cuda':
        torch.cuda.empty_cache()

    llm_model
    llm_model.eval()  # Keep LLM in eval mode
    
    # Freeze LLM parameters
    for param in llm_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=learning_rate)
    
    # Create datasets
    train_dataset = ConversationDataset(train_data, llm_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_data:
        val_dataset = ConversationDataset(val_data, llm_tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        mamba_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_texts = batch['input_text']
            target_texts = batch['target_text']

            input_data = prepare_input(
                mamba_model=mamba_model,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                system_prompt=system_prompt,
                input_texts=input_texts,
                device=device
            )
            
            # with torch.no_grad():
            llm_outputs = llm_model(
                inputs_embeds=input_data['input_embeds'],
                attention_mask=input_data['attention_mask'],
                # max_length=128,
                labels=input_data['labels'],
                return_dict=True
            )
            
            loss = llm_outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(mamba_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            del input_data
            del llm_outputs

            torch.cuda.empty_cache()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Average training loss: {avg_loss:.4f}')
        
        # Validation
        if val_data:
            mamba_model.eval()
            val_loss = 0
            
            for batch in val_loader:
                input_texts = batch['input_text']
                # print(input_texts)
                
                input_data = prepare_input(
                    mamba_model=mamba_model,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    system_prompt=system_prompt,
                    input_texts=input_texts,
                    device=device
                )
                
                with torch.no_grad():
                    llm_outputs = llm_model(
                        inputs_embeds=input_data['input_embeds'],
                        attention_mask=input_data['attention_mask'],
                        labels=input_data['labels'],
                        return_dict=True
                    )

                val_loss += llm_outputs.loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f'Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}')

def train_mamba_compressor_on_single_utterance(
    train_data_path: str,
    valid_data_path: str,
    model_dir: str,
    frozen_llm: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    model = None, # continue training from a model
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2.5e-5,
    device: str = "cuda"
) -> Tuple[MambaCompressor, dict]:
    """
    Training MambaCompressor with a frozen Qwen LLM for reconstruction single utterances
    
    """
    log_dir = Path(model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading data...")

    logging.info("Converting data to Conversations...")
    train_conversations = csv_to_conversations(train_data_path)
    valid_conversations = csv_to_conversations(valid_data_path)

    logging.info("Preparing training data...")
    train_data = prepare_training_data_single(train_conversations)
    valid_data = prepare_training_data_single(valid_conversations)

    logging.info("Loading LLM model...")
    frozen_llm = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", torch_dtype=torch.float16).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", add_bos_token = True)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>']})
    mem_token_id = llm_tokenizer.convert_tokens_to_ids('<MEM>')
    llm_input_size = frozen_llm.config.hidden_size
    if model is None:
        model = MambaCompressor(device=device, llm_input_size=llm_input_size, 
                            tokenizer_len=len(llm_tokenizer), 
                            mem_token_id=mem_token_id,
                            torch_dtype=torch.float16).to(device)

    for param in frozen_llm.parameters():
        param.requires_grad = False

    logging.info("Starting training...")
    system_prompt = "Please reconstruct the conversation in a natural way."

    internal_train(
        mamba_model=model,
        llm_model=frozen_llm,
        llm_tokenizer=llm_tokenizer,
        system_prompt=system_prompt,
        train_data=train_data,
        val_data=valid_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    logging.info("Training complete!")
    model.save_pretrained(log_dir)
    logging.info(f"Model saved to {log_dir}")

    return model

def train_mamba_compressor_on_conversations(
    train_data_path: str,
    valid_data_path: str,
    model_dir: str,
    frozen_llm: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    model = None, # continue training from a model
    batch_size: int = 1,
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    device: str = "cuda"
) -> Tuple[MambaCompressor, dict]:
    """
    Training MambaCompressor with a frozen Qwen LLM for reconstruction on full conversations
    
    """
    log_dir = Path(model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading data...")

    logging.info("Converting data to Conversations...")
    train_conversations = csv_to_conversations(train_data_path)
    valid_conversations = csv_to_conversations(valid_data_path)

    logging.info("Preparing training data...")
    train_data = prepare_training_data_mulitple(train_conversations)
    valid_data = prepare_training_data_mulitple(valid_conversations)

    logging.info("Loading LLM model...")
    
    mem_token_id = llm_tokenizer.convert_tokens_to_ids('<MEM>')
    llm_input_size = frozen_llm.config.hidden_size
    if model is None:
        model = MambaCompressor(device=device, llm_input_size=llm_input_size, 
                            tokenizer_len=len(llm_tokenizer), 
                            mem_token_id=mem_token_id,
                            torch_dtype=torch.float16).to(device)

    for param in frozen_llm.parameters():
        param.requires_grad = False

    logging.info("Starting training...")
    system_prompt = "Please reconstruct the conversation history"

    internal_train(
        mamba_model=model,
        llm_model=frozen_llm,
        llm_tokenizer=llm_tokenizer,
        system_prompt=system_prompt,
        train_data=train_data,
        val_data=valid_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    logging.info("Training complete!")
    model.save_pretrained(log_dir)
    logging.info(f"Model saved to {log_dir}")

    return model

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frozen_llm = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", torch_dtype=torch.float16).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", add_bos_token = True)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>']})

    model = train_mamba_compressor_on_single_utterance(
        train_data_path="MESC/train.csv",
        valid_data_path="MESC/val.csv",
        model_dir='./mamba_compressor_log_2702_1',
        frozen_llm=frozen_llm,
        llm_tokenizer=llm_tokenizer
    )

    train_mamba_compressor_on_conversations(
        model=model,
        train_data_path="MESC/train.csv",
        valid_data_path="MESC/val.csv",
        model_dir='./mamba_compressor_log_2702_2',
        frozen_llm=frozen_llm,
        llm_tokenizer=llm_tokenizer
    )

if __name__ == "__main__":
    train()