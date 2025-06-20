# PPO Training Pipeline for MultiMood

## Overview

This project implements PPO pipeline for training MultiMood on SMES data.

1. **Data Preparation**: Convert conversational therapy data into preference pairs (chosen/rejected format)
2. **Reward Model Training**: Train a reward model to evaluate response quality
3. **PPO Training**: Fine-tune the policy model using Proximal Policy Optimization with the trained reward model

## Project Structure

```
├── reward_training/
│   ├── train.py              # Reward model training script
│   └── prepare_dataset.py    # Data preparation utilities
├── smes_ppo/
│   ├── train.py             # PPO training script
│   └── dataset.py           # Dataset loading and preprocessing
├── scripts/
│   ├── train.sh             # Training execution scripts
│   ├── reward_train.sh      # Reward model training commands
│   └── ds_config.json       # DeepSpeed configuration
├── environment.yml          # Conda environment configuration
├── install_deps.sh          # Dependency installation script
└── test_reward_data.json    # Sample reward training data
```

## Prerequisites

### Environment Setup

```bash
conda create -f environment.yml
conda activate smes_multimodal

chmod +x install_deps.sh
./install_deps.sh
```

## Training Pipeline

### Stage 1: Data Preparation For Reward Training

The pipeline starts with conversational therapy data in JSONL format containing:
- Chat history between therapist and client
- Client emotions and situation context
- Therapist responses with emotion/strategy labels

**Key Components:**
- `smes_ppo/dataset.py`: Processes raw therapy conversations into chat format to be used for generating preference pairs.
- `reward_training/prepare_dataset.py`: We use an LLM to generate the responses. Then, we choose preference pairs with ROUGE-L score below a threshold.

**Output Data Format:**
```python

# Output: Preference pairs
{
    "chosen": <groundtruth response>,
    "rejected": <Lower-quality model generated response>,
}
```

### Stage 2: Reward Model Training

Trains a sequence classification model to score response quality based on preference pairs.

**Configuration:**
- Update your preference configuration in `scripts/reward_train.sh`

### Stage 3: PPO Training

Fine-tunes the policy model using the trained reward model through reinforcement learning.

**Key Components:**
- **Policy Model**: The model being optimized (same as SFT base)
- **Reference Model**: Frozen copy for KL divergence penalty
- **Reward Model**: Trained scorer from Stage 2
- **Value Model**: Estimates state values (can be same as reward model)

**Configuration:**
- Update your preference configuration in `scripts/train.sh`


## Data Requirements

### Input Data Format
Can depend on the dataset you use, but for reward training, the input should be in the preference pairs format:
```json
{
    "chosen": "Groundtruth response",
    "rejected": "Lower quality response"
}
```
