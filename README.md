# Multimood

Multimood is a project focused on advancing multimodal learning, adapted from the VideoLLaMA 2 codebase. This repository provides the necessary setup and instructions to get started with the project.

## Requirements

- **CUDA Version**: >= 12.4
- **PyTorch Version**: >= 2.5.0
- **Transformers**: == 4.42.3

## Installation

To set up the Multimood project, follow these steps:

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Multimood
   ```

2. Create a conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the conda environment and install additional dependencies:
   ```bash
   apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
   pip install bitsandbytes==0.43.1
   pip install h5py
   pip install flash-attn --no-build-isolation
   pip install deepspeed==0.15.4
   ```

## Training

### Training Data Structure
Suppose your data structure is like:
```
Multimood
└── datasets
       ├── videos
       ├── train.jsonl
       └── test.jsonl
```

Then you should re-organize the annotated video/image sft data according to the following JSON format:
```json
[
  {
    "Dialogue_ID": 0, 
    "Utterance": [
        "Utterance 1", 
        "Utterance 2"
    ],
    "Emotion": "emotion",
    "Strategy": "strategy",
    "path_to_vid_user_most_recent": [
        "video01.mp4", 
        "video02.mp4", 
        "video03.mp4", 
        "video04.mp4", 
        "video05.mp4"
    ], 

    "history_chat": [
      "hist 1", 
      "hist 2", 
      "hist 3", 
    ],

    "utt_user_most_recent":  [
        "Recent utterences"
    ], 

    "situation": "situation" , 

    "problem_type": "problem" ,  
  }
]
```

### Training Command
```bash
   bash scripts/custom/finetune.sh
   ```

This line will train the whole proposed pipeline.


### Training Parameters Explanation
- `--nnodes $WORLD_SIZE$`: Number of nodes for distributed training.
- `--deepspeed scripts/zero.json`: Configuration file for DeepSpeed optimization, using Zero strategy.
- `--model_type $MODEL_TYPE$`: Specifies the model type (using VideoLLaMA2 with Qwen2 architecture).
- `--model_path $MODEL_PATH$`: Path to the pre-trained model weights.
- `--vision_tower $VISION_PATH$`: Vision encoder model.
- `--mamba_compressor $COMPRESSOR-PATH$`: Compressor model for efficient processing.
- `--audio_tower $AUDIO_PATH$`: Audio encoder model file.
- `--pretrain_projectors_repo $PROJECTOR_REPO$`: Repository for pre-trained projectors.
- `--mm_projector_type stc_connector_v35`: Type of multimodal projector.
- `--data_path ./data/train.jsonl`: Path to the training data file in JSONL format.
- `--vid_folder ./data/video_data/`: Directory containing video data.
- `--num_k_vid `: Number of videos.
- `--num_frames `: Number of frames to process from each video.
- `--bf16`: Enables bfloat16 precision.
- `--tf32`: Enables TensorFloat32 precision.
- `--fp16`: Disables float16 precision.
- `--output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME}`: Directory to save training outputs.
- `--num_train_epochs`: Number of training epochs.
- `--per_device_train_batch_size $LOCAL_BATCH_SIZE`: Batch size per device for training.
- `--per_device_eval_batch_size`: Batch size per device for evaluation.
- `--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS`: Steps to accumulate gradients.
- `--save_strategy`: Strategy for saving checkpoints (by steps).
- `--save_steps`: Save checkpoint every specified number of steps.
- `--save_total_limit`: Maximum number of checkpoints to retain.
- `--learning_rate `: Learning rate for training.
- `--weight_decay`: Weight decay for regularization.
- `--warmup_ratio`: Ratio of warmup steps for the learning rate scheduler.
- `--max_grad_norm`: Maximum gradient norm for clipping.
- `--lr_scheduler_type`: Type of learning rate scheduler.
- `--logging_steps`: Frequency of logging steps.
- `--model_max_length `: Maximum sequence length for the model.
- `--gradient_checkpointing `: Enables gradient checkpointing to save memory.


