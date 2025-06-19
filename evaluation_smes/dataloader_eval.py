import re
import os
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import h5py
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append('./')
from framework.model import *
from framework.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from framework.mm_utils import tokenizer_multimodal_token, process_video, process_image, process_audio_file
from framework.videollama2_trainer import (VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

import subprocess
import shutil
import uuid

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None






def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="SMES_llama", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV", metadata={"help": "This is the videollama2 model path"})
    num_k_vid: Optional[int]  = field(default=1, metadata={"help": "This is the number of current videos"})
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None )
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
    # Audio tower Arguments
    audio_tower: Optional[str] = field(default=None)
    tune_audio_tower: bool = field(default=False)
    pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None )
    
    # Labels Arguments 
    num_user_emotion_classes: Optional[int] = field(default=7)
    num_system_emotion_classes: Optional[int] = field(default=7)
    num_strategy_classes: Optional[int] = field(default=10)
    # MambaCompressor
    mamba_compressor: Optional[str] = field(default=None, metadata={"help": "path to the mamba compressor weights."})
    freeze_mamba_compressor: bool = field(default=False, metadata={"help": "Whether to freeze the mamba compressor."})


@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data, file .jsonl , raw samples."})
    # image_folder: Optional[str] = field(default=None)
    vid_folder: Optional[str] = field(default='/workspace/SMES_Therapy_framework/data/mesc/video_data', metadata={"help": "folder video"})
    data_folder: Optional[str] = field(default=None)
    # k vid description's user most recently is constant=5 , it is set in the data process pipeline
    k_cur_vid_user:  Optional[int] = field(default=5, metadata={"help": "k video most recently of user, max is 10"})
    k_turn_history: Optional[int] = field(default=7, metadata={"help": "k turn user-assistant most recently"})
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    output_dir: str = field(default="./output", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})


def ffmpeg_concat_videos(input_paths: List[str], output_dir='./concated_video_tmp') -> str:
    """
    Concatenates multiple videos using FFmpeg without re-encoding, ensuring compatibility in multi-GPU environments.
    Prevents FFmpeg from printing output to the terminal.

    Args:
        input_paths (List[str]): List of video file paths to concatenate.
        output_dir (str): Directory to store the temporary concatenated video.

    Returns:
        str: Path to the concatenated video.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique temporary video filename
    temp_video_name = f"concatenated_{uuid.uuid4().hex}.mp4"
    concatenated_video_path = os.path.join(output_dir, temp_video_name)

    # Create a temporary file for URLs list
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as urls_file:
        urls_file_path = urls_file.name
        for path in input_paths:
            urls_file.write(f"file '{path}'\n")

    # FFmpeg command using the temporary file
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", urls_file_path, "-c", "copy", concatenated_video_path
    ]

    # Run FFmpeg process without printing output
    with open(os.devnull, 'w') as devnull:
        process = subprocess.run(command, stdout=devnull, stderr=devnull)

    # Cleanup: Delete the temporary file after FFmpeg execution
    os.remove(urls_file_path)

    # Check for errors
    if process.returncode != 0:
        return None  # Return None if FFmpeg fails

    return concatenated_video_path 

def tokenizer_totalprompt_label(components : Dict[str,str] , tokenizer : transformers.PreTrainedTokenizer):

    label = (
        f"Client's emotion: {components['client_emotion']} \n"
        f"Therapist's emotion: {components['therapist_emotion']} \n"
        f"Therapist's strategy: {components['therapist_strategy']} \n"
        f"Therapist's response: {components['therapist_utterance']}"

    )
    # Base prompt template
    total_prompt = (

        "Problem: {problem_type} \n"
        "[CONTEXT] \n"
        "Situation: {situation} \n"
        "History: Chat above \n"
        "[CURRENT CONTEXT] \n"
        "Client's video: <video> \n"
        "History: <history> \n"
        "Client utterance: {user_question} \n"
        "[CONTEXT] is the past conversation between Client and Therapist. [CURRENT CONTEXT] is the Client's current turn. As the Therapist, analyze the context to predict the Client’s emotion, your emotion, and your strategy, then respond with empathy. Follow these steps: \n"
        "Step 1: Understand the conversation context and content. \n"
        "Step 2: Predict and explain: \n"
        "Client's emotion: Pick one (anger, sadness, disgust, depression, neutral, joy, fear). \n"
        "Therapist's emotion: Pick one (anger, sadness, disgust, depression, neutral, joy, fear). \n"
        "Therapist's strategy: Pick one (open question, approval, self-disclosure, restatement, interpretation, communication skills, advisement, structuring the therapy, guiding the pace and depth of the conversation, others).\n"
        "Guideline for Therapist's Strategy: \n"
        "- Open questions: Encourage detailed responses for self-reflection. \n"
        "- Approval: Affirm client’s worth or actions. \n"
        "- Self-disclosure: Share feelings or experiences to build rapport. \n"
        "- Restatement: Rephrase client’s words to show understanding. \n"
        "- Interpretation: Uncover deeper meanings in client’s behavior or feelings. \n"
        "- Communication skills: Use small talk, simple phrases, and body language for a positive atmosphere. \n"
        "- Advisement: Offer guidance or solutions for emotional distress. \n"
        "- Structuring the therapy: Set clear therapy goals, duration, and rules. \n"
        "- Guiding the pace and depth: Adjust conversation flow and focus. \n"
        "- Others: Use unlisted support strategies. \n"
        "Step 3: As Therapist, use your emotion and strategy to craft an empathetic response. Understand the Client’s emotion, align with their perspective, express sympathy for negatives or approval for positives. Avoid negativity (e.g., disgust, hatred) and support well-being with honest, comforting responses that respect autonomy and emotional health. \n"
        "Step 4: Ensure the response reflects Client’s words, allows differing views without harm, and considers its impact. \n"
        "The final response must follow this OUTPUT FORMAT, do not return anything more. \n"
        "[OUTPUT FORMAT] \n"
        "Client's emotion: \n"
        "Therapist's emotion: \n"
        "Therapist's strategy: \n"
        "Therapist's response: \n"

    ).format(
        problem_type=components['problem_type'],
        situation=components['situation'],
        user_question=components['cur_user_utt'],
    )
    message_input = [
        {"role": "system", "content": ("You are an expert in emotional psychology. Your task is to analyze the client's emotional state, predict the therapist's emotional response,"
                                        "determine the therapist's strategy, and generate an appropriate response based on the given inputs and historical context.")
        },
        # *components['chat_history'],
        {"role": "user", "content": total_prompt}]
    # Tokenize total_prompt
    # input = tokenizer.apply_chat_template(message_input, tokenize=False, add_generation_prompt=False)
    return message_input , label

def load_jsonl(file_path):
  """Loads a JSONL file into a list of dictionaries.

  Args:
    file_path: The path to the JSONL file.

  Returns:
    A list of dictionaries, where each dictionary represents a line in the JSONL file.
  """
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      try:
        data.append(json.loads(line))
      except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # You might want to handle the error differently, e.g., skip the invalid line
  return data

def tokenizer_text_his(text_his : str , tokenizer :  transformers.PreTrainedTokenizer):
    if text_his:
        text_his_ids = tokenizer(text_his, truncation=True, max_length=512, return_tensors='pt')
    else:
        text_his_ids = tokenizer(tokenizer.pad_token, truncation=True, max_length=512, return_tensors='pt')
    return text_his_ids

# Dataset class
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, 
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments
        ):
        super(LazySupervisedDataset, self).__init__()
        self.mix_sampler_tag = False
        self.tokenizer = tokenizer
        self.data_args = data_args # để biết data_args lấy từ đâu thì cứ lần ngược ra , nơi khởi tạo obj của lớp này chỗ nào ...
        self.raw_data_samples = load_jsonl(data_path)

        for idx, item in enumerate(self.raw_data_samples):
            if item is None:
                print(f"None found at index {idx}")
        print(len(self.raw_data_samples))

    def __len__(self):
        return len(self.raw_data_samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.raw_data_samples[i]
        # print(sources)

        # Extract raw data
        # print(type(sources['history_chat']))
        chat_history = sources['history_chat'][-self.data_args.k_turn_history * 2 : ]

        cur_user_vid = sources['path_to_vid_user_most_recent'][-self.data_args.k_cur_vid_user : ]
        cur_user_utt = " ".join(sources['utt_user_most_recent'])

        situation = sources['situation']
        problem_type = sources['problem_type']

        therapist_emotion = sources['Emotion']
        therapist_strategy = sources['Strategy']
        therapist_utterance = " ".join(sources['Utterance'])
        client_emotion = sources['get_emotion_user_most_recent']
        mamba_history_chat = sources['history_chat_mamba']
        
        text_his_ids = tokenizer_text_his(mamba_history_chat, self.tokenizer)


        components = {
            "therapist_emotion" : therapist_emotion,
            "therapist_strategy" : therapist_strategy,
            "therapist_utterance" : therapist_utterance,
            "client_emotion" : client_emotion,
            "problem_type": problem_type,
            "situation": situation,
            "cur_user_utt": cur_user_utt,
            'chat_history' : chat_history
        }
        total_prompt , label = tokenizer_totalprompt_label(components, self.tokenizer)
        
        concatenated_video_path = None
        # Process video current
        if cur_user_vid:
            vid_folder = self.data_args.vid_folder
            video_files = [os.path.join(vid_folder, vid) for vid in cur_user_vid]
            # video_files = ['/workspace/SMES_Therapy_framework/data_eval_test/vid_test/Breakup.mp4','/workspace/SMES_Therapy_framework/data_eval_test/vid_test/Dream_analysis.mp4']
            # Concatenate videos
            concatenated_video_path = ffmpeg_concat_videos(input_paths = video_files, output_dir= './concated_video_tmp')

        # Return dictionary with tokenized tensors
        return {
            'label': label,
            'total_prompt': total_prompt,
            'video' : concatenated_video_path,
            'text_his_ids': text_his_ids,
        }

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # parser = transformers.HfArgumentParser((ModelArguments,DataArguments))
    # model_args, data_args = parser.parse_args_into_dataclasses()

   

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.add_special_tokens(
            {'additional_special_tokens': 
                [
                    '<|im_start|>', 
                    '<|im_end|>',
                    '<history>',
                    '<video>',
                    '<MEM>'
                ]
            }
        )
    dataset = LazySupervisedDataset(data_path="data_inference/test.jsonl", tokenizer= tokenizer , data_args = data_args)
    # print(dataset[0])
    # for data_point in dataset:
    #     print(data_point['video'])