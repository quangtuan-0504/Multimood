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
from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import TrainerCallback

import sys
sys.path.append('./')
from framework.model import *
from framework.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from framework.mm_utils import tokenizer_multimodal_token, process_video, process_image, process_audio_file
from framework.trainer import (MultiMoodTrainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)



# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


import os
import subprocess
import shutil
import torch
from typing import List
import uuid


class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

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
            abs_path = os.path.abspath(path)
            urls_file.write(f"file '{abs_path}'\n")

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
    pretrain_projectors_repo: Optional[str] = field(default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV", metadata={"help": "This is the videollama2 model path"})
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
    tune_mm_mlp_adapter_a: bool = field(default=False)
    tune_adapter_llm: bool = field(default=False)
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
    vid_folder: Optional[str] = field(default=None, metadata={"help": "folder video"})
    data_folder: Optional[str] = field(default=None)
    # note: in the one turn, it can have several video desc. Because the utterances continously of user is concatnated.
    # k video description's user most recently is constant=5, don't get video description in current user turn , it is set in the data process pipeline
    k_cur_vid_user:  Optional[int] = field(default=5, metadata={"help": "k video most recently of user in current turn user, max is 10"})
    k_turn_history: Optional[int] = field(default=7, metadata={"help": "k turn user-assistant most recently, don't get current turn user-assistant"})
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
    save_safetensors: bool = field(default=False)
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

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])

        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX

        input_ids.append(input_id)
        targets.append(target)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    input_ids = []
    targets = []

    for source in sources:
        source = source[1:]  # Remove system prompt if present

        if source[0]["role"] != "user":
            source = source[1:]

        # Get whole conversation as tokens
        conversation = tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=False)
        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        input_ids.append(input_id)
        target = [IGNORE_INDEX] * len(input_id)

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        # Prepare message history
        message = []
        cur = 0
        for idx in range(0, len(source), 2):
            user_msg = source[idx]
            assistant_msg = source[idx + 1]

            tmp_message = [user_msg, assistant_msg]

            # instruction: user prompt up to this turn (with generation prompt)
            instruction = tokenizer.apply_chat_template(
                message + tmp_message[:1], tokenize=False, add_generation_prompt=True
            )
            # conversation: user + assistant (full dialogue until this point)
            conversation = tokenizer.apply_chat_template(
                message + tmp_message, tokenize=False, add_generation_prompt=False
            )

            instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
            conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

            # Mask all previous messages
            if idx < len(source) - 2:
                # Not final assistant message, so fully mask
                pass  # Nothing to change, remains IGNORE_INDEX
            else:
                # This is the final assistant message, unmask it
                for i in range(instruction_len, conversation_len):
                    target[i] = input_id[i]

            cur = conversation_len
            message += tmp_message

        targets.append(torch.tensor(target))

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"Unsupported modal token {modal_token}."

    for source in sources:
        for sentence in source:
            if modal_token in sentence['content']:
                sentence['content'] = sentence['content'].replace(modal_token, '').strip()
                sentence['content'] = modal_token + '\n' + sentence['content']
                sentence['content'] = sentence['content'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["content"] = sentence["content"].replace(modal_token, replace_token)

    return sources

def tokenizer_text_his(text_his : str , tokenizer :  transformers.PreTrainedTokenizer):
    if text_his:
        text_his_ids = tokenizer(text_his, truncation=True, max_length=512, return_tensors='pt')
    else:
        text_his_ids = tokenizer(tokenizer.pad_token, truncation=True, max_length=512, return_tensors='pt')
    return text_his_ids



def get_conversation(components) -> List[Dict[str,str]] :

    label = (
        f"Client's emotion: {components['client_emotion']} \n"
        f"Therapist's emotion: {components['therapist_emotion']} \n"
        f"Therapist's strategy: {components['therapist_strategy']} \n"
        f"Therapist's response: {components['therapist_utterance']}"

    )



    total_prompt = (
        "[CONTEXT] \n"
        "Problem: {problem_type} \n"
        "Situation: {situation} \n"
        "History chat informations above \n"
        "[CURRENT CONTEXT] \n"
        "Client's video: <video> \n"
        "Chat history: <history> \n"
        "Client utterance: {user_question} \n"
        
        "The [CONTEXT] is history of current conversation between 'Client' and 'Therapist'. And [CURRENT CONTEXT] is the current 'Client' turn in the conversation \n"
        "Now you are the 'Therapist' and you need to understand the context to predict the Client’s emotion, Therapist’s emotion, then Therapist’s strategy. After that you need to make an empathy response to the 'Client' based on the context. Let's think about it step by step: \n"
        "Step 1: Describe and understanding the context and content of the conversation \n"
        "Step 2: Predict the following and explain why for each components, notice that Therapist's strategy have guideline: \n"
            "Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            "Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            # "Guide for Therapist's strategy: Communication Skills: Involve small talk and daily communication with clients, along with using simple phrases and body language during listening, thereby establishing a positive communication atmosphere.; Advisement: Offering guidance, advice, or possible solutions to clients, helping them address psychological issues and emotional distress they encounter.; Structuring the therapy: The therapist sets a clear framework and structure for the therapy process. This includes defining the goals of therapy, its duration, the main activities during therapy, and rules, etc.; Guiding the Pace and Depth of the Conversation: Therapists regulate the pace and depth of conversations during sessions through various techniques and methods, such as shifting the topic when the client is emotionally unstable or guiding the conversation back to key themes when necessary.; Others: Employ support strategies not encompassed by the previously mentioned categories."
            "Therapist's strategy: Choose only one from (open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others). \n"
            "Guide for Therapist's strategy: Communication Skills: Involve small talk and daily communication with clients, along with using simple phrases and body language during listening, thereby establishing a positive communication atmosphere.; Advisement: Offering guidance, advice, or possible solutions to clients, helping them address psychological issues and emotional distress they encounter.; Structuring the therapy: The therapist sets a clear framework and structure for the therapy process. This includes defining the goals of therapy, its duration, the main activities during therapy, and rules, etc.; Guiding the Pace and Depth of the Conversation: Therapists regulate the pace and depth of conversations during sessions through various techniques and methods, such as shifting the topic when the client is emotionally unstable or guiding the conversation back to key themes when necessary.; Others: Employ support strategies not encompassed by the previously mentioned categories."
        "Step 3: You are the 'Therapist', leverage Therapist's emotion and Therapist's strategy, think about how to reply to 'Client' in empathy. Follow this guideline: Understand the Client's emotion, follow Client's point of view and intention, express sympathy for Client's negative situation or approval of Client's positive situation. The response should not imply negative emotions or triggers toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc while supporting user well-being. Keep the information in the response truthful, avoid misinformation. The response should open and honest, foster understanding, connection, and provide comfort or support. The response should safeguard human autonomy, identity and data dignity. Ensuring AI behaviors guide emotional health responsibly, avoiding manipulation or harm. \n"
        "Step 4: You need to consider the potential impact of your reply, the response should repeat a few words from Client utterance, you can express a different position or opinion, but you should not hurt Client's feelings \n"
               "You must follow the output format in [OUTPUT FORMAT] below, just print what is listed in [OUTPUT FORMAT], do not print anything more even your step thought. \n"
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
    conversation = [
        {"role": "system", "content": ("You are an expert in emotional psychology. Your task is to analyze the client's emotional state, predict the therapist's emotional response,"
                                        "determine the therapist's strategy, and generate an appropriate response based on the given inputs and historical context.")
        },
        *components['chat_history'],
        {"role": "user", "content": total_prompt},
        {"role": "assistant", "content": label}
    ]

    return conversation




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
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.mix_sampler_tag = False
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_data_samples = load_jsonl(data_path)

        for idx, item in enumerate(self.raw_data_samples):
            if item is None:
                print(f"None found at index {idx}")

    def __len__(self):
        return len(self.raw_data_samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.raw_data_samples[i]
        # print(sources)
        video_processor = self.data_args.video_processor
        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames

        # Extract raw data
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
        conversation = get_conversation(components)


        data_dict = preprocess([conversation], self.tokenizer, modal_token='<video>')

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], text_his_ids= text_his_ids["input_ids"][0])
        
            
        videos = {"video": None, "audio": None}
        # Process video current
        if cur_user_vid:
            vid_folder = self.data_args.vid_folder
            video_files = [os.path.join(vid_folder, vid) for vid in cur_user_vid]

            # Concatenate videos
            concatenated_video_path = ffmpeg_concat_videos(input_paths = video_files, output_dir= os.path.join(vid_folder, './concated_video_tmp' ) )

            if concatenated_video_path:
                # Process concatenated video
                processed_video = process_video(
                    concatenated_video_path,
                    video_processor,
                    aspect_ratio=self.data_args.image_aspect_ratio,
                    num_frames=num_frames,
                    va=True
                )
                videos['video'] = processed_video['video']
                videos['audio'] = processed_video['audio']

                # Cleanup concatenated video file after processing
                os.remove(concatenated_video_path)
        
        else:
            videos['video'] = torch.zeros(num_frames, 3, self.data_args.image_size, self.data_args.image_size)
            videos['audio'] = torch.zeros(1, 16000 * 5)  # Assuming 5 seconds of silence at 16kHz
      
        data_dict['video'] = videos
        # Return dictionary with tokenized tensors
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, hist_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "text_his_ids"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, # [[ids,ids,...],[ids,ids,...],[ids,ids,...],...]
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side = 'left')
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
            padding_side = 'left')
      
        hist_ids = torch.nn.utils.rnn.pad_sequence(
            hist_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX,
            padding_side = 'left')
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        hist_ids = hist_ids[:, :self.tokenizer.model_max_length]
            
        if hist_ids.shape[1] == 1: 
            hist_ids = None
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            history_input_ids=hist_ids
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch


# TODO: create dataloader here
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    # nn.Dataset load csv + video + audio
    # -> nn.Dataloader
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)




def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error: 
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4, # Use these two lines when training gemma2 and comment the config
            # load_in_8bit=training_args.bits == 8
            quantization_config=BitsAndBytesConfig( # Use this config when training other models and comment the above two lines
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    
    ### Only use these for gemma training
    # chat_template = """{%- for message in messages %}
    #     {% if message['role'] == 'system' -%}
    #     <s>[SYSTEM] {{ message['content'] }} [/SYSTEM]</s>
    #     {% elif message['role'] == 'user' -%}
    #     <s>[USER] {{ message['content'] }} [/USER]</s>
    #     {% elif message['role'] == 'assistant' -%}
    #     <s>[ASSISTANT] {{ message['content'] }} [/ASSISTANT]</s>
    #     {% endif -%}
    #     {%- endfor %}"""
    
    # tokenizer.chat_template = chat_template

 
    tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>']})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    
    model_args.hidden_size = config.hidden_size
    # Parameters for mamba
    if model_args.mamba_compressor:
        config.mamba_compressor = model_args.mamba_compressor
        config.freeze_mamba_compressor = model_args.freeze_mamba_compressor
        config.tokenizer_len = len(tokenizer) 
        config.mem_token_id = tokenizer.convert_tokens_to_ids('<MEM>')

    
    if 'gemma2' in model_args.model_type:
        config._attn_implementation = 'eager'
    else:
        config._attn_implementation = attn_implementation

    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False


    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_size = vision_tower.image_size

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if model_args.tune_mm_mlp_adapter:
            data_args.is_pretraining = True
        else:
            data_args.is_pretraining = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames
        
        
    if model_args.audio_tower is not None:
        # initialize audio encoder + multi-modal projector
        model.get_model().initialize_audio_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        audio_tower = model.get_audio_tower()
        audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter_a = training_args.tune_mm_mlp_adapter_a = model_args.tune_mm_mlp_adapter_a
        training_args.pretrain_mm_mlp_adapter_a = model_args.pretrain_mm_mlp_adapter_a
        training_args.tune_audio_tower = model_args.tune_audio_tower
        # only update mm_mlp's parameters while the remaining ones are kept frozen
        if model_args.tune_mm_mlp_adapter_a:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector_a.parameters():
                p.requires_grad = True
        
        if model_args.tune_audio_tower or model_args.tune_adapter_llm:
            data_args.is_pretraining = False
        else:
            data_args.is_pretraining = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector_a.parameters():
                p.requires_grad = False
  
        if model_args.tune_adapter_llm:
            model.requires_grad_(True)
            if hasattr(model.get_model(), 'vision_tower'):
                for p in model.get_model().vision_tower.parameters():
                    p.requires_grad = False
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = False   
                
        if model_args.freeze_backbone:
            model.requires_grad_(False)

        if model_args.tune_audio_tower:
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = True
        else:
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector_a.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
    
    if model_args.mamba_compressor is not None:
        print("loading mamba checkpoint")
        model.get_model().initialize_mamba_compressor_modules(
                            model_args=config,
                            fsdp=training_args.fsdp)
        model.get_model().mamba_compressor.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        if config.freeze_mamba_compressor:
                for param in model.get_model().mamba_compressor.parameters():
                    param.requires_grad = False
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    with open("model_arch.txt", "w") as f: 
        f.write(str(model))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = VideoLLaMA2Trainer(model=model, 
                                 tokenizer=tokenizer, 
                                 args=training_args, 
                                 **data_module, 
                                 callbacks=[ClearCacheCallback()])
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()