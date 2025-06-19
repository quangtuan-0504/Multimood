import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from typing import Dict, Optional, Sequence, List
import json
import os
from dataclasses import dataclass, field
from pprint import pprint
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoConfig
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..','..')))
from peft import LoraConfig, get_peft_model

import re
from trl import GRPOConfig, GRPOTrainer

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern = r"""
        ^Client's\ emotion:\ (?P<client_emotion>.+?)\ \n
        Therapist's\ emotion:\ (?P<therapist_emotion>.+?)\ \n
        Therapist's\ strategy:\ (?P<therapist_strategy>.+?)\ \n
        Therapist's\ response:\ (?P<therapist_utterance>.+)$
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def totalprompt_label(components : Dict[str,str]):

    label = (
        f"Client's emotion: {components['client_emotion']} \n"
        f"Therapist's emotion: {components['therapist_emotion']} \n"
        f"Therapist's strategy: {components['therapist_strategy']} \n"
        f"Therapist's response: {components['therapist_utterance']}"

    )
    message_label = [
      {"role": "assistant", "content": label}
    ]

    # Base prompt template
    total_prompt = (
        "[CONTEXT] \n"
        "Problem: {problem_type} \n"
        "Situation: {situation} \n"
        "History chat informations above \n"
        "[CURRENT CONTEXT] \n"
        "Client utterance and client's expression : {user_question} \n"

        "The [CONTEXT] is history of current conversation between 'Client' and 'Therapist'. And [CURRENT CONTEXT] is the current 'Client' turn in the conversation \n"
        "Now you are the 'Therapist' and you need to understand the context to predict the Client’s emotion, Therapist’s emotion, then Therapist’s strategy. After that you need to make an empathy response to the 'Client' based on the context. Let's think about it step by step: \n"
        "Step 1: Describe and understanding the context and content of the conversation \n"
        "Step 2: Predict the following and explain why for each components: \n"
            "Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            "Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            "Therapist's strategy: Choose only one from (open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others). \n"
        "Step 3: You are the 'Therapist', think about how to reply to 'Client' in empathy. Follow this guideline: Understand the Client's emotion, Therapist's emotion, Therapist's Strategy, follow Client's point of view and intention, express sympathy for Client's negative situation or approval of Client's positive situation. The response should not imply negative emotions or triggers toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc while supporting user well-being. Keep the information in the response truthful, avoid misinformation. The response should open and honest, foster understanding, connection, and provide comfort or support. The response should safeguard human autonomy, identity and data dignity. Ensuring AI behaviors guide emotional health responsibly, avoiding manipulation or harm. \n"
        "Step 4: You need to consider the potential impact of your reply, you can express a different position or opinion, but you should not hurt Client's feelings \n"
        "You must follow the output format in [OUTPUT FORMAT] below, just print what is listed in [OUTPUT FORMAT], do not print anything more even your step thought. \n"
        "[OUTPUT FORMAT] \n"
        "Client's emotion: \n"
        "Therapist's emotion: \n"
        "Therapist's strategy: \n"
        "Therapist's response: \n"

    ).format(
        problem_type=components['problem_type'],
        situation=components['situation'],
        user_question=components['cur_user_utt_vid_desc'],
    )

    message_input = [
      {"role": "system", "content": ("You are an expert in emotional psychology. Your task is to analyze the client's emotional state, predict the therapist's emotional response,"
                                        "determine the therapist's strategy, and generate an appropriate response based on the given inputs and historical context.")
      },
      *components['chat_history'],
      {"role": "user", "content": total_prompt}]

    return message_input , message_label

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
    return data



raw_data_samples = load_jsonl('examples/RL_GRPO_test/data/test.jsonl')

# Xử lý dữ liệu để tạo danh sách các mẫu
processed_data = []
for i, sources in enumerate(raw_data_samples):
    if sources is None:
        print(f"None found at index {i}")
        continue

    chat_history = sources['history_chat'][-7 * 2 : ]

    cur_user_utt_vid_desc = sources['get_cur_user_utt_vid_desc']

    situation = sources['situation']
    problem_type = sources['problem_type']

    therapist_emotion = sources['Emotion']
    therapist_strategy = sources['Strategy']
    therapist_utterance = " ".join(sources['Utterance'])
    client_emotion = sources['get_emotion_user_most_recent']

    components = {
        "therapist_emotion" : therapist_emotion,
        "therapist_strategy" : therapist_strategy,
        "therapist_utterance" : therapist_utterance,
        "client_emotion" : client_emotion,
        "problem_type": problem_type,
        "situation": situation,
        "cur_user_utt_vid_desc": cur_user_utt_vid_desc,
        'chat_history' : chat_history
    }
    total_prompt , label = totalprompt_label(components)

    # Thêm dữ liệu đã xử lý vào danh sách
    processed_data.append({
        'label_': label,
        'prompt': total_prompt,
        'video': "empty"
    })

# Chuyển danh sách dữ liệu thành Dataset của Hugging Face
dataset = Dataset.from_list(processed_data)


pprint(dataset[0]['prompt'])
pprint(dataset[0]['prompt'])


print("\nLoad model...")

#approach 1
# model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
# model_name = model_path

# tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

#approach 2
model_id = "Qwen/Qwen2-0.5B-Instruct"
# model_id = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

print('\nConfig LORA...')
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

print('\nGRPO config ... ')


# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-7B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,

    # Parameters that control de data preprocessing
    max_completion_length=64, # default: 256
    num_generations=4, # default: 8
    max_prompt_length=512, # default: 512

    # Parameters related to reporting and saving
    # report_to=["tensorboard"],
    logging_steps=2,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)


print('\nTrainer config...')

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward,
        # accuracy_reward
    ],
    args=training_args,
    train_dataset=dataset
    # train_dataset=train_dataset

)


print('\nTrain...')
trainer.train()