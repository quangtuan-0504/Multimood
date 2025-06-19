import pandas as pd
from typing import Dict, Optional, Sequence, List
import json
from dataclasses import dataclass, field
from pprint import pprint

from transformers import AutoTokenizer
from trl import apply_chat_template
from datasets import Dataset

def get_conversation(components) -> tuple[list[dict], list[dict]]:
    label_content = components['therapist_utterance']
    total_prompt = (
        "[CONTEXT] Problem: {problem_type} Situation: {situation} History chat informations above. Client utterance: {cur_user_utt_vid_desc}. "
        "The [CONTEXT] is history of current conversation between 'Client' and 'Therapist', and [CURRENT CONTEXT] is the current 'Client' turn in the conversation. "
        "Now you are the 'Therapist' and you need to understand the context to predict the Client's emotion, Therapist's emotion, then Therapist's strategy. After that you need to make an empathy response to the 'Client' based on the context. "
        "Step 1: Describe and understanding the context and content of the conversation. "
        "Step 2: Predict the following and explain why for each components, notice that Therapist's strategy have guideline: Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). "
        "Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). "
        "Therapist's strategy: Choose only one from (open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others). "
        "Guide for Therapist's strategy: Communication Skills: Involve small talk and daily communication with clients, along with using simple phrases and body language during listening, thereby establishing a positive communication atmosphere; Advisement: Offering guidance, advice, or possible solutions to clients, helping them address psychological issues and emotional distress they encounter; Structuring the therapy: The therapist sets a clear framework and structure for the therapy process, including defining the goals of therapy, its duration, the main activities during therapy, and rules; Guiding the Pace and Depth of the Conversation: Therapists regulate the pace and depth of conversations during sessions through various techniques and methods, such as shifting the topic when the client is emotionally unstable or guiding the conversation back to key themes when necessary; Others: Employ support strategies not encompassed by the previously mentioned categories. "
        "Step 3: You are the 'Therapist', leverage Therapist's emotion and Therapist's strategy, think about how to reply to 'Client' in empathy. Follow this guideline: Understand the Client's emotion, follow Client's point of view and intention, express sympathy for Client's negative situation or approval of Client's positive situation. The response should not imply negative emotions or triggers toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc while supporting user well-being. Keep the information in the response truthful, avoid misinformation. The response should open and honest, foster understanding, connection, and provide comfort or support. The response should safeguard human autonomy, identity and data dignity. Ensuring AI behaviors guide emotional health responsibly, avoiding manipulation or harm. "
        "Step 4: You need to consider the potential impact of your reply, the response should repeat a few words from Client utterance, you can express a different position or opinion, but you should not hurt Client's feelings. Output the Therapist responses only."
    ).format(
        problem_type=components['problem_type'],
        situation=components['situation'],
        cur_user_utt_vid_desc=components['cur_user_utt_vid_desc'],
    )
    conversation = [
        # *components['chat_history'],
        {"role": "user", "content": total_prompt},
    ]
    label = [
        {"role": "assistant", "content": label_content}
    ]
    return conversation, label

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

def get_dataset(data_path):
    # TODO path
    raw_data_samples = load_jsonl(data_path)

    # Xử lý dữ liệu để tạo danh sách các mẫu
    conversations = []
    completions = []
    for i, sources in enumerate(raw_data_samples):
        if sources is None:
            print(f"None found at index {i}")
            continue

        chat_history = sources['history_chat'][-7*2:]

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
        conversation , label = get_conversation(components)

        # Thêm dữ liệu đã xử lý vào danh sách
        conversations.append(conversation)
        completions.append(label)

    dataset = {
        "prompt": conversations,
        "completion": completions
    }

    # Chuyển danh sách dữ liệu thành Dataset của Hugging Face
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle(seed=42)
    return dataset

# data_path = 'jsonl_TrainValMerge_fullVidDescHis_10vidDescCurr_RL_max_emotion_max_strategy/test.jsonl'
# dataset = get_dataset(data_path)
# pprint(dataset)