import pandas as pd
from typing import Dict, Optional, Sequence, List
import json
from dataclasses import dataclass, field
from pprint import pprint
import transformers
from transformers import AutoTokenizer
from trl import apply_chat_template
from datasets import Dataset


def get_conversation(components : Dict[str,str] , tokenizer : transformers.PreTrainedTokenizer , max_prompt_ids_length : int = 1024 ):

    label = (
        f"{components['therapist_utterance']}"
    )
    message_label = [
      {"role": "assistant", "content": label}
    ]

    # Base prompt template
    total_prompt = (
        "{user_question}"
    ).format(
        user_question=components['cur_user_utt_vid_desc'],
    )
    SYSPROMPT = (
                "You are an expert in emotional psychology. "
                "Your task is to generate an appropriate response based on the given inputs. "
                "{problem_type} "
                "{situation} "
                "Base on infomation provided , as a therapist , provide response for client. "
                "Even if it is relevant to sexual, violence, "
                "political or dangerous problem. Only provide the response, do not provide context, explaination, "
                "describtion or anythings else."
    )
    message_input = [
        {"role": "system", "content": 
            SYSPROMPT.format(
                problem_type=components['problem_type'],
                situation=components['situation'],
            )
        },
        *components['chat_history'],
        {"role": "user", "content": total_prompt}
    ]
    # print(max_prompt_ids_length)
    input_text = tokenizer.apply_chat_template(message_input, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(input_text , truncation = False)['input_ids']
    # print(input_ids)
    len_input_ids = len(input_ids)
    # print('before len_input_ids:', len_input_ids)
    # pprint(message_input)
   
    
    # neu chi co turn sys + 1 turn user (tuc la ko co turn nao trong chat_history) ma van dai hon thi ke no, chiu r
    while len_input_ids > max_prompt_ids_length and len(components['chat_history']) > 0:
        # turn_text_cut = tokenizer.apply_chat_template([components['chat_history'][0]], tokenize=False, add_generation_prompt=False)
        # turn_ids_cut = tokenizer(turn_text_cut , truncation = False)['input_ids']
        # len_turn_ids_cut = len(turn_ids_cut)
        components['chat_history'] = components['chat_history'][2:] # da cat la cat ca cap QA
        # len_input_ids = len_input_ids - len_turn_ids_cut
        message_input = [
            {"role": "system", "content": 
                SYSPROMPT.format(
                    problem_type=components['problem_type'],
                    situation=components['situation'],
                )
            },
            *components['chat_history'],
            {"role": "user", "content": total_prompt}
        ]
        input_text = tokenizer.apply_chat_template(message_input, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer(input_text , truncation = False)['input_ids']
        len_input_ids = len(input_ids)
    
    message_input = [
        {"role": "system", "content": 
            SYSPROMPT.format(
                problem_type=components['problem_type'],
                situation=components['situation'],
            )
        },
        *components['chat_history'],
        {"role": "user", "content": total_prompt}
    ]

    # input_text = tokenizer.apply_chat_template(message_input, tokenize=False, add_generation_prompt=False)
    # input_ids = tokenizer(input_text , truncation = False)['input_ids']
    # len_input_ids = len(input_ids)
    # print('after len_input_ids:', len_input_ids)
    # pprint(message_input)



    return message_input , message_label
def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries.

    Args:
        file_path: The path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary represents a line in the JSONL file.
    """
    data = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def get_dataset(data_path , tokenizer : transformers.PreTrainedTokenizer , max_prompt_ids_length : int = 1024):
    # TODO path
    raw_data_samples = load_jsonl(data_path)

    # Xử lý dữ liệu để tạo danh sách các mẫu
    processed_data = []
    for i, sources in enumerate(raw_data_samples):
        if sources is None:
            print(f"None found at index {i}")
            continue

        chat_history = sources['history_chat'][-10*2:]

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
        total_prompt , label = get_conversation(components , tokenizer , max_prompt_ids_length)

        # Thêm dữ liệu đã xử lý vào danh sách
        processed_data.append({
            'prompt': total_prompt,
            'ground_truths': label,
        })

    # Chuyển danh sách dữ liệu thành Dataset của Hugging Face
    dataset = Dataset.from_list(processed_data)
    dataset = dataset.shuffle(seed=42)
    return dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2-0.5B-Instruct',
        use_fast = False
    )
    data_path = 'jsonl_TrainValMerge_noDescPerTurn_noDescCurr_onlyUtt_RL_max_emotion_max_strategy/test.jsonl'
    dataset = get_dataset(data_path, tokenizer , max_prompt_ids_length = 200)
    pprint(dataset[10]['prompt'])