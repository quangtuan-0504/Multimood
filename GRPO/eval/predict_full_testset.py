from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from dataloader import get_dataset
from trl import apply_chat_template
from pprint import pprint
from tqdm import tqdm
import torch
import json
import time


def generate_with_reasoning(prompt, model,tokenizer,max_prompt_tokens = 4096-100):
    # Build the prompt from the dataset
    prompt = apply_chat_template({'prompt':prompt}, tokenizer)['prompt']
    # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # prompt = " ".join(entry["content"] for entry in prompt)

    # Tokenize and move to the same device as the model
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = {
        'input_ids': input_ids['input_ids'][:,-max_prompt_tokens:] ,
        'attention_mask': input_ids['attention_mask'][:,-max_prompt_tokens:]
    }
    print("previous cut:",input_ids['input_ids'][:,-max_prompt_tokens:].shape, 'after cut:', input_ids['input_ids'].shape)
    # Generate text without gradients
    start_time = time.time()
    with torch.no_grad():
        # output_ids = model.generate(**input_ids, max_length=4096, temperature = 0.1)
        output_ids = model.generate(**input_ids, max_new_tokens = 100 , max_length=4096, temperature = 0.02)
    end_time = time.time()

    # Cắt bỏ phần prompt dựa trên số token đầu vào
    generated_tokens = output_ids[0][input_ids["input_ids"].shape[1]:]
    # Giải mã chỉ phần sinh thêm
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Decode and extract model response
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Get inference time
    inference_duration = end_time - start_time

    # Get number of generated tokens
    num_input_tokens = input_ids["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens

def inference_test_set(model_id:str, data_path:str):
    print("Load Model ... \n")
    trained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    print("Load Tokenizer ... \n")
    trained_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False )

    print("Load Dataset Eval ... \n")
    dataset = get_dataset(data_path,trained_tokenizer)

    print("Start Predict ... \n")
    # Use the model name in the output filename
    if not os.path.exists('result_eval'):
        os.makedirs('result_eval')
    output_file = f'result_eval/{model_id.split("/")[-1]}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset):# lấy từng sample ra predict
            # print(sample)
            # break
            prompt = sample['prompt']

            generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(prompt,trained_model,trained_tokenizer)
            # print(type(generated_text))
            # print(generated_text)
            row = {
                'label':f"Therapist's response: {sample['ground_truths'][0]['content']}",
                'predict': f"Therapist's response: {generated_text}"
            }
            f.write(json.dumps(row) + '\n')
            # break
    print("End Predict ... ")
def main():
    # model_id = "/workspace/Models/RL_Qwen25_instruct_400steps"
    model_id = '/workspace/SMES_Therapy_framework/Qwen/Qwen2-0.5B-Instruct_grpo/checkpoint-10'
    data_path = '/workspace/SMES_Therapy_framework/jsonl_TrainValMerge_noDescPerTurn_noDescCurr_onlyUtt_RL_max_emotion_max_strategy/test.jsonl'
    inference_test_set(
        model_id = model_id,
        data_path= data_path
    )
if __name__ == "__main__":
    main()