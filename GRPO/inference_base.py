from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoConfig
from dataloader import get_dataset
from trl import apply_chat_template
from pprint import pprint
import torch
import json
import time


def generate_with_reasoning(prompt, model,tokenizer,max_prompt_tokens = 4096-100):
    prompt = apply_chat_template({'prompt':prompt}, tokenizer)['prompt']

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = {
        'input_ids': input_ids['input_ids'][:,-max_prompt_tokens:] ,
        'attention_mask': input_ids['attention_mask'][:,-max_prompt_tokens:]
    }
    print("previous cut:",input_ids['input_ids'][:,-max_prompt_tokens:].shape, 'after cut:', input_ids['input_ids'].shape)
    # Generate text without gradients
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**input_ids, max_new_tokens = 100 , max_length=4096, temperature = 0.02)
    end_time = time.time()

    generated_tokens = output_ids[0][input_ids["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)


    # Get inference time
    inference_duration = end_time - start_time

    # Get number of generated tokens
    num_input_tokens = input_ids["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens

model_id = 'Qwen/Qwen2-0.5B-Instruct'
trained_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

trained_tokenizer = AutoTokenizer.from_pretrained(model_id)

data_path = '/workspace/SMES_Therapy_framework/jsonl_TrainValMerge_noDescPerTurn_noDescCurr_onlyUtt_RL_max_emotion_max_strategy/test.jsonl'
dataset = get_dataset(data_path)
prompt = dataset[10]['prompt']

print(type(prompt))
generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(prompt,trained_model,trained_tokenizer)
pprint(generated_text)
print('#####')
pprint(dataset[10]['ground_truths'])