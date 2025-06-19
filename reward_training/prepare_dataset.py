import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
import json
from tqdm import tqdm

# Import the dataset module from the correct path
from smes_ppo.dataset import get_dataset

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True  # Required for some models like Qwen
    )
    
    return model, tokenizer


def calculate_rouge_score(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score['rougeL'].fmeasure


def prepare_dataset_for_reward_training(data_path, model_name="Qwen/Qwen2-7B-Instruct", threshold=0.3, batch_size=4):
    dataset = get_dataset(data_path)
    
    model, tokenizer = load_model(model_name)
    
    reward_training_data = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
        batch = dataset[i:i+batch_size]
        
        prompts = batch["prompt"]
        ground_truths = batch["completion"]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_ground_truths = []
        for truth in ground_truths:
            batch_ground_truths.append(truth[0]["content"])
        
        # Calculate ROUGE-L scores and prepare data for reward training
        for j in range(len(decoded_outputs)):
            model_output = decoded_outputs[j]
            reference = batch_ground_truths[j]
            
            rouge_score = calculate_rouge_score(model_output, reference)
            
            # If the score is below the threshold, add to reward training data
            if rouge_score < threshold:
                reward_training_data.append({
                    "chosen": reference,
                    "rejected": model_output,
                })
    
    print(f"Created {len(reward_training_data)} pairs for reward training")
    return reward_training_data

def load_reward_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    dataset = datasets.Dataset.from_list(data)
    dataset = dataset.shuffle(seed=42)
    return dataset

if __name__ == "__main__":
    data_path = "../test.jsonl"
    model_name = "Qwen/Qwen2-7B-Instruct"
    output_path = "../test_reward_data.json"
    threshold = 0.3  # ROUGE-L score threshold
    
    reward_data = prepare_dataset_for_reward_training(
        data_path=data_path,
        model_name=model_name,
        threshold=threshold
    )
    
    with open(output_path, 'w') as f:
        json.dump(reward_data, f, indent=2)
    print(f"Saved reward training data to {output_path}")