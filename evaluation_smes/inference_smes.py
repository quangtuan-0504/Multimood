import sys
import os
sys.path.append('/')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__))))
from framework import model_init, mm_infer
from framework.utils import disable_torch_init
from dataloader_eval import LazySupervisedDataset,data_args
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

def save_lst_to_jsonl(lst, filename):
    """Saves a Pandas DataFrame to a JSONL file.

    Args:
        df: The DataFrame to save.
        filename: The name of the JSONL file to create.
    """
    with open(filename, 'w') as f:
        for ele in lst:
            json.dump(ele, f)  # Convert each row to a dictionary and write as JSON
            f.write('\n')
      
def evaluate_smes(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    # put dataloader here
    # get per sample data (prompt,path to vdieo) from dataset , inference , and save predict, label
    # loop per sample data below.
    
    dataset_test = LazySupervisedDataset(data_path = args.dataset_path , tokenizer = tokenizer, data_args = data_args)
    # len(dataset_test)
    # print(type(dataset_test))

    # Extract model name from path for the output filename
    model_name = model_path.split('/')[-1]
    
    if not os.path.exists(args.res_folder):
        os.makedirs(args.res_folder)
    outputs = []
    # Use the model name in the output filename
    output_file = f'{args.res_folder}/{model_name}_{args.modal_type}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for data_point in tqdm(dataset_test):
            # Audio-visual Inference
            print('Note video_path:',data_point['video'],'\n')
            audio_video_path = data_point['video']
            if audio_video_path:
                preprocess = processor['audio' if args.modal_type == "a" else "video"]
                if args.modal_type == "a":
                    audio_video_tensor = preprocess(audio_video_path)
                else:
                    audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
                os.remove(audio_video_path)
            else:
                audio_video_tensor = {
                    'video': torch.zeros(data_args.num_frames, 3, data_args.image_size, data_args.image_size),
                    'audio' : torch.zeros(1, 16000 * 5)
                }  # Assuming 5 seconds of silence at 16kHz
            question = data_point['total_prompt']
            chat_history = data_point['text_his_ids']
            label = data_point['label']
            
            with open("model_arch.txt", "w") as model_f: 
                model_f.write(str(model))
                
                
            output = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                chat_history= chat_history, 
                tokenizer=tokenizer,
                temparature = 0.1,
                modal='audio' if args.modal_type == "a" else "video",
                do_sample=False,
                max_prompt_tokens = 4096,
            )
            outputs.append(output)
            row = {'label':label,'predict':output}
            # break
            f.write(json.dumps(row) + '\n')

        
    
    print(f"Results saved to: {output_file}")
    
    return len(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='jsonl test set path',default='dataset_path')
    parser.add_argument('--model-path', help='', required=False, default='model_path')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', default='av')
    parser.add_argument('--res-folder', help='path to file .jsonl predict', default='./eval_test/result')

    args = parser.parse_args()

    print('num answer : ', evaluate_smes(args))