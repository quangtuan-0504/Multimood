import pandas as pd
import requests
import json
from FlagEmbedding import BGEM3FlagModel
import torch
import os
import time




def get_score_from_bgem3(completion : str, ground_truth : str) -> int:
    url = "http://0.0.0.0:8000/compute-score/"

    payload = {
        "sentence_1": ground_truth,
        "sentence_2": completion
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result['score']
        else:
            print(f"Lỗi: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print("Không thể kết nối đến server. Vui lòng kiểm tra xem server đã chạy chưa.")
        return 0.01
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")
        return 0.01
      
def get_reward_similarity_with_bgem3(prompts, completions,**kwargs) -> list:
   
    print("completions len:", len(completions))
    print('label:' , kwargs['ground_truths'][0][0]['content'])
    print("completions: ")
    for completion in completions:
        print("predict:",completion[0]['content'])
    print("-----\n")
    reward_lst = []
   
    for i in range(len(completions)):
        score = get_score_from_bgem3(completions[i][0]['content'],kwargs['ground_truths'][i][0]['content'])
        time.sleep(0.25)
        reward_lst.append(score)
    print("reward_lst...:",reward_lst)
    return reward_lst