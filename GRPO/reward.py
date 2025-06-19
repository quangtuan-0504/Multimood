import pandas as pd
import requests
import json
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import datetime
from dotenv import load_dotenv, find_dotenv
import random
from pprint import pprint
import shutil
from FlagEmbedding import BGEM3FlagModel
import torch



def compute_bge_score(sentences_1, sentences_2 , model_bgem3):
            sentence_pairs = [[sentences_1, sentences_2]]
            bge_score = model_bgem3.compute_score(sentence_pairs, 
                                        max_passage_length=128,
                                        weights_for_different_modes=[1 , 0.3 ,1])
            return bge_score['colbert+sparse+dense'][0]

print('\nLoad BGE M3 Model ...')
model_bgem3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, return_dense=True, return_sparse=True, return_colbert_vecs=True) 
def get_reward_similarity_with_bgem3(prompts , completions,**kwargs):
   
    # print("prompt len ids:", prompt)
    # print(kwargs['ground_truths'])
    print("completions len:", len(completions))
    # format_rewards = format_reward(completions, **kwargs)
    # print("format reward:", format_rewards)
    print('label:' , kwargs['ground_truths'][0][0]['content'])
    print("completions: ")
    for completion in completions:
        print("predict:",completion[0]['content'])
    print("-----\n")
    reward_lst = []
   
    for i in range(len(completions)):
        score = compute_bge_score(completions[i][0]['content'],kwargs['ground_truths'][i][0]['content'], model_bgem3)
        reward_lst.append(score)
    print("reward_lst...:",reward_lst)
    return reward_lst

