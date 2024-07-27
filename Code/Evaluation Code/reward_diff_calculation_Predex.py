import pandas as pd
df_merged=pd.read_csv('llama2_ppo_inference_PredEx_data.csv')

df_merged

import torch
from transformers import pipeline
device = 0 if torch.cuda.is_available() else "cpu"
reward_model = pipeline("text-classification", model="L-NLProc/PredEx_InLegalBert_Pred", device=device)

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 4
}

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

compare_results = {}

df_merged['LLAMA_2_7b_Vanilla'] = df_merged['LLAMA_2_7b_Finetuned'].astype(str)
df_merged['llama2_ppo_predictions'] = df_merged['llama2_ppo_predictions'].astype(str)


compare_results["sft_query"] = df_merged['Case Description'].apply(lambda x: ' '.join(x.split()[-100:]) if len(x.split()) > 100 else x)
compare_results["ppo_query"] = df_merged['Case Description'].apply(lambda x: ' '.join(x.split()[-100:]) if len(x.split()) > 100 else x)

compare_results["response_before"] = df_merged['LLAMA_2_7b_Finetuned'].apply(lambda x: ' '.join(x.split()[:100]) if len(x.split()) > 100 else x)
compare_results["response_after"] = df_merged['llama2_ppo_predictions'].apply(lambda x: ' '.join(x.split()[:100]) if len(x.split()) > 100 else x)



# Sentiment analysis of query/response pairs before/after.
texts_before = [d + s for d, s in zip(compare_results["sft_query"], compare_results["response_before"])]



rewards_before = reward_model(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[0]["score"] for reward in rewards_before]

texts_after = [d + s for d, s in zip(compare_results["ppo_query"], compare_results["response_after"])]


rewards_after = reward_model(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[0]["score"] for reward in rewards_after]

compare_results["query"] = df_merged['Case Description']
compare_results["response_before"] = df_merged['LLAMA_2_7b_Finetuned']
compare_results["response_after"] = df_merged['llama2_ppo_predictions']

pd.set_option('display.max_colwidth', 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)
df_compare_results_sorted

def create_final_column(row):
    if float(row['reward_diff']) >= 0 or float(row['reward_diff']) == 0.0:
        return row['response_after']
    else:
        return row['response_before']

df_compare_results_sorted['final'] = df_compare_results_sorted.apply(create_final_column,axis=1)

df_compare_results_sorted

df_compare_results_sorted.to_csv('PPO-Predex-Llama2-Result_compare-vanilla-and-ppo.csv')