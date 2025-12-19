from datasets import load_metric
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
nltk.download('punkt')
from nltk.tokenize import word_tokenize

bertscore = load_metric("bertscore",trust_remote_code=True)
meteor = load_metric("meteor",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
rouge = load_metric('rouge',trust_remote_code=True)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df1=pd.read_csv("llama2_ppo_inference_PredEx_data.csv")
df2=pd.read_csv("PPO-Predex-Llama2-Result_compare-vanilla-and-ppo.csv")

df2['Official Reasoning']=df1['Official Reasoning']
df2 = df2.rename(columns={'Official Reasoning': 'original_summary', 'final': 'generated_summary'})
df=df2

df

def calculate_bleu_score(candidate, references):
    candidate_tokens = nltk.word_tokenize(candidate)
    reference_tokens = [nltk.word_tokenize(ref) for ref in references]

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    return bleu_score

def metrics(actual, pred):
  predictions = [pred]
  references = [actual]
  metrics = {}
  #Rouge
  rouge_score = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
  metrics["rouge"] = [{
      "rouge1": rouge_score["rouge1"][0].fmeasure,
      "rouge2": rouge_score["rouge2"][0].fmeasure,
      "rougeL": rouge_score["rougeL"][0].fmeasure
  }]
  #BERT
  bert_score = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-uncased")
  metrics["bert"] = bert_score["f1"][0]
  #Meteor
  meteor_score = meteor.compute(predictions=predictions, references=references)
  metrics["meteor"] = meteor_score["meteor"]
  #BLEU
  bleu_score = calculate_bleu_score(predictions[0], references)
  metrics["bleu"] = bleu_score
  return metrics

df.head(2)

df['original_summary'][1]

df['generated_summary'][35]

from tqdm import tqdm
import json

df = df.dropna()
df

df['generated_summary'][36]

all_metrics = []
for i, row in tqdm(df.iterrows()):
    # Check if 'Output' is a string
    if isinstance(row['original_summary'], str):
        actual = row['original_summary']
        pred = row['generated_summary']
        metric = metrics(actual, pred)
        all_metrics.append(metric)
    else:
        # Handle cases where 'Output' is not a string
        # For example, you might want to skip or set a default value
        continue

# prompt: store list all metrics in json and save json

import json
with open("predex-ppo_Evaluation_Scores_LawRL", "w") as outfile:
    json.dump(all_metrics, outfile)

all_metrics

def avg(l):
  return sum(l)/len(l)

"""#Rouge"""

r1 = []
r2 = []
r3 = []
for m in all_metrics:
  r1.append(m['rouge'][0]['rouge1'])
  r2.append(m['rouge'][0]['rouge2'])
  r3.append(m['rouge'][0]['rougeL'])

print("Average R1: ", avg(r1))
print("Average R2: ", avg(r2))
print("Average R3: ", avg(r3))

"""#Blue

"""

blue = []
for m in all_metrics:
  blue.append(m['bleu'])

print("Average BLEU: ", avg(blue))

"""#Meteor"""

meteor = []
for m in all_metrics:
  meteor.append(m['meteor'])

print("Average meteor: ", avg(meteor))

"""#BERT"""

bert = []
for m in all_metrics:
  bert.append(m['bert'])

print("Average BERT: ", avg(bert))

"""#BLANC

"""

from blanc import BlancHelp, BlancTune

import nltk
nltk.download('punkt')

bl = BlancHelp(device='cuda', inference_batch_size=128)

def avg(l):
  return sum(l)/len(l)

def cal_BLANC(actual, pred):
  k = bl.eval_once(actual, pred)
  return k

all_blanc = []
for i,row in tqdm(df.iterrows()):
  # Check if 'Output' is a string
  if isinstance(row['original_summary'], str):
    pred = row['generated_summary']
    actual = row['original_summary']
    metric = cal_BLANC(actual, pred)
    all_blanc.append(metric)
  else:
    continue

print("Average BLANC: ", avg(all_blanc))

