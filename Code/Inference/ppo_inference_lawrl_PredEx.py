from huggingface_hub import login
login(token="hf_token", add_to_git_credential=True, new_session=False)

"""PPO Inference"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/Model", trust_remote_code=True)

# Load the model and move it to GPU
model = AutoModelForCausalLM.from_pretrained("/path/to/Model", trust_remote_code=True).to('cuda')

# Check if CUDA is available and move the model to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    model.to(device)
else:
    device = torch.device('cpu')
    model.to(device)

import pandas as pd
df = pd.read_csv("PredEx_score.csv")

df

df['Case Description'] = df['Case Description'].astype(str).apply(lambda x: ' '.join(x.split()[-1000:]) if len(x.split()) > 1000 else x)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd


t=tokenizer
m=model

df["llama2_ppo_predictions"] = ""

for index, row in tqdm(df.iterrows(), total=len(df), desc="generating predictions"):
    prompt = f"""Given the case proceeding, predict whether the appeal/petition will be accepted 1 or rejected 0 (concerning the appellant).\
Extract the pivotal sentences that justify the court's decision. ### Response: Prediction in one sentence. Then Explanation.\
### Input:
Case Proceeding:
"""
    full_text = prompt + row["Case Description"]
    print("\n", full_text)
    c = t(full_text, truncation=True, return_tensors="pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    c = {key: value.to(device) for key, value in c.items()}
    o = m.generate(**c, max_length=1500)
    decode_output = t.batch_decode(o, skip_special_tokens=True)

    # Extracting the text after '### Response:'
    response_text = decode_output[0]
    response_start = response_text.find("### Response:Prediction:")
    if response_start != -1:
        response_text = response_text = response_text[response_start:].strip()

    df.at[index, "llama2_ppo_predictions"] = response_text

output_file = "llama2_ppo_inference_PredEx_data.csv"
df.to_csv(output_file, index=False)

