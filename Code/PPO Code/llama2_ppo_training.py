from huggingface_hub import login
login(token="HUGGING FACE TOKEN", add_to_git_credential=True, new_session=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-2-7b-chat-hf'

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer


device = 0 if torch.cuda.is_available() else "cpu"
# Use a pipeline as a high-level helper
from transformers import pipeline
reward_model = pipeline("text-classification", model="L-NLProc/PredEx_InLegalBert_Pred", device=device)

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM


ppo_model_wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(model1,torch_dtype=torch.float32, is_trainable=True)

ref_model_wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(model1,torch_dtype=torch.float32, is_trainable=True)

import pandas as pd

df = pd.read_csv("train.csv")
df.head()

df['Input'] = df['Input'].fillna('')

df['Input'] = df['Input'].apply(lambda x: ' '.join(x.split()[-100:]) if len(x.split()) > 100 else x)


def build_dataset(model_name):

    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.

    Returns:
    - dataset: Preprocessed dataset containing input_id and query.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(sample):

        # Wrap each dialogue with the instruction.
        prompt = f"""Given the case proceeding, predict whether the appeal/petition will be accepted 1 or rejected 0 (concerning the appellant). \
Extract the pivotal sentences that justify the court's decision.

### Input:
Case Proceeding: {sample["Input"]}

### Response: Prediction in one sentence. Then Explanation.
"""
        sample["input_ids"] = tokenizer.encode(prompt, max_length=512, truncation=True)


        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return sample

    # Tokenize each dialogue.
    from datasets import Dataset

    dataset =Dataset.from_pandas(df)
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    return dataset

dataset = build_dataset(model_name=model_name)

print(dataset)

for column in dataset.features:
    print(f"Data in column '{column}':")
    # Displaying the first few entries of each column
    for i in range(min(1, len(dataset))):
        print(dataset[i][column])
    print("\n")

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


from trl import PPOTrainer, PPOConfig

learning_rate=1.41e-5
max_ppo_epochs=1
mini_batch_size=2
batch_size=4

config = PPOConfig(
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

ppo_trainer = PPOTrainer(config=config,
                         model=ppo_model_wrapped,
                         ref_model=ref_model_wrapped,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         data_collator=collator)

import torch

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 500,  # Adjust this value as needed
}

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 4
}

from trl.core import LengthSampler
output_min_length = 100
output_max_length = 500
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# Ensure GPU is available and being used
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from transformers import GenerationConfig
from tqdm import tqdm
from trl.core import LengthSampler
import torch
import time
from torch.cuda.amp import autocast, GradScaler
import gc

# Add a new padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Update the model to recognize the new special tokens
model1.resize_token_embeddings(len(tokenizer))

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Enable mixed precision training
scaler = GradScaler()

# Set batch size and mini batch size
ppo_trainer.config.batch_size = 4
ppo_trainer.config.mini_batch_size = 2

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    else:
        return batch

# Assuming reward_model.tokenizer is the correct tokenizer for the reward model
def truncate_text(text, max_length=500):
    encoded_text = reward_model.tokenizer.encode(text, add_special_tokens=False)
    if len(encoded_text) > max_length:
        encoded_text = encoded_text[:max_length]
    return reward_model.tokenizer.decode(encoded_text, skip_special_tokens=True)

epochs = 1

# Added counter and save interval
batch_counter = 0
save_interval = 100

for epoch in tqdm(range(epochs), desc="Epoch"):
    for batch in tqdm(ppo_trainer.dataloader, desc="Batch"):

        batch = to_device(batch, ppo_trainer.accelerator.device)
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        with autocast():
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        truncated_texts = [truncate_text(text) for text in texts]

        pipe_outputs = reward_model(truncated_texts)

        # print("Pipe outputs:", pipe_outputs)
        rewards = [torch.tensor(output['score'], device=ppo_trainer.accelerator.device) for output in pipe_outputs]

        # Run PPO step
        with autocast():
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        ppo_trainer.log_stats(stats, batch, rewards)

         # Increment the batch counter
        batch_counter += 1

        # Save the model every 100 batches
        if batch_counter % save_interval == 0:
            ppo_trainer.save_pretrained("llama2_ppo_model_checkpoint")
            print(f"Model saved at batch {batch_counter}")

        # Clear unnecessary tensors from GPU memory
        del query_tensors, response_tensors, rewards
        torch.cuda.empty_cache()
        gc.collect()

print(f'kl div loss: {stats["objective/kl"]}')
print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
print('-' * 100)

# Save model
ppo_trainer.save_pretrained("llama2_ppo_model")
print("Final model saved")

from huggingface_hub import login
login(token="HUGGING FACE TOKEN", add_to_git_credential=True, new_session=False)

ppo_trainer.push_to_hub("PATH TO HUGGING FACE REPOSITORY")

