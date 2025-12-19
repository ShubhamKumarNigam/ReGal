import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("path_to_input_csv")
#df = df.iloc[:5000]
print(df.shape[0])
# Replace 'your_access_token' with your actual token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("path_to_model")
model = AutoModelForCausalLM.from_pretrained("path_to_model").to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define preprocess_input function
def preprocess_input(text, input_length=2000, output_length=512):
    words = text.split()
    truncated_text = " ".join(words[:input_length])  # Truncate the text to the first 2000 words
    prompt = (
        f"### Instructions: Analyze the case proceeding and predict whether "
        f"the appeal/petition will be accepted (1) or rejected (0), and subsequently provide a complete explanation behind this prediction with important textual evidence from the case. "
        f"Please provide inference within {output_length} words.\n"
        f"### Case Proceeding: "
    )
    response = "### Prediction and explanation: "
    return prompt + truncated_text + response

# Open a CSV file to store the results incrementally
output_file = "path_to_output_csv"
# Write headers to the CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write headers
    writer.writerow(["INPUT", "OUTPUT", "PROCESSED_INPUT", "llama_3.2_OUTPUT"])

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    
        input_text = preprocess_input(row["Input"], output_length=512)
        input_ids = tokenizer(input_text, return_tensors='pt', truncation=True).input_ids.cuda()
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1024)
        generated_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_text):]
        # Print the generated text for debugging
        print(f"Row {idx}: Generated text = {generated_text}")

        # Write the result row to the CSV file
        writer.writerow([row["Input"], row["Output"], input_text, generated_text])

