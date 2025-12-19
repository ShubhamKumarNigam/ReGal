# Importing packages
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# wandb initialization
wandb.init(project="wandb_project_name")

# Step-1: Train and Validation Dataset Generation
df_train = pd.read_csv("path_to_train_csv")  # Path to TRAIN.CSV
df_val = pd.read_csv("path_to_val_csv")  # Path to VAL.CSV

print("Train data size: ", df_train.shape[0])
print("Validation data size: ", df_val.shape[0])

train_data = Dataset.from_pandas(df_train)
val_data = Dataset.from_pandas(df_val)

# Step-2: Model Training
model_id = "meta-llama/Llama-3.2-3B"  # Model Path (local or HuggingFace)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Specific to Llama models
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# Function to count trainable parameters
def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')


# Print trainable parameters
print_trainable_parameters(model)

OUTPUT_DIR = "path_to_output_directory"  # Path to save the fine-tuned model

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",  # Using standard AdamW optimizer
    logging_steps=2,
    learning_rate=1e-4,
    fp16=True,  # Enabling 16-bit precision
    max_grad_norm=0.3,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    eval_steps=0.5,
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="wandb",
)

model.config.use_cache = False

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2060,  # Adjust sequence length based on available resources
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
peft_model_path = OUTPUT_DIR
trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

