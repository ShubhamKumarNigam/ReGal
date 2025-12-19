# LLaMA 3 Training with LLaMA-Factory

This directory contains configurations and data for training LLaMA 3 models using DPO (Direct Preference Optimization) and PPO (Proximal Policy Optimization) via the LLaMA-Factory framework.

## Setup

### 1. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 2. Prepare Data

The training data is derived from:
- **PredEx Dataset**: Court Judgment Prediction and Explanation (CJPE) task
- **IL-TUR Dataset**: Legal document summarization task

Both datasets have been converted to LLaMA-Factory format (instruction, input, chosen/rejected for DPO; instruction, input, output for PPO).

Copy the dataset registry to LLaMA-Factory:

```bash
cp Training/dataset_info.json LLaMA-Factory/data/dataset_info.json
```

Copy your dataset files to the LLaMA-Factory data directory:

```bash
cp /path/to/your/datasets/*.json LLaMA-Factory/data/
```

### 3. Copy Training Configurations

```bash
cp Training/*.yaml LLaMA-Factory/examples/train_lora/
```

### 4. Update Configuration Paths

Before training, update the following in the YAML files:

**For DPO** (`predex_llama3_lora_dpo.yaml`):
- `output_dir`: Your checkpoint output directory

**For PPO** (`predex_llama3_lora_ppo.yaml`):
- `model_name_or_path`: Path to your SFT checkpoint
- `reward_model`: Path to reward model (default: `L-NLProc/PredEx_InLegalBert_Pred`)
- `output_dir`: Your checkpoint output directory

## Training

### DPO Training

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/predex_llama3_lora_dpo.yaml
```

### PPO Training

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/predex_llama3_lora_ppo.yaml
```

## Configuration Details

### DPO
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- LoRA rank: 8
- Learning rate: 5.0e-6
- Max sequence length: 128,000 tokens
- Epochs: 3

### PPO
- Base: SFT checkpoint
- LoRA rank: 8
- Learning rate: 1.0e-5
- Max sequence length: 2,048 tokens
- Epochs: 3
- Reward model: `L-NLProc/PredEx_InLegalBert_Pred`

## Hardware Requirements

- GPU: 24GB+ VRAM (tested on A100 40GB, RTX 3090 24GB)
- RAM: 32GB+ system memory
- Storage: 50GB+ free space

## Inference

See [Inference/](Inference/) directory for inference scripts using trained models.

## References

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- PredEx Dataset: https://github.com/ShubhamKumarNigam/PredEx
- IL-TUR Dataset: https://exploration-lab.github.io/IL-TUR/
