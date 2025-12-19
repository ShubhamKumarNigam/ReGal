# ReGal: PPO-based Legal AI for Judgment Prediction and Summarization in India

<div align="center">
<img src="Assets/ReGal_Logo.png" width="100" alt="ReGal" />
<br>

**ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India** (AAAI 2025)

</div>

![task_overview](https://github.com/ShubhamKumarNigam/ReGal/raw/main/Assets/task_overview.jpg)

<p align="center">
  <a href="https://github.com/ShubhamKumarNigam/ReGal"><b>[🌐 GitHub]</b></a> •
  <a href="https://arxiv.org/abs/your-paper-link"><b>[📄 ArXiv]</b></a> •
  <a href="https://huggingface.co/collections/L-NLProc/regal-models"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/collections/L-NLProc/regal-datasets"><b>[🤗 HF Dataset]</b></a>
</p>

<p align="center">
  This is the official implementation of the paper:
</p>

<p align="center">
  <a href="https://sites.google.com/view/shubhamkumarnigam">Shubham Kumar Nigam</a>, <a href="#">Tanuj Tyagi</a>, <a href="#">Siddharth Shukla</a>, <a href="#">Aditya Kumar Guru</a>, <a href="#">Balaramamahanthi Deepak Patnaik</a>, <a href="#">Danush Khanna</a>, <a href="#">Noel Shallum</a>, <a href="https://sites.google.com/view/kripabandhughosh-homepage/home">Kripabandhu Ghosh</a>, and <a href="https://www.cse.iitk.ac.in/users/arnabb/">Arnab Bhattacharya</a>:
</p>

<p align="center">
  <a href="https://arxiv.org/abs/your-paper-link"><strong>ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India</strong></a> (AAAI 2025)
</p>

## Overview

ReGal presents an early exploration of Reinforcement Learning (RL) methodologies for legal AI in the Indian context. We introduce a framework that integrates Multi-Task Instruction Tuning with Reinforcement Learning from AI Feedback (RLAIF) using Proximal Policy Optimization (PPO). Our approach is evaluated across two critical legal tasks: **(i) Court Judgment Prediction and Explanation (CJPE)**, and **(ii) Legal Document Summarization**.

While ReGal underperforms compared to supervised and proprietary models, it provides valuable insights into the challenges of applying RL to legal texts, including reward model alignment, legal language complexity, and domain-specific adaptation. This work establishes a foundation for future improvements in RL-based legal AI systems.

If you have any questions about this work, please open a [GitHub issue](https://github.com/ShubhamKumarNigam/ReGal/issues) or email the authors at:

```
shubhamkumarnigam@gmail.com, tanujtyagiofficial@gmail.com, danush.s.khanna@gmail.com
```

---

## Key Contributions

1. **First PPO Application**: One of the first applications of PPO-based reinforcement learning in Indian legal judgment prediction and summarization.
2. **Comprehensive Analysis**: Empirical and qualitative evidence on the limitations of PPO for legal NLP tasks.
3. **Future Directions**: A clear path for more effective legal-AI pipelines integrating RLHF, human feedback, and domain-adapted modeling.

---

## Getting Started

### General Instructions

Ensure you have the necessary hardware and software requirements in place to replicate our experimental setup. Follow the steps below to configure your environment for optimal performance.

### Recommended Hardware Configuration

**Hardware Specifications:**
- Two cores of [NVIDIA A100-PCIE-40GB](https://www.nvidia.com/en-gb/data-center/a100/) with 126GB RAM for instruction fine-tuning and PPO training.
- Alternatively, [Vast.ai](https://vast.ai/) A100 80GB GPU rental (~$100 for complete training).
- Google Colab Pro with A100 accelerator is sufficient for inference and baseline experiments.

### Recommended Software Configuration

**Software Setup:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Additional dependencies for PPO training (TRL, bitsandbytes)
- Install necessary drivers and libraries for GPU acceleration.

### Installation

```bash
git clone https://github.com/ShubhamKumarNigam/ReGal.git
cd ReGal
pip install -r requirements.txt
```

---

## Tasks & Datasets

### Task 1: Court Judgment Prediction and Explanation (CJPE)

**Task 1A - Judgment Prediction:** Predict whether an appeal/petition from the Supreme Court of India was accepted (1) or rejected (0).

**Task 1B - Rationale Explanation:** Generate natural language explanations supporting the predicted outcome.

**Dataset - PredEx:** The largest annotated dataset for Indian legal judgment prediction and explanation with 15,222 Supreme Court judgment documents.

| Metric | Value |
|--------|-------|
| Train Documents | 12,178 |
| Test Documents | 3,044 |
| Avg. Tokens per Document | 4,586 |
| Max Tokens | 117,733 |
| Acceptance Rate | 53.44% |

### Task 2: Legal Judgment Summarization

Generate concise, abstractive summaries capturing essential components (background, legal issues, arguments, verdict) from full judgment texts.

**Dataset - In-Abs:** Expert-curated abstractive summaries from Indian Supreme Court judgments.

| Metric | Value |
|--------|-------|
| Total Documents | 7,130 |
| Train/Test Split | 7,030 / 100 |
| Avg. Document Size | 4,376.98 words |
| Avg. Summary Size | 842.52 words |
| Compression Ratio | 0.235 |

---

## Methodology

### Two-Stage Approach

1. **Stage 1: Supervised Fine-Tuning (SFT)**
   - Fine-tune Llama-2-7B on labeled legal data
   - Serves as reference policy (π_SFT) for PPO training

2. **Stage 2: Proximal Policy Optimization (PPO)**
   - Use task-specific reward models to guide policy optimization
   - Minimize PPO loss to align outputs with desired legal outcomes
   - Apply KL-divergence penalty to prevent deviation from SFT baseline

### Reward Models

- **CJPE Reward Model**: Fine-tuned InLegalBERT classifier (binary rewards: 1 for correct prediction, 0 otherwise)
- **Summarization Reward Model**: ROUGE-based n-gram overlap + semantic similarity scoring

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Base Model | Llama-2-7B |
| Learning Rate | 1.41e-5 |
| Batch Size | 4 |
| Mini-batch Size | 2 |
| PPO Epochs | 1 |
| Output Length | 100-500 tokens |
| Clipping Parameter (ε) | 0.1 |
| GPU | NVIDIA A100 80GB |
| Total Training Cost | ~$100 |

---

## Evaluation Metrics

### Lexical-Based Metrics
- **ROUGE-1/2/L**: Recall-based n-gram overlap
- **BLEU**: Precision-based evaluation
- **METEOR**: Synonym and stemming-aware comparison

### Semantic-Based Metrics
- **BERTScore**: Semantic similarity using contextual embeddings
- **BLANC**: Semantic relevance assessment

---

## Results

### Judgment Prediction and Explanation Performance

<table>
<tr>
<td>

| Model | R1 | R2 | RL | BLEU |
|---|---|---|---|---|
| Gemini Pro | 0.31 | 0.24 | 0.26 | 0.08 |
| LLaMA-2 | 0.32 | 0.19 | 0.21 | 0.06 |
| LLaMA-2 SFT | 0.50 | 0.43 | 0.44 | 0.25 |
| **ReGal (Ours)** | **0.19** | **0.04** | **0.12** | **0.01** |

**PredEx Dataset**

</td>
<td>

| Model | R1 | R2 | RL | BLEU |
|---|---|---|---|---|
| GPT-3.5 Turbo | 0.54 | 0.43 | 0.45 | 0.28 |
| LLaMA-2 | 0.45 | 0.25 | 0.30 | 0.15 |
| LLaMA-2 SFT | 0.49 | 0.38 | 0.40 | 0.29 |
| **ReGal (Ours)** | **0.25** | **0.05** | **0.16** | **0.01** |

**ILDC Expert Dataset**

</td>
</tr>
</table>

### Legal Summarization Performance

| Method | R1 | R2 | RL | BLEU | METEOR |
|---|---|---|---|---|---|
| Vanilla Inference | 0.47 | 0.29 | 0.28 | 0.15 | 0.34 |
| SFT Inference | 0.44 | 0.24 | 0.24 | 0.12 | 0.34 |
| DPO Inference | 0.44 | 0.24 | 0.24 | 0.12 | 0.34 |
| **PPO Inference** | **0.41** | **0.21** | **0.22** | **0.10** | **0.31** |

**In-Abs Summarization Dataset**

---

## Key Findings & Challenges

### Reasons for Underperformance

1. **Objective Mismatch**: SFT baseline not fully optimized for legal reasoning
2. **Reward Model Limitations**: Difficulty capturing nuanced legal interpretations
3. **Legal Complexity**: Long, intricate documents with rich semantic references
4. **Training Data Constraints**: Limited diversity in legal reasoning patterns
5. **Hallucination Issues**: Model generates plausible but factually incorrect outputs
6. **Domain Pretraining Gap**: Lack of deep domain adaptation compared to GPT-3.5
7. **Hyperparameter Sensitivity**: Suboptimal tuning of learning rates and penalties
8. **Model Architecture**: Llama-2-7B may be undersized for complex legal tasks

### Hallucination Analysis

The ReGal framework exhibits significant hallucination issues, particularly when:
- Input facts are sparse or ambiguously phrased
- The model over-optimizes for weak reward patterns
- Outputs mimic style without substantive accuracy

Examples include fabricated legal principles, invented precedent citations, and claims unsupported by source documents.

---

## Ablation Study

### Base Model Variants

- **Phi-3 Mini**: Too small for complex legal texts; severe performance degradation
- **Llama-2-7B (Pretrained)**: Insufficient without legal domain fine-tuning
- **Llama-2-7B SFT**: Optimal balance of capacity and domain adaptation

### Reward Model Variants

- **Task-Specific Reward Model**: Best performance with domain-aligned scoring
- **General InLegalBERT**: Noisier feedback; degraded PPO optimization

**Conclusion**: Success hinges on strong initialization and precisely aligned reward functions.

---

## Future Work

1. **Better Reward Models**: Develop hallucination-aware and factuality-constrained reward signals
2. **Domain Pretraining**: Leverage domain-adaptive pretraining strategies for legal corpora
3. **Human-in-the-Loop**: Integrate human feedback to improve learning signals
4. **Advanced Architectures**: Explore larger models and specialized legal LLMs
5. **Factuality Constraints**: Augment PPO with stronger constraints on factual accuracy
6. **Cross-Jurisdiction Transfer**: Extend to multiple legal systems and languages

---

## Citation

If you use this work, please cite our paper:

```bibtex
@article{nigam2025regal,
  title={ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India},
  author={Nigam, Shubham Kumar and Tyagi, Tanuj and Shukla, Siddharth and Guru, Aditya Kumar and Patnaik, Balaramamahanthi Deepak and Khanna, Danush and Shallum, Noel and Ghosh, Kripabandhu and Bhattacharya, Arnab},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- IIT Kanpur, Manipal University Jaipur, Symbiosis Law School Pune, and IISER Kolkata
- Funded through academic research initiatives
- Special thanks to the legal experts who annotated the datasets

---

## Contact

For questions or collaborations, reach out to:

- **Shubham Kumar Nigam**: shubhamkumarnigam@gmail.com
- **Danush Khanna**: danush.s.khanna@gmail.com
- **Kripabandhu Ghosh**: kripaghosh@iiserkol.ac.in
- **Arnab Bhattacharya**: arnabb@cse.iitk.ac.in

---

**Repository**: [GitHub - ReGal](https://github.com/ShubhamKumarNigam/ReGal)
