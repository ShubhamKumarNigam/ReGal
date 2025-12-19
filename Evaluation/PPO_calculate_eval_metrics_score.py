import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
import blanc
from bert_score import score
import nltk
from blanc import BlancHelp

# Download necessary NLTK resources for tokenization
nltk.download('punkt')

# Load the dataframe
df = pd.read_csv("REGAL_EVAL_DATASET/PPO_model_test_infer_1024_maxtokens.csv")
print("Actual data size:",df.shape[0])

# Drop rows where either OUTPUT or llama_3.2_OUTPUT is NaN
df = df.dropna(subset=['OUTPUT', 'llama_3.2_OUTPUT'])

# Ensure both columns are strings
df['OUTPUT'] = df['OUTPUT'].astype(str)
df['llama_3.2_OUTPUT'] = df['llama_3.2_OUTPUT'].astype(str)

print("Data size after dropping null values:",df.shape[0])

# Initialize variables to store the scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
bleu_scores = []
meteor_scores = []
bert_scores = []
blanc_scores = []

# Initialize Rouge scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize BLANC scorer
bl = BlancHelp(device='cuda', inference_batch_size=128)

# Loop through the rows in the dataframe
for _, row in df.iterrows():
    actual_output = row['OUTPUT'].strip()
    wrapper_output = row['llama_3.2_OUTPUT'].strip()
    
    if not actual_output or not wrapper_output:  # Ignore empty strings
        continue

    # ROUGE scores
    rouge_scores = scorer.score(actual_output, wrapper_output)
    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    # BLEU score
    bleu_scores.append(bleu_score.sentence_bleu([actual_output.split()], wrapper_output.split()))

    # METEOR score
    actual_tokens = nltk.word_tokenize(actual_output)
    wrapper_tokens = nltk.word_tokenize(wrapper_output)
    meteor_scores.append(meteor_score.single_meteor_score(actual_tokens, wrapper_tokens))

    # BERTScore
    P, R, F1 = score([wrapper_output], [actual_output], lang='en')
    bert_scores.append(F1.item())

    # Blanc Score
    blanc_scores.append(bl.eval_once(actual_output, wrapper_output))

# Calculate the average of each score across all valid rows
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
average_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
average_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
average_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
average_blanc = sum(blanc_scores) / len(blanc_scores) if blanc_scores else 0

# Print the average scores
print(f"Average ROUGE-1: {average_rouge1:.4f}")
print(f"Average ROUGE-2: {average_rouge2:.4f}")
print(f"Average ROUGE-L: {average_rougeL:.4f}")
print(f"Average BLEU: {average_bleu:.4f}")
print(f"Average METEOR: {average_meteor:.4f}")
print(f"Average BERTScore: {average_bert:.4f}")
print(f"Average BLANC: {average_blanc:.4f}")

# Add the scores to the dataframe
df['ROUGE-1'] = rouge1_scores
df['ROUGE-2'] = rouge2_scores
df['ROUGE-L'] = rougeL_scores
df['BLEU'] = bleu_scores
df['METEOR'] = meteor_scores
df['BERTScore'] = bert_scores
df['BLANC'] = blanc_scores

# Save the updated dataframe
df.to_csv('evaluated_output_with_avg_PPO.csv', index=False)

