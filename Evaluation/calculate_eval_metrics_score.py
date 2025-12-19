#M8_Wrapper_output_docgen_local_inference.csv

import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
import blanc
from bert_score import score
import nltk
from blanc import BlancHelp, BlancTune

# Download necessary NLTK resources for tokenization
nltk.download('punkt')

# Assuming you already have the dataframe `df`
df = pd.read_csv("REGAL_EVAL_DATASET/llama_3.2_16bit_QLoRA_SFT_infer_Over_predex_Test_merged.csv")  # Load your dataframe if not already loaded

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

# Loop through the rows in the dataframe
for _, row in df.iterrows():
    actual_output = row['OUTPUT']
    wrapper_output = row['llama_3.2_OUTPUT']
    
    # ROUGE scores
    rouge_scores = scorer.score(actual_output, wrapper_output)
    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
    
    # BLEU score
    bleu_scores.append(bleu_score.sentence_bleu([actual_output.split()], wrapper_output.split()))
    
    # METEOR score (Tokenize both actual_output and wrapper_output)
    actual_tokens = nltk.word_tokenize(actual_output)  # Tokenize the actual output
    wrapper_tokens = nltk.word_tokenize(wrapper_output)  # Tokenize the wrapper output
    meteor_scores.append(meteor_score.single_meteor_score(actual_tokens, wrapper_tokens))
    
    # BERTScore
    P, R, F1 = score([wrapper_output], [actual_output], lang='en')
    bert_scores.append(F1.item())
    
    # Blanc Score
    bl = BlancHelp(device='cuda', inference_batch_size=128)
    blanc_score = bl.eval_once(actual_output, wrapper_output)
    blanc_scores.append(blanc_score)

# Calculate the average of each score across all rows
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)
average_bleu = sum(bleu_scores) / len(bleu_scores)
average_meteor = sum(meteor_scores) / len(meteor_scores)
average_bert = sum(bert_scores) / len(bert_scores)
average_blanc = sum(blanc_scores) / len(blanc_scores)

# Print the average scores
print(f"Average ROUGE-1: {average_rouge1:.4f}")
print(f"Average ROUGE-2: {average_rouge2:.4f}")
print(f"Average ROUGE-L: {average_rougeL:.4f}")
print(f"Average BLEU: {average_bleu:.4f}")
print(f"Average METEOR: {average_meteor:.4f}")
print(f"Average BERTScore: {average_bert:.4f}")
print(f"Average BLANC: {average_blanc:.4f}")

# Adding the scores to the dataframe (if needed)
df['ROUGE-1'] = rouge1_scores
df['ROUGE-2'] = rouge2_scores
df['ROUGE-L'] = rougeL_scores
df['BLEU'] = bleu_scores
df['METEOR'] = meteor_scores
df['BERTScore'] = bert_scores
df['BLANC'] = blanc_scores

# Save the updated dataframe to a new CSV file if needed
df.to_csv('evaluated_output_with_avg_llama2_7B_chat.csv', index=False)  # Save to a new CSV file

