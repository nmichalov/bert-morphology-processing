from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Function to calculate entropy of a vector
def calculate_entropy(vector):
    # Normalize vector to create a probability-like distribution
    probabilities = torch.abs(vector) / torch.sum(torch.abs(vector))
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))  # Add epsilon to avoid log(0)
    return entropy.item()

# Function to get BERT embeddings
def get_bert_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings for the word (pooling)
    # return outputs.last_hidden_state.mean(dim=1).squeeze()
    token_vecs = outputs.hidden_states[-2][0]
    return torch.mean(token_vecs, dim=0)

# Load the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states = True)

# Input and output file paths
input_file = "novel_words_and_nonce.txt"  # Replace with the actual input file path
output_file = "output_word_entropy.txt"  # Replace with the desired output file path

# Process the input file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Parse the line
        score, word1, word2 = line.strip().split()
        score = float(score)

        # Get embedding for the first word
        embedding = get_bert_embedding(word1, tokenizer, model)

        # Calculate entropy of the embedding
        entropy = calculate_entropy(embedding)

        # Write the result to the output file
        outfile.write(f"{word1} {word2} {score} {entropy:.4f}\n")
