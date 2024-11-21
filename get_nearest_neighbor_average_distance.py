from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to get BERT embedding
def get_bert_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to get a single embedding vector for the word
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# Function to find 10 nearest neighbors and compute the average cosine distance
def compute_average_distance(target_embedding, all_embeddings):
    similarities = [
        compute_cosine_similarity(target_embedding, emb) for emb in all_embeddings
    ]
    # Sort by similarity and pick the 10 closest
    top_10_similarities = sorted(similarities, reverse=True)[1:11]  # Exclude the word itself
    # Compute the average cosine distance (1 - similarity)
    average_distance = np.mean([1 - sim for sim in top_10_similarities])
    return average_distance

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Input and output file paths
input_file = "novel_words_and_nonce.txt"  # Replace with actual input file path
output_file = "output_neighbors_distances.txt"  # Replace with desired output file path

# Process the input file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # Build a vocabulary of embeddings for all words in the file
    all_words = []
    all_embeddings = []
    for line in infile:
        _, word1, word2 = line.strip().split()
        all_words.extend([word1, word2])  # Collect both words
    all_embeddings = [
        get_bert_embedding(word, tokenizer, model) for word in all_words
    ]

    # Reset the file pointer to process each line again
    infile.seek(0)
    for line in infile:
        score, word1, word2 = line.strip().split()
        score = float(score)

        # Get the embedding for the first word
        target_embedding = get_bert_embedding(word1, tokenizer, model)

        # Compute the average cosine distance to its 10 nearest neighbors
        avg_distance = compute_average_distance(target_embedding, all_embeddings)

        # Write the result to the output file
        outfile.write(f"{word1} {word2} {score} {avg_distance:.4f}\n")
