from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine

# Function to calculate cosine similarity
# def cosine_similarity(vec1, vec2):
#     # return 1 - cosine(vec1, vec2)
#     return cosine(vec1, vec2)

def dot_product(vec1, vec2):
    vec1 = vec1.detach().numpy()
    vec2 = vec2.detach().numpy()
    return np.dot(vec1, vec2)

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states = True)

# Load BERT vocabulary
vocab = list(tokenizer.vocab.keys())

# Function to generate BERT embedding for a word
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states = True)
    # Take the mean of the last hidden state for simplicity
    # return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return outputs.hidden_states[-2][0][1][:]
    # token_vecs = outputs.hidden_states[-2][0]
    # return torch.mean(token_vecs, dim=0)

# Pre-generate embeddings for all vocabulary words
vocab_embeddings = {}
for vocab_word in vocab:
    vocab_embeddings[vocab_word] = get_embedding(vocab_word)

# Process input file and compute neighbors and cosine similarity
def process_file(input_file, output_file):
    results = []
    with open(input_file, "r") as infile:
        for line in infile:
            print(line)
            number, word, stem = line.strip().split()
            word_embedding = get_embedding(word)
            
            # Calculate cosine similarities to all vocabulary words
            similarities = []
            for vocab_word, vocab_embedding in vocab_embeddings.items():
                similarity = dot_product(word_embedding, vocab_embedding)
                similarities.append((vocab_word, similarity))
            
            # Sort and find the 10 nearest neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:10]
            
            # Calculate the average cosine similarity
            avg_similarity = np.mean([sim[1] for sim in top_neighbors])
            results.append(f"{word}\t{number}\t{avg_similarity:.4f}")

    # Write to output file
    with open(output_file, "w") as outfile:
        outfile.write("\n".join(results))

# Define input and output file names
input_filename = "single_token_stems.txt"  # Replace with your input file
output_filename = "nearest_ten_neighbors_single_token_stems.txt"

# Run the processing
process_file(input_filename, output_filename)

# Notify the user of the output location
output_filename
