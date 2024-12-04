from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Function to calculate cosine similarity
# def cosine_similarity(vec1, vec2):
#     vec1 = vec1.detach().numpy()
#     vec2 = vec2.detach().numpy()
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# try dot product
def dot_product(vec1, vec2):
    vec1 = vec1.detach().numpy()
    vec2 = vec2.detach().numpy()
    return np.dot(vec1, vec2)

# Function to get BERT embeddings
def get_bert_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings for the word (pooling)
    # return outputs.last_hidden_state.mean(dim=1).squeeze()
    # return outputs.hidden_states[-2][0][1][:]
    return outputs.last_hidden_state[-2][0][1][:]
    # return torch.mean(token_vecs, dim=0)

# Load the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states = True)

# Input and output file paths
input_file = "single_token_stems.txt"  # Replace with the actual input file path
output_file = "single_token_stem_last_hidden_state_distances.txt"  # Replace with the desired output file path

# Process the input file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Parse the line
        score, word1, word2 = line.strip().split()
        score = float(score)

        # Get embeddings for the two words
        embedding1 = get_bert_embedding(word1, tokenizer, model)
        embedding2 = get_bert_embedding(word2, tokenizer, model)

        # Calculate cosine similarity
        # similarity = cosine_similarity(embedding1, embedding2)
        similarity = dot_product(embedding1, embedding2)

        # Write the result to the output file
        outfile.write(f"{word1} {word2} {score} {similarity:.4f}\n")
