from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define file names
input_file = "novel_words_and_nonce.txt"  
output_file = "multi_token_stems.txt"  

# Open the output file for writing
with open(output_file, "w") as out_file:
    # Read the input file line by line
    with open(input_file, "r") as in_file:
        for line in in_file:
            # Split the line into components
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Skip malformed lines
            
            # Get the last word on the line
            last_word = parts[-1]
            
            # Tokenize the word and check if it is a single token
            tokens = tokenizer.tokenize(last_word)
            if len(tokens) > 1:
                # Write the line to the output file
                out_file.write(line)

print(f"Filtered lines have been written to {output_file}")
