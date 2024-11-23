import matplotlib.pyplot as plt
import numpy as np

# Define the input file
# input_file = "output_cosine_distances.txt"
input_file = "output_word_entropy.txt"

# Initialize lists to store the x and y values
x_values = []
y_values = []

# Read the file and extract the numbers
with open(input_file, "r") as infile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) >= 4:  # Ensure the line has enough parts
            try:
                x_values.append(float(parts[2]))  # Third element for x-axis
                y_values.append(float(parts[3]))  # Fourth element for y-axis
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")

# Ensure there are valid data points
if len(x_values) > 0 and len(y_values) > 0:
    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(x_values, y_values)[0, 1]
    print(f"Correlation Coefficient: {correlation_coefficient:.4f}")

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7)

    # Add labels and title
    plt.xlabel("X-axis (Novel Word Plausibility)")
    # plt.ylabel("Y-axis (Cosine Distance Between Nonce and Stem)")
    plt.ylabel("Y-axis (Vector Entropy)")
    plt.title("Scatter Plot of Numbers from File")

    # Show grid and the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid data points found in the file.")
