import matplotlib.pyplot as plt
import numpy as np

# Define the input file

# PROXIMITY FILES
# input_file = "output_cosine_distances.txt" # 0.0761

# input_file = "data_files/able_distance_scores.txt"  # - 0.2955
# input_file = "er_distance_scores.txt"    # - 0.0817
# input_file = "ist_distance_score.txt"    #   0.092
# input_file = "less_distance_scores.txt"  # - 0.1707

# input_file = "un_distance_scores.txt"    # - 0.0819
# input_file = "re_distance_scores.txt"    #   0.2708

# input_file = "single_token_stems_distances.txt"
# input_file = "able_single_token_distances.txt"
input_file = "single_token_stem_token_distances.txt"

# input_file = "multi_token_stems_distances.txt"

# input_file = "able_single_token_stems.txt" # -0.4741
# input_file = "less_single_token_stems.txt" # 0.1413
# input_file = "re_single_token_stems.txt"

# # ENTROPY FILES
# input_file = "output_word_entropy.txt"
# input_file = "able_entropy.txt"

## Neighbor Files
# input_file = "output_neighbor_data.txt"
# input_file = "output_single_nearest_neighbor_data.txt"
# input_file = "able_single_nearest_neighbor.txt"
# input_file = "able_nearest_neighbor_single_token_stems.txt"
# input_file = "nearest_neighbor_single_token_stems.txt"


# Initialize lists to store the x and y values
x_values = []
y_values = []

# Read the file and extract the numbers
with open(input_file, "r") as infile:
    for line in infile: 
        parts = line.strip().split()
        if len(parts) == 4:  # Ensure the line has enough parts               #  len 4 for entrop and distance
            try:
                x_values.append(float(parts[2]))  # Third element for x-axis
                y_values.append(float(parts[3]))  # Fourth element for y-axis
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
        elif len(parts) == 3:      # len 3 for neighbors
            try:
                x_values.append(float(parts[1]))
                y_values.append(float(parts[2]))
            except ValueError:
                print(f"Skipping line due to conversion error: {line}")

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
    # plt.ylabel("Y-axis (Cosine Measure)")
    # plt.ylabel("Y-axis (Vector Entropy)")
    plt.ylabel("Y-axis (distance between token and token in context)")
    plt.title("Scatter Plot of Numbers from File")

    # Set axis limits 
    plt.xlim(1, 5)  # X-axis range from 1 to 5

    # proximity and neighbor (both dot product)
    plt.ylim(0, 500)  # Y-axis range from -1 to 1

    # entropy
    # plt.ylim(0, 1)  # Y-axis range from 0 to 1

    # Show grid and the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid data points found in the file.")
