# Define input and output file names
input_file = "output_cosine_distances.txt"
output_file = "output_cosine_distances_sorted.txt"

# Read the input file
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Split lines into components and sort by the last element in descending order
sorted_lines = sorted(lines, key=lambda line: float(line.strip().split()[-1]), reverse=True)

# Write the sorted lines to the output file
with open(output_file, "w") as outfile:
    outfile.writelines(sorted_lines)

print(f"Sorted lines have been written to {output_file}")
