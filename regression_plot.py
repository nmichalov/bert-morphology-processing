import matplotlib.pyplot as plt
from scipy.stats import linregress, t
import numpy as np

def process_and_analyze_with_ci(file_name, confidence=0.95):
    """
    Process the file, plot a scatter plot, and perform regression analysis with confidence intervals.
    
    Args:
        file_name (str): The name of the input file.
        confidence (float): The desired confidence level (default is 95%).
    """
    x_values = []
    y_values = []

    # Read the file
    with open(file_name, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3: # use for nearest neighbor
                try:
                    x_values.append(float(parts[1]))  
                    y_values.append(float(parts[2]))  
                except ValueError:
                    print(f"Skipping line due to invalid number format: {line}")
            elif len(parts) == 4: # use for distance
                try:
                    x_values.append(float(parts[2]))  
                    y_values.append(float(parts[3]))  
                except ValueError:
                    print(f"Skipping line due to invalid number format: {line}")

    # Convert to numpy arrays for analysis
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Perform regression analysis
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

    # Degrees of freedom
    n = len(x_values)
    df = n - 2

    # Critical t-value for the given confidence level
    t_critical = t.ppf((1 + confidence) / 2, df)

    # Confidence intervals for slope and intercept
    slope_ci = (slope - t_critical * std_err, slope + t_critical * std_err)

    intercept_std_err = std_err * (np.sum(x_values**2) / (n * np.var(x_values)))
    intercept_ci = (intercept - t_critical * intercept_std_err,
                    intercept + t_critical * intercept_std_err)

    # Print regression analysis results
    print("Regression Analysis:")
    print(f"Slope: {slope} (95% CI: {slope_ci})")
    print(f"Intercept: {intercept} (95% CI: {intercept_ci})")
    print(f"R-squared: {r_value**2}")
    print(f"P-value: {p_value}")
    print(f"Standard Error: {std_err}")
    
    correlation_coefficient = np.corrcoef(x_values, y_values)[0, 1]
    print(f"Correlation Coefficient: {correlation_coefficient:.4f}")

    # Create the scatter plot
    plt.scatter(x_values, y_values, label="Data points", alpha=0.7)

    # Set axis limits 
    plt.xlim(1, 5)  # X-axis range from 1 to 5

    # proximity and neighbor (both dot product)
    plt.ylim(0, 500)  # Y-axis range from -1 to 1

    # Add the regression line
    regression_line = slope * x_values + intercept
    plt.plot(x_values, regression_line, color="red", label="Regression line")

    # Customize the plot
    plt.title("Scatter Plot with Regression Line")
    plt.xlabel("Word Plausibility (X)")
    plt.ylabel("Average Distance to Ten Nearest Neighbors (Y)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Usage
file_name = "nearest_ten_neighbors_single_token_stems.txt"  # Replace with your file name
process_and_analyze_with_ci(file_name)


# able_single_token_stems.txt
# Slope: -53.54812999657995 (95% CI: (np.float64(-85.32451904170932), np.float64(-21.771740951450578)))
# Intercept: 406.82770178862825 (95% CI: (np.float64(-2370.625356047336), np.float64(3184.280759624592)))
# R-squared: 0.22480081202263244
# P-value: 0.0015139194483789794
# Standard Error: 15.722515447735963

# able_nearest_neighbor_single_token_stems.txt
# Slope: -36.337867344332984 (95% CI: (np.float64(-57.99641491235262), np.float64(-14.67931977631335)))
# Intercept: 372.9672575186761 (95% CI: (np.float64(-1520.1238386926589), np.float64(2266.058353730011)))
# R-squared: 0.22327197871528404
# P-value: 0.0015792989812706164
# Standard Error: 10.716348173799446