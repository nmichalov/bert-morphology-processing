import scipy.stats as stats

def calculate_p_value(r, n):
    """
    Calculate the p-value for a given correlation coefficient.

    Args:
        r (float): Correlation coefficient (-1 to 1).
        n (int): Number of data points.

    Returns:
        float: Two-tailed p-value.
    """
    if n <= 2:
        raise ValueError("Sample size must be greater than 2 to calculate p-value.")

    # Calculate the t-statistic
    t_statistic = r * ((n - 2) ** 0.5) / ((1 - r**2) ** 0.5)

    # Degrees of freedom
    df = n - 2

    # Calculate the two-tailed p-value
    p_value = 2 * stats.t.sf(abs(t_statistic), df)
    return p_value

# Example usage
r = -0.4741  # Correlation coefficient
n = 42   # Sample size

p_value = calculate_p_value(r, n)
print(f"Correlation coefficient: {r}")
print(f"p-value: {p_value}")
