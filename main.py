import pandas as pd
import numpy as np

# Define the number of rows and columns
num_rows = 10000  # Increase the number of rows for larger data
num_columns = 10  # Number of channels

# Generate random data
data = np.random.rand(num_rows, num_columns)

# Create a DataFrame with columns named 'channel1', 'channel2', ..., 'channel10'
columns = [f"channel{i+1}" for i in range(num_columns)]
df = pd.DataFrame(data, columns=columns)

# Save the generated data to a CSV file
df.to_csv("data/simulated_neural_data.csv", index=False)

# Preview the first few rows
print(df.head())
