import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('MMA_Fighter_Compare\\data\\all_fights_final.csv')  # Replace with your actual file path

# Count missing values in each column
missing_values = df.isnull().sum()

# Calculate percentage of missing values
missing_percentage = 100 * df.isnull().sum() / len(df)

# Combine count and percentage into a single dataframe
missing_table = pd.concat([missing_values, missing_percentage], axis=1, keys=['Missing Values', 'Percentage Missing'])

# Sort the table by the number of missing values in descending order
missing_table = missing_table.sort_values('Missing Values', ascending=False)

# Only show columns with missing values
missing_table = missing_table[missing_table['Missing Values'] > 0]

# Print the result
print(missing_table)

# Optionally, you can save this to a CSV file
# missing_table.to_csv('missing_values_report.csv')

# Print total number of missing values
print(f"\nTotal number of missing values: {df.isnull().sum().sum()}")

# Print number of columns with missing values
print(f"Number of columns with missing values: {len(missing_table)}")