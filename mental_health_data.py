import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Get a summary of the dataset
print(df.info())
# Display the first few rows
print(df.head())