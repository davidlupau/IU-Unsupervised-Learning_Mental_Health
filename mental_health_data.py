import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Exploring the dataset
# Get a summary of the dataset
print(df.info())
print(df.isnull().sum())

# Get descriptive statistics for numerical columns
print(df.describe())

# Calculate
missing_percentage = df.isnull().mean() * 100
# Save to a text file
missing_percentage.to_csv('missing_percentage.csv')


