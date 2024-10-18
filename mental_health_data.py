import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Exploring the dataset
# Get a summary of the dataset
# print(df.info())
print(df.isnull().sum())

# Get descriptive statistics for numerical columns in a csv file
# describe_dataset = df.describe(include='all')
# describe_dataset.to_csv('describe_dataset.csv')

# Calculate missing values in each column
# missing_percentage = df.isnull().mean() * 100
# Save to a text file
# missing_percentage.to_csv('missing_percentage.csv')


