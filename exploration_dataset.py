import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Get a summary of the dataset
print(df.info())

# Get descriptive statistics for numerical columns in a csv file
describe_dataset = df.describe(include='all')
# Save to a csv file
describe_dataset.to_csv('exploration_describe_dataset.csv')

# Calculate missing values in each column
missing_percentage = df.isnull().mean() * 100
# Save to a csv file
missing_percentage.to_csv('exploration_missing_percentage.csv')


