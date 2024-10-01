import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Get descriptive statistics for both numerical and non-numerical columns
describe_dataset = df.describe(include='all')
describe_dataset.to_csv('describe_dataset.csv')