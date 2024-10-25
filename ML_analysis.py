import pandas as pd
from cleaning_dataset import drop_columns, correct_age, replace_nulls

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df2 = drop_columns(df)
df3= correct_age(df2)

# Convert categorical values to numerical values when applicable and replace null values based on a specified distribution
df4 = replace_nulls(df3, 'tech_organization', distribution={1: 0.78, 0: 0.22})
