import pandas as pd
from cleaning_dataset import drop_columns, correct_age, clean_gender

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df_column_cleaned = drop_columns(df)
df_age_cleaned = correct_age(df_column_cleaned)


#percentages = df['self_employed'].value_counts(normalize=True) * 100
#print(f"Percentage of self-employed: {percentages[1]:.2f}%")
#print(f"Percentage of employees: {percentages[0]:.2f}%")