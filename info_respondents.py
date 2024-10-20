import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

percentages = df['self_employed'].value_counts(normalize=True) * 100
print(f"Percentage of self-employed: {percentages[1]:.2f}%")
print(f"Percentage of employees: {percentages[0]:.2f}%")