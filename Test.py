import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Calculate the percentage of each response in the 'remote_worker' column
remote_worker_percentages = df['remote_worker'].value_counts(normalize=True) * 100

# Display the percentages
print(remote_worker_percentages)