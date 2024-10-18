import pandas as pd

# Set display option to show all columns
pd.set_option('display.max_columns', None)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Check for missing values
print(df.isnull().sum())
df.isnull().sum().to_csv('missing_values_summary.csv')

# Optionally, reset the display option back to default after checking
pd.reset_option('display.max_columns')