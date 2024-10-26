import pandas as pd
import numpy as np
from prepare_dataset import replace_nulls, transform_categorical_to_numerical

def print_percentage_of_values(df, column_name):
    # Calculate the value counts and convert to percentage
    value_counts_percentage = df[column_name].value_counts(normalize=True) * 100

    # Print each value and its percentage
    for value, percentage in value_counts_percentage.items():
        print(f"{value}: {percentage:.2f}%")

# Load dataset into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

print(df['ask_mh_leave'].isnull().sum())

#df.to_csv('mental_health_tech_cleaned.csv')