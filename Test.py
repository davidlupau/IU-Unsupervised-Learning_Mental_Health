import pandas as pd
import numpy as np
from cleaning_dataset import replace_nulls, transform_categorical_to_numerical

# Load your dataset
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

mapping = {
    'Yes': 1,
    'No': 0,
    'I don\'t know': 0.5,
}

df = transform_categorical_to_numerical(df, 'employer_discussed_mh', mapping)

df = replace_nulls(df, 'tech_organization', distribution={1: 0.78, 0: 0.22})

df.to_csv('test.csv')