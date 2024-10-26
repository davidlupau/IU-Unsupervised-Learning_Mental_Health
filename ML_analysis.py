import pandas as pd
import numpy as np
from prepare_dataset import drop_columns, correct_age, replace_nulls, transform_categorical_to_numerical, filter_rows, one_hot_encode, merge_normalise_current_previous_employer
from Test import print_percentage_of_values

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df = filter_rows(df)
df = drop_columns(df)
df = correct_age(df)

# Preparing the dataset for analysis
# Transforming categorical values to numerical values and replacing null values for 'company_size'
mapping = {
    '1-5 employees': 1,
	'6-25 employees': 2,
	'26-100 employees': 3,
	'100-500 employees': 4,
	'500-1000 employees': 5,
	'More than 1000 employees': 6,
}
df = transform_categorical_to_numerical(df, 'company_size', mapping)
df = replace_nulls(df, 'company_size', distribution={1: 0.05, 2: 0.22, 3: 0.25, 4: 0.07, 5: 0.18, 6: 0.22})

# Perform one-hot encoding on the 'self_employed' column
df = one_hot_encode(df, 'self_employed')

# Transforming categorical values to numerical values and replacing null values for 'tech_company'
mapping = {
    'Very difficult': 1,
	'Somewhat difficult': 2,
	'Neither easy nor difficult': 3,
	'Somewhat easy': 4,
	'Very easy': 5,
	'I don’t know': np.nan,
}
df = transform_categorical_to_numerical(df, 'ask_mh_leave', mapping)
df = replace_nulls(df, 'ask_mh_leave', distribution={1: 0.1, 2: 0.17, 3: 0.16, 4: 0.25, 5: 0.19})

# Merging and normalising same questions about current and previous employer
merge_normalise_current_previous_employer(df)

# Replace null values in merged columns based on a specified distribution of each column
df = replace_nulls(df, 'employer_mh_benefits', distribution={1: 0.65, 0: 0.35})
df = replace_nulls(df, 'know_mh_benefits', distribution={1: 0.64, 0: 0.36})
df = replace_nulls(df, 'employer_discussed_mh', distribution={1: 0.21, 0: 0.79})
df = replace_nulls(df, 'employer_mh_resources', distribution={1: 0.34, 0: 0.66})
df = replace_nulls(df, 'anonymous_mh_employer_resources', distribution={1: 0.75, 0: 0.25})
df = replace_nulls(df, 'discuss_mh_with_employer_negative', distribution={1: 0.29, 0.5: 0.37, 0: 0.34})
df = replace_nulls(df, 'discuss_ph_with_employer_negative', distribution={1: 0.13, 0.5: 0.19, 0: 0.68})
df = replace_nulls(df, 'discuss_mh_with_coworkers', distribution={1: 0.33, 0.5: 0.34, 0: 0.33})
df = replace_nulls(df, 'discuss_mh_with_manager', distribution={1: 0.42, 0.5: 0.28, 0: 0.30})
df = replace_nulls(df, 'employer_mh_as_serious_as_ph', distribution={1: 0.53, 0: 0.47})
df = replace_nulls(df, 'observed_negative_consequences_coworkers', distribution={1: 0.16, 0: 0.84})

# Prepare the 'tech_organization' column for one-hot encoding
df = replace_nulls(df, 'tech_organization', distribution={1: 0.78, 0: 0.22})
df = one_hot_encode(df, 'tech_organization')


# Prepare the 'gender' column for one-hot encoding
df = replace_nulls(df, 'gender', distribution={'Male': 1})
df = one_hot_encode(df, 'gender')


