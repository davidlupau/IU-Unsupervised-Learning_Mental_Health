import pandas as pd
import numpy as np
from functions import drop_columns, correct_age, replace_nulls, transform_categorical_to_numerical, filter_rows, one_hot_encode, merge_normalise_current_previous_employer

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df = filter_rows(df)
df = drop_columns(df)
df = correct_age(df)

# Preparing the dataset for analysis
# Perform one-hot encoding on the 'self_employed' column
df = one_hot_encode(df, 'self_employed')

# Create binary "flag" for respondents who skipped company questions
# First identify the columns for questions 2-16
company_related_columns = ['company_size', 'tech_organization', 'employer_mh_benefits', 'know_mh_benefits', 'employer_discussed_mh', 'employer_mh_resources', 'anonymous_mh_employer_resources', 'ask_mh_leave', 'discuss_mh_with_employer_negative', 'discuss_ph_with_employer_negative', 'discuss_mh_with_coworkers', 'discuss_mh_with_manager', 'employer_mh_as_serious_as_ph', 'observed_negative_consequences_coworkers']
# Create the "flag"
df['skipped_company_questions'] = df[company_related_columns].isnull().all(axis=1).astype(int)

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

# Prepare the 'tech_organization' column for one-hot encoding
df = replace_nulls(df, 'tech_organization', distribution={1: 0.78, 0: 0.22})
df = one_hot_encode(df, 'tech_organization')

# Transforming categorical values to numerical values and replacing null values for 'ask_mh_leave'
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



#df.to_csv('mental_health_tech_merged.csv', index=False)