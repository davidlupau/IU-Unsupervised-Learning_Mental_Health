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
company_related_columns = [
	'company_size', 'tech_organization', 'employer_mh_benefits', 'know_mh_benefits', 'employer_discussed_mh',
	'employer_mh_resources', 'anonymous_mh_employer_resources', 'ask_mh_leave', 'discuss_mh_with_employer_negative',
	'discuss_ph_with_employer_negative', 'discuss_mh_with_coworkers', 'discuss_mh_with_manager',
	'employer_mh_as_serious_as_ph', 'observed_negative_consequences_coworkers'
]
# Create the "flag"
df['skipped_company_questions'] = df[company_related_columns].isnull().all(axis=1).astype(int)

# Transforming categorical values to numerical values using midpoints and replacing null values for 'company_size'
mapping = {
    '1-5 employees': 0.003,       # midpoint = 3
    '6-25 employees': 0.015,      # midpoint = 15
    '26-100 employees': 0.063,    # midpoint = 63
    '100-500 employees': 0.3,     # midpoint = 300
    '500-1000 employees': 0.75,   # midpoint = 750
    'More than 1000 employees': 1 # setting as maximum
}
df = transform_categorical_to_numerical(df, 'company_size', mapping)
df = replace_nulls(df, 'company_size', distribution={0.003: 0.05, 0.015: 0.22, 0.063: 0.25, 0.3: 0.07, 0.75: 0.18, 1: 0.22})

# Prepare the 'tech_organization' column for one-hot encoding
df = replace_nulls(df, 'tech_organization', distribution={1: 0.78, 0: 0.22})
df = one_hot_encode(df, 'tech_organization')

# Transforming categorical values to numerical values and replacing null values for 'ask_mh_leave'
mapping = {
    'Very difficult': 0,
	'Somewhat difficult': 0.1,
	'Neither easy nor difficult': 0.5,
	'Somewhat easy': 0.9,
	'Very easy': 1,
	'I don’t know': np.nan,
}
df = transform_categorical_to_numerical(df, 'ask_mh_leave', mapping)
df = replace_nulls(df, 'ask_mh_leave', distribution={0: 0.1, 0.1: 0.17, 0.5: 0.16, 0.9: 0.25, 1: 0.19})

# Merging and normalising same questions about current and previous employer
merge_normalise_current_previous_employer(df)

# Drop colmuns with high number of null values after merge
columns_to_drop = [
	'know_mh_benefits', 'anonymous_mh_employer_resources'
]
df = df.drop(columns=columns_to_drop, axis=1)

# Replace null values in merged columns based on a specified distribution of each column
df = replace_nulls(df, 'employer_mh_benefits', distribution={1: 0.65, 0: 0.35})
df = replace_nulls(df, 'employer_discussed_mh', distribution={1: 0.21, 0: 0.79})
df = replace_nulls(df, 'employer_mh_resources', distribution={1: 0.34, 0: 0.66})
df = replace_nulls(df, 'discuss_ph_with_employer_negative', distribution={1: 0.13, 0.5: 0.19, 0: 0.68})
df = replace_nulls(df, 'discuss_mh_with_employer_negative', distribution={1: 0.29, 0.5: 0.37, 0: 0.34})
df = replace_nulls(df, 'discuss_mh_with_coworkers', distribution={1: 0.33, 0.5: 0.34, 0: 0.33})
df = replace_nulls(df, 'discuss_mh_with_manager', distribution={1: 0.42, 0.5: 0.28, 0: 0.30})
df = replace_nulls(df, 'employer_mh_as_serious_as_ph', distribution={1: 0.53, 0: 0.47})
df = replace_nulls(df, 'observed_negative_consequences_coworkers', distribution={1: 0.16, 0: 0.84})

# Transforming categorical values to numerical values for columns with responses yes, no, maybe
mapping = {
    'No': 0,
	'Maybe': 0.5,
	'Yes': 1,
}
df = transform_categorical_to_numerical(df, 'ph_interview', mapping)
df = transform_categorical_to_numerical(df, 'talk_mh_interview', mapping)
df = transform_categorical_to_numerical(df, 'past_mh_disorder', mapping)
df = transform_categorical_to_numerical(df, 'current_mh_disorder', mapping)

# Transforming categorical values to numerical values for column career_impact
mapping = {
    'Yes, it has': 0,
	'Yes, I think it would': 0.1,
	'Maybe': 0.5,
	'No, I don\'t think it would': 0.9,
	'No, it has not': 1,
}
df = transform_categorical_to_numerical(df, 'career_impact', mapping)

# Transforming categorical values to numerical values for column negative_viewed_coworkers and replacing null values
mapping = {
    'Yes, they do': 0,
	'Yes, I think they would': 0.1,
	'Maybe': 0.5,
	'No, I don\'t think they would': 0.9,
	'No, they would not': 1,
}
df = transform_categorical_to_numerical(df, 'negative_viewed_coworkers', mapping)
df = replace_nulls(df, 'negative_viewed_coworkers', distribution={0: 0.07, 0.1: 0.38, 0.5: 0.41, 0.9: 0.24, 1: 0.03})

# Transforming categorical values to numerical values and replacing null values for 'share_family_friends'
mapping = {
    'Not open at all': 0,
	'Somewhat not open': 0.1,
	'Neutral': 0.5,
	'Somewhat open': 0.9,
	'Very open': 1,
	'Not applicable to me (I do not have a mental illness)': np.nan,
}
df = transform_categorical_to_numerical(df, 'share_family_friends', mapping)
df = replace_nulls(df, 'share_family_friends', distribution={0: 0.06, 0.1: 0.16, 0.5: 0.11, 0.9: 0.48, 1: 0.19})

# Transforming categorical values to numerical values and replacing null values for 'bad_response_previous_work'
mapping = {
    'No': 0,
	'Maybe/Not sure': 0.3,
	'Yes, I observed': 0.7,
	'Yes, I experienced': 1,
	'N/A': np.nan,
}
df = transform_categorical_to_numerical(df, 'bad_response_previous_work', mapping)
df = replace_nulls(df, 'bad_response_previous_work', distribution={0: 0.42, 0.3: 0.26, 0.7: 0.20, 1: 0.12})

# Transforming categorical values to numerical values for column family_history
mapping = {
    'No': 0,
	'I don\'t know': 0.5,
	'Yes': 1,
}
df = transform_categorical_to_numerical(df, 'family_history', mapping)

# Prepare the 'family_history' column for one-hot encoding
df = one_hot_encode(df, 'family_history')

# Transforming categorical values to numerical values for mh_diagnostic_medical_pro column
mapping = {
    'No': 0,
	'Yes': 1,
}
df = transform_categorical_to_numerical(df, 'mh_diagnostic_medical_pro', mapping)

# Transforming categorical values to numerical values and replacing null values for treated_mh_interfere_work and not_treated_mh_interfere_work columns
mapping = {
    'Never': 0,
	'Rarely': 0.3,
	'Sometimes': 0.7,
	'Often': 1,
	'Not applicable to me': np.nan,
}
df = transform_categorical_to_numerical(df, 'treated_mh_interfere_work', mapping)
df = replace_nulls(df, 'treated_mh_interfere_work', distribution={0: 0.14, 0.3: 0.37, 0.7: 0.42, 1: 0.07})
df = transform_categorical_to_numerical(df, 'not_treated_mh_interfere_work', mapping)
df = replace_nulls(df, 'not_treated_mh_interfere_work', distribution={0: 0.01, 0.3: 0.05, 0.7: 0.38, 1: 0.56})

# Grouping age into categories and mapping to 0-1 scale
# Define bins and labels
bins = [16, 25, 35, 45, 55, 75]  # covers full range
labels = ['Under 25', '25-34', '35-44', '45-54', '55+']

# Create age groups
df['age'] = pd.cut(df['age'], bins=bins, labels=labels)

# Map to 0-1 scale
mapping = {
    'Under 25': 0,
    '25-34': 0.25,
    '35-44': 0.5,
    '45-54': 0.75,
    '55+': 1,
}
df = transform_categorical_to_numerical(df, 'age', mapping)

# Prepare the 'gender' column for one-hot encoding
df = replace_nulls(df, 'gender', distribution={'Male': 1})
df = one_hot_encode(df, 'gender')

# Transforming categorical values to numerical values and replacing null values for 'remote_worker'
mapping = {
    'Never': 0,
	'Sometimes': 0.5,
	'Always': 1,
}
df = transform_categorical_to_numerical(df, 'remote_worker', mapping)

df.to_csv('mental_health_tech_cleaned.csv', index=False)