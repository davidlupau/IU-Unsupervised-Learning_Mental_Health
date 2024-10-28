import pandas as pd
import numpy as np
from prepare_dataset import drop_columns, correct_age, replace_nulls, transform_categorical_to_numerical, filter_rows, one_hot_encode, merge_normalise_current_previous_employer
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df = filter_rows(df)
df = drop_columns(df)
df = correct_age(df)

# Perform one-hot encoding on the 'self_employed' column
df = one_hot_encode(df, 'self_employed')

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
    'Yes, it has': 1,
	'Yes, I think it would': 2,
	'Maybe': 3,
	'No, I don\'t think it would': 4,
	'No, it has not': 5,
}
df = transform_categorical_to_numerical(df, 'career_impact', mapping)

# Transforming categorical values to numerical values for column negative_viewed_coworkers and replacing null values
mapping = {
    'Yes, they do': 1,
	'Yes, I think they would': 2,
	'Maybe': 3,
	'No, I don\'t think they would': 4,
	'No, they would not': 5,
}
df = transform_categorical_to_numerical(df, 'negative_viewed_coworkers', mapping)
df = replace_nulls(df, 'negative_viewed_coworkers', distribution={1: 0.07, 2: 0.38, 3: 0.41, 4: 0.24, 5: 0.03})

# Transforming categorical values to numerical values and replacing null values for 'share_family_friends'
mapping = {
    'Not open at all': 1,
	'Somewhat not open': 2,
	'Neutral': 3,
	'Somewhat open': 4,
	'Very open': 5,
	'Not applicable to me (I do not have a mental illness)': np.nan,
}
df = transform_categorical_to_numerical(df, 'share_family_friends', mapping)
df = replace_nulls(df, 'share_family_friends', distribution={1: 0.06, 2: 0.16, 3: 0.11, 4: 0.48, 5: 0.19})

# Transforming categorical values to numerical values and replacing null values for 'bad_response_previous_work'
mapping = {
    'No': 1,
	'Maybe/Not sure': 2,
	'Yes, I observed': 3,
	'Yes, I experienced': 4,
	'N/A': np.nan,
}
df = transform_categorical_to_numerical(df, 'bad_response_previous_work', mapping)
df = replace_nulls(df, 'bad_response_previous_work', distribution={1: 0.42, 2: 0.26, 3: 0.20, 4: 0.12})

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
    'Never': 1,
	'Rarely': 2,
	'Sometimes': 3,
	'Often': 4,
	'Not applicable to me': np.nan,
}
df = transform_categorical_to_numerical(df, 'treated_mh_interfere_work', mapping)
df = replace_nulls(df, 'treated_mh_interfere_work', distribution={1: 0.14, 2: 0.37, 3: 0.42, 4: 0.07})
df = transform_categorical_to_numerical(df, 'not_treated_mh_interfere_work', mapping)
df = replace_nulls(df, 'not_treated_mh_interfere_work', distribution={1: 0.01, 2: 0.05, 3: 0.38, 4: 0.56})

# Prepare the 'gender' column for one-hot encoding
df = replace_nulls(df, 'gender', distribution={'Male': 1})
df = one_hot_encode(df, 'gender')

# Transforming categorical values to numerical values and replacing null values for 'remote_worker'
mapping = {
    'Never': 1,
	'Sometimes': 2,
	'Always': 3,
}
df = transform_categorical_to_numerical(df, 'remote_worker', mapping)

#df.to_csv('mental_health_tech_cleaned.csv')
#print(df.isnull().sum())