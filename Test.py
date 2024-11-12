import pandas as pd


# Load clean dataset
df = pd.read_csv('mental_health_tech_cleaned.csv')

# Drop columns with high correlation
columns_to_drop = [
        'skipped_company_questions', 'self_employed_1', 'tech_organization_0.0',
        'gender_Non-binary/Other', 'gender_Female', 'family_history_0.5', 'family_history_0.0'
    ]
df = df.drop(columns=columns_to_drop, axis=1)

# Create mental health risk score
df['mental_health_score'] = (df['mh_treatment'] +
                           df['past_mh_disorder'] +
                           df['current_mh_disorder'] +
                           df['mh_diagnostic_medical_pro']) / 4  # Divide by 4 to keep 0-1 scale

# Remove original features after creating the new one
columns_to_drop = ['mh_treatment', 'past_mh_disorder',
                  'current_mh_disorder', 'mh_diagnostic_medical_pro']
df = df.drop(columns=columns_to_drop)

# List of features with variance >= 0.147
selected_features = [
    'family_history_1.0',
    'employer_mh_benefits',
    'employer_mh_as_serious_as_ph',
    'gender_Male',
    'discuss_mh_with_manager',
    'tech_organization_1.0',
    'employer_mh_resources',
    'company_size',
    'discuss_mh_with_employer_negative',
    'mental_health_score',
    'discuss_mh_with_coworkers',
    'ask_mh_leave',
    'employer_discussed_mh',
    'self_employed_0'
]

# Create new dataframe with only selected features
df = df[selected_features]

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Run the analysis
