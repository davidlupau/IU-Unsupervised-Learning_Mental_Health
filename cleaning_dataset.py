# Function to drop columns identified as non-adding value to the analysis
def drop_columns(df):
    columns_to_drop = [
        'tech_IT_primary_role', 'medical_coverage_mh', 'know_help_resources', 'reveal_mh_problem_client',
        'negative_impact_revealed_client', 'reveal_mh_problem_coworkers', 'negative_impact_revealed_coworkers',
        'productivity_affected', 'percent_worktime_impacted', 'previous_employer', 'reason_ph_interview', 'reason_mh_interview', 'mh_diagnostic',
        'possible_mh_condition', 'mh_condition_diagnosed', 'us_state', 'work_us_state', 'work_position'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, axis=1)
    return df_cleaned

# Function to correct the incorrect age values
def correct_age(df):
    median_age = df['age'].median()  # Calculate the median
    df.loc[df['age'] < 15, 'age'] = median_age  # Replace ages below 15
    df.loc[df['age'] > 75, 'age'] = median_age  # Replace ages above 75
    return df

