import pandas as pd
import numpy as np

# Function to drop columns identified as non-adding value to the analysis
def drop_columns(df):
    """Drops predefined columns that are not adding value to the analysis."""
    columns_to_drop = [
        'tech_IT_primary_role', 'medical_coverage_mh', 'know_help_resources', 'observations_less_likely_reveal', 'reveal_mh_problem_client',
        'negative_impact_revealed_client', 'reveal_mh_problem_coworkers', 'negative_impact_revealed_coworkers',
        'productivity_affected', 'percent_worktime_impacted', 'previous_employer', 'reason_ph_interview', 'reason_mh_interview', 'mh_diagnostic',
        'possible_mh_condition', 'mh_condition_diagnosed', 'us_state', 'work_us_state', 'work_position'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, axis=1)
    return df_cleaned

# Function to correct the incorrect age values
def correct_age(df):
    """Replaces ages below 15 and above 75 with the median age."""
    median_age = df['age'].median()  # Calculate the median
    df.loc[df['age'] < 15, 'age'] = median_age  # Replace ages below 15
    df.loc[df['age'] > 75, 'age'] = median_age  # Replace ages above 75
    return df

# Function to replace missing values based on a specified distribution
def replace_nulls(df, column, distribution=None):
    """Replace null values in a specified column based on a provided distribution.
    Parameters:
    df (DataFrame): The DataFrame to modify.
    column (str): The specific column to operate on.
    distribution (dict): A dictionary where keys are the values to replace with,
                         and values are the fraction of total nulls for each value.
    Returns:
    DataFrame: The modified DataFrame with null values replaced."""
    # Check for null values
    null_count = df[column].isnull().sum()

    if null_count > 0:
        # Create the replacement array based on the distribution
        replacement_values = []
        for value, fraction in distribution.items():
            replacement_count = int(null_count * fraction)
            replacement_values.extend([value] * replacement_count)

        # If the replacement values are fewer than nulls, fill the remaining with the last value
        if len(replacement_values) < null_count:
            remaining_count = null_count - len(replacement_values)
            replacement_values.extend([list(distribution.keys())[-1]] * remaining_count)

        # Shuffle the replacement values
        np.random.shuffle(replacement_values)

        # Replace nulls in the specified column
        df.loc[df[column].isnull(), column] = replacement_values[:null_count]

    return df

# Function to transform categorical values to numerical values based on a mapping
def transform_categorical_to_numerical(df, column, mapping):
    """Transform categorical values in a specified column to numerical values based on a provided mapping.
    Parameters:
    df (DataFrame): The DataFrame to modify.
    column (str): The specific column to operate on.
    mapping (dict): A dictionary where keys are the categorical values and values are the corresponding numerical values.
    Returns:
    DataFrame: The modified DataFrame with categorical values replaced by numerical values.
    """
    if column in df.columns:
        df[column] = df[column].map(mapping)

    return df
