import pandas as pd
import numpy as np

# Function to drop columns identified as non-adding value to the analysis
def drop_columns(df):
    """Drops predefined columns that are not adding value to the analysis."""
    columns_to_drop = [
        'tech_IT_primary_role', 'medical_coverage_mh', 'know_help_resources', 'observations_less_likely_reveal', 'reveal_mh_problem_client',
        'negative_impact_revealed_client', 'reveal_mh_problem_coworkers', 'negative_impact_revealed_coworkers',
        'productivity_affected', 'percent_worktime_impacted', 'previous_employer', 'reason_ph_interview', 'reason_mh_interview', 'mh_diagnostic',
        'possible_mh_condition', 'mh_condition_diagnosed', 'country', 'work_country', 'us_state', 'work_us_state', 'work_position'
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

# Function to delete rows where 'previous_employer' is 0 and 'employer_mh_benefits' is null
def filter_rows(df):
    """Filter out rows where 'previous_employer' is 0 and 'employer_mh_benefits' is null.
    Parameters:
    df (DataFrame): The DataFrame to filter.
    Returns:
    DataFrame: The filtered DataFrame.
    """
    df = df[(df['previous_employer'] != 0) | (df['employer_mh_benefits'].notna())]

    return df

# Function to perform one-hot encoding on a column
def one_hot_encode(df, column):
    """Perform one-hot encoding on a specified column.
    Parameters:
    df (DataFrame): The DataFrame to modify.
    column (str): The specific column to encode.
    Returns:
    DataFrame: The modified DataFrame with one-hot encoding applied.
    """
    df[column] = df[column].astype(str)
    # Perform one-hot encoding only on the specified column
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
    # Convert True/False to 0/1 if necessary
    dummies = dummies.astype(int)
    # Drop the original column from df if you do not need it anymore
    df.drop(column, axis=1, inplace=True)
    # Join the one-hot encoded DataFrame back to the original DataFrame
    df = pd.concat([df, dummies], axis=1)
    return df

def merge_normalise_current_previous_employer(df):
    '''Merge and normalise responses to the same questions about current and previous employer
    Parameters:
    df (DataFrame): The DataFrame to modify.
    Returns:
    dataframe: The modified DataFrame with merged and normalised responses.'''
    # Normalise responses
    def normalize_responses(value):
        response_mapping = {
            "not eligible for coverage / n/a": np.nan,
            "i don't know": np.nan,
            "yes": 1,
            "no": 0,
            "maybe": 0.5,
            "n/a (not currently aware)": np.nan,
            "i am not sure": np.nan,
            "yes, they all did": 1,
            "no, none did": 0,
            "some did": 1,
            "n/a": np.nan,
            "i was aware of some": 1,
            "yes, i was aware of all of them": 1,
            "no, i only became aware later": 1,
            "none did": 0,
            "yes, always": 1,
            "sometimes": 1,
            "none of them": 0,
            "yes, all of them": 1,
            "some of them": 1,
            "some of my previous employers": 1,
            "no, at none of my previous employers": 0,
            "yes, at all of my previous employers": 1,
        }
        if pd.isna(value) or not isinstance(value, str):
            return np.nan
        return response_mapping.get(value.lower(), 'unknown')

    # Merge columns
    def merge_columns(df, current_col, previous_col):
        df[current_col] = df[current_col].apply(normalize_responses)
        df[previous_col] = df[previous_col].apply(normalize_responses)

        # Merge columns
        df[current_col] = df[current_col].fillna(df[previous_col])
        return df

    # Listing columns to merge
    columns_to_merge = [
        ('employer_mh_benefits', 'previous_employer_mh_benefits'),
        ('know_mh_benefits', 'knew_options_mh_benefits'),
        ('employer_discussed_mh', 'previous_employer_discussed_mh'),
        ('employer_mh_resources', 'previous_employer_mh_resources'),
        ('anonymous_mh_employer_resources', 'anonymous_mh_previous_employer_resources'),
        ('discuss_mh_with_employer_negative', 'previous_employer_mh_negative_consequences'),
        ('discuss_ph_with_employer_negative', 'previous_employer_ph_negative_consequences'),
        ('discuss_mh_with_coworkers', 'discuss_mh_with_previous_coworkers'),
        ('discuss_mh_with_manager', 'discuss_mh_with_previous_manager'),
        ('employer_mh_as_serious_as_ph', 'previous_employer_mh_as_serious_as_ph'),
        ('observed_negative_consequences_coworkers', 'observed_negative_consequences_previous_coworkers')
    ]

    for current, previous in columns_to_merge:
        df = merge_columns(df, current, previous)

    # Drop the previous employer columns as they are no longer needed
    df.drop(columns=[col[1] for col in columns_to_merge], inplace=True)

    return df
