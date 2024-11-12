import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

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
            # Complete Negatives (0)
        "no": 0,
        "no, none did": 0,
        "none did": 0,
        "none of them": 0,
        "no, at none of my previous employers": 0,

        # Partial/Limited Cases (0.5)
        "sometimes": 0.5,
        "some did": 0.5,
        "some of them": 0.5,
        "some of my previous employers": 0.5,
        "i was aware of some": 0.5,
        "no, i only became aware later": 0.5,

        # Complete Positives (1)
        "yes": 1,
        "yes, they all did": 1,
        "yes, always": 1,
        "yes, all of them": 1,
        "yes, i was aware of all of them": 1,
        "yes, at all of my previous employers": 1,

        # Uncertainty/NA (np.nan)
        "not eligible for coverage / n/a": np.nan,
        "i don't know": np.nan,
        "n/a (not currently aware)": np.nan,
        "i am not sure": np.nan,
        "n/a": np.nan,
        "maybe": np.nan,
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

def analyze_pca_components(pca, feature_names):
    """Analyze the principal components of a PCA model.
    Parameters:
    pca (PCA): The fitted PCA model.
    feature_names (list): The list of feature names.
    Returns:
    DataFrame: A DataFrame with the feature loadings for the first two principal components.
    """
    # Get the feature loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=feature_names
    )

    # Get absolute loadings for first two PCs
    pc1_loadings = abs(loadings['PC1'])
    pc2_loadings = abs(loadings['PC2'])

    # Create a dataframe with the loadings
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'PC1_loading': pc1_loadings,
        'PC2_loading': pc2_loadings
    })

    # Sort by importance in PC1 and PC2
    pc1_top = importance_df.nlargest(5, 'PC1_loading')
    pc2_top = importance_df.nlargest(5, 'PC2_loading')

    return importance_df.sort_values(by=['PC1_loading', 'PC2_loading'], ascending=False)

def plot_correlation_heatmap(df):
    """Plot a correlation heatmap for the given DataFrame.
    Parameters:
    df (DataFrame): The DataFrame to plot the heatmap for.
    """
    # Compute the correlation matrix
    cor_mat = df.corr(method='pearson')

    # Create a larger figure size to accommodate more features
    plt.figure(figsize=(20, 16))

    # Create heatmap
    ax = sns.heatmap(cor_mat,
                     vmin=-1,
                     vmax=1,
                     annot=True,
                     fmt='.2f',
                     cmap='coolwarm',
                     annot_kws={'size': 8})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Compute the correlation matrix
    cor_mat = df.corr(method='pearson')

    # Create a larger figure size to accommodate more features
    plt.figure(figsize=(20, 16))

    # Create heatmap
    ax = sns.heatmap(cor_mat,
                     vmin=-1,
                     vmax=1,
                     annot=True,
                     fmt='.2f',
                     cmap='coolwarm',
                     annot_kws={'size': 8})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show plot
    plt.show()

def perform_pca(df):
    # 1. Standardize the data
    X_std = StandardScaler().fit_transform(df)

    # 2. Compute PCA and get explained variance ratio
    pca = PCA().fit(X_std)
    var_exp = pca.explained_variance_ratio_
    print("Explained variance per PC:", var_exp)

    # 3. Calculate cumulative explained variance
    cum_var_exp = np.cumsum(var_exp)
    print("Cumulative Explained Variance:", cum_var_exp)

    # 4. Display the proportion of variance explained by first two PCs
    print("Explained variance by PC1 and PC2:", sum(var_exp[0:2]))

    # 5. Project data to two-dimensional space
    Y = PCA(n_components=2).fit_transform(X_std)

    # 6. Visualize the projected data
    plt.figure(figsize=(8, 6))
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Data projected onto first two principal components')
    plt.show()

def calculate_feature_variances(df):
    # Calculate variance for each feature using original data
    feature_variances = pd.DataFrame({
        'Feature': df.columns,
        'Variance': df.var()
    })

    # Sort features by variance in descending order
    feature_variances = feature_variances.sort_values('Variance', ascending=False)

    # Display top features by variance
    print("Feature variances (original data):")
    print(feature_variances)

    # Calculate variance for each feature using original data
    feature_variances = pd.DataFrame({
        'Feature': df.columns,
        'Variance': df.var()
    })

    # Sort features by variance in descending order
    feature_variances = feature_variances.sort_values('Variance', ascending=False)

    # Display top features by variance
    print("Feature variances (original data):")
    print(feature_variances)

    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_variances)), feature_variances['Variance'])
    plt.xticks(range(len(feature_variances)), feature_variances['Feature'], rotation=45, ha='right')
    plt.title('Feature Variances (Original Data)')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.tight_layout()
    plt.show()

def analyze_optimal_clusters(df, k_range=(2, 11)):
    """
    Analyze optimal number of clusters using KElbowVisualizer and silhouette score
    :param df: DataFrame
    :param k_range: tuple, range of k values to test
    :return: X_scaled, results
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Create figure for subplots
    plt.figure(figsize=(15, 5))

    # Plot 1: KElbowVisualizer
    plt.subplot(1, 2, 1)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 8), timings=False, ax=plt.gca())
    visualizer.fit(df)
    visualizer.ax.set_ylabel('Elbow Score', labelpad=30)

    # Plot 2: Silhouette Analysis
    plt.subplot(1, 2, 2)

    # Calculate silhouette scores
    silhouette_scores = []
    k_values = range(2, k_range[1])

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Plot silhouette scores
    plt.plot(k_values, silhouette_scores, 'ro-', marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print the silhouette scores for detailed analysis
    results = pd.DataFrame({
        'k': list(k_values),
        'silhouette_score': silhouette_scores
    })
    print("\nSilhouette scores for each k:")
    print(results.to_string(index=False))

    return X_scaled, results

def analyze_clusters(df):
    """
    Perform k-means clustering and analyze the results
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to original dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    # Basic cluster statistics
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("\nCluster Sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} samples ({size/len(df)*100:.1f}%)")

    # Calculate cluster characteristics
    cluster_means = df_clustered.groupby('Cluster').mean()
    cluster_std = df_clustered.groupby('Cluster').std()

    # Find distinctive features for each cluster
    print("\nDistinctive Features per Cluster:")
    for cluster in range(3):
        # Calculate z-scores for this cluster's means
        z_scores = (cluster_means.loc[cluster] - df.mean()) / df.std()

        # Get top distinctive features (most deviant from overall mean)
        distinctive_features = z_scores.abs().sort_values(ascending=False)[:5]

        print(f"\nCluster {cluster} ({cluster_sizes[cluster]} samples):")
        for feature in distinctive_features.index:
            direction = "higher" if z_scores[feature] > 0 else "lower"
            print(f"- {feature}: {direction} than average "
                  f"(z-score: {z_scores[feature]:.2f})")

    # Calculate feature importance for cluster separation
    feature_importance = pd.DataFrame(
        scaler.transform(cluster_means),
        columns=df.columns,
        index=[f"Cluster {i}" for i in range(3)]
    )

    return df_clustered, feature_importance
