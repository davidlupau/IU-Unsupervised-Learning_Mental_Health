import pandas as pd
from functions import plot_correlation_heatmap, perform_pca, calculate_feature_variances, analyze_optimal_clusters, analyze_clusters

# Load clean dataset
df = pd.read_csv('mental_health_tech_cleaned.csv')

# Calculations Pearson correlation matrix
print("Pearson correlation matrix")
plot_correlation_heatmap(df)

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

# Check the number of features and list them
print("Number of features: " + str(len(df.columns)))
print("List of features:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# Perform PCA on the dataset and analyze the principal components
print("\nResults of PCA:")
pca_results = perform_pca(df)

# Calculate variance for each feature to select the most relevant ones
print("\nFeature variances:")
calculate_feature_variances(df)

# List of features with variance >= 0.147 + remote_worker
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
    'self_employed_0',
    'remote_worker'
]

# Create new dataframe with only selected features
df = df[selected_features]

# Calculate the optimal number of clusters using the elbow method and silhouette score
analyze_optimal_clusters(df, k_range=(2, 11))

# Run the analysis
X_scaled, results = analyze_optimal_clusters(df)

# k-Means clustering with 3 clusters
print("\nResults of k-Means clustering:")
df_clustered, feature_importance = analyze_clusters(df)

# Investigate mental health score by cluster
# Calculate average mental health score per cluster
print("\nMental Health Score by Cluster:")
print(df_clustered.groupby('Cluster')['mental_health_score'].agg(['mean', 'std']))