import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from functions import plot_correlation_heatmap, perform_pca, calculate_feature_variances

# Load clean dataset
df = pd.read_csv('mental_health_tech_cleaned.csv')

# Calculations Pearson correlation matrix
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
perform_pca(df)

# Calculate variance for each feature to select the most relevant ones
calculate_feature_variances(df)

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

# Find the optimal number of clusters using the Elbow and silhouette methods
# create a k-Means model an Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), \
   timings=False)
# fit the visualizer and show the plot
visualizer.fit(df)
visualizer.ax.set_ylabel('Elbow Score', labelpad=30)
visualizer.show()

# Calculate the silhouette score for different numbers of clusters
# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Trying different numbers of clusters and storing silhouette scores
silhouette_scores = []
K = range(2, 11)  # Trying clusters from 2 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True)
plt.show()

# k-Means clustering with 3 clusters
