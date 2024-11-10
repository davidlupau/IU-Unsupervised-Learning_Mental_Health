import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from functions import analyze_pca_components

# Load clean dataset
df = pd.read_csv('mental_health_tech_cleaned.csv')

# Calculations Pearson correlation matrix
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

# Drop columns with high correlation
columns_to_drop = [
        'discuss_mh_with_coworkers', 'self_employed_0', 'mh_treatment', 'tech_organization_0.0',
        'gender_Non-binary/Other', 'gender_Female', 'family_history_0.5', 'family_history_0.0',
        'past_mh_disorder', 'mh_diagnostic_medical_pro'
    ]
df = df.drop(columns=columns_to_drop, axis=1)

# Standardizing the data
X_std = StandardScaler().fit_transform(df)

pca = PCA().fit(X_std)

# extract the explained variance ratios
var_exp = pca.explained_variance_ratio_
print("Explained variance ratio", var_exp)

# calculate the explained cumulative variance
cum_var_exp = np.cumsum(var_exp)
print("Cumulative variance", cum_var_exp)

# extract the Eigenvectors
eig_vecs = pca.components_

# use PCA to project the data to a two-dimensional feature space
Y = PCA(n_components=2).fit(X_std).transform(X_std)
pca = PCA().fit(X_std)

# Use the analyze_pca_components function
feature_names = df.columns.tolist()
results = analyze_pca_components(pca, feature_names)

# Print the top features
print("\nTop features by contribution to PC1 and PC2:")
print(results.head(5))

# Sort by PC1 and PC2 loadings
top_features_pc1 = results.nlargest(10, 'PC1_loading')
top_features_pc2 = results.nlargest(10, 'PC2_loading')

# Calculate total loading
results['Total_loading'] = results['PC1_loading'] + results['PC2_loading']
top_overall = results.nlargest(10, 'Total_loading')

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot PC1 top features
sns.barplot(data=top_features_pc1, x='PC1_loading', y='Feature', ax=ax1, color='blue', alpha=0.6)
ax1.set_title('Top 10 Features in PC1')
ax1.set_xlabel('Loading Value')

# Plot PC2 top features
sns.barplot(data=top_features_pc2, x='PC2_loading', y='Feature', ax=ax2, color='green', alpha=0.6)
ax2.set_title('Top 10 Features in PC2')
ax2.set_xlabel('Loading Value')

plt.tight_layout()
#plt.show()

# Print the top overall features
print("\nTop 10 Most Important Features (Combined PC1 and PC2 loadings):")
for idx, row in top_overall.iterrows():
    print(f"{row['Feature']}: {row['Total_loading']:.3f} (PC1: {row['PC1_loading']:.3f}, PC2: {row['PC2_loading']:.3f})")

# Select the top 5 features
selected_features = [
    'discuss_mh_with_manager',
    'observed_negative_consequences_coworkers',
    'career_impact',
    'discuss_mh_with_employer_negative',
    'current_mh_disorder'
]

# Create new dataframe with selected features
df_selected = df[selected_features]

# Create correlation matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(df_selected.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.show()

# Find the optimal number of clusters using the Elbow and silhouette methods
# create a k-Means model an Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), \
   timings=False)
# fit the visualizer and show the plot
visualizer.fit(df_selected)
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
