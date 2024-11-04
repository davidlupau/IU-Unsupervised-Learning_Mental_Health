import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns

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
        'gender_Non-binary/Other', 'gender_Female', 'family_history_0.5', 'family_history_0.0'
    ]
df = df.drop(columns=columns_to_drop, axis=1)



# Find the optimal number of clusters using the Elbow and silhouette methods
# create a k-Means model an Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), \
   timings=True)
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
