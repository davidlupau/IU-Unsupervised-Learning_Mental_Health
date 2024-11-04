import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# load clean data
df = pd.read_csv('mental_health_tech_cleaned.csv')

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

# plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.show()

def analyze_pca_components(pca, feature_names):
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

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for PC1
    ax1.barh(pc1_top['Feature'], pc1_top['PC1_loading'])
    ax1.set_title('Top Features in PC1')
    ax1.set_xlabel('Absolute Loading Value')

    # Plot for PC2
    ax2.barh(pc2_top['Feature'], pc2_top['PC2_loading'])
    ax2.set_title('Top Features in PC2')
    ax2.set_xlabel('Absolute Loading Value')

    plt.tight_layout()
    plt.show()

    return importance_df.sort_values(by=['PC1_loading', 'PC2_loading'], ascending=False)

# Use the function (you'll need to run this with your data)
feature_names = df.columns.tolist()
results = analyze_pca_components(pca, feature_names)

# Print the top features
print("\nTop features by contribution to PC1 and PC2:")
print(results.head(5))
#results.to_csv('pca_results.csv', index=False)

# Read the results
results = pd.read_csv('pca_results.csv')

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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your original dataframe is called 'df'
# Select the top 5 features
selected_features = [
    'past_mh_disorder',
    'mh_diagnostic_medical_pro',
    'mh_treatment',
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

# Basic statistics of selected features
print("\nBasic Statistics of Selected Features:")
print(df_selected.describe())