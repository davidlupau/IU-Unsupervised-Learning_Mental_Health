import pandas as pd
from sklearn import mixture

# Load the dataset
df = pd.read_csv('selected_features.csv')

# Assuming the relevant features are already selected and present in df
X = df[['past_mh_disorder', 'mh_diagnostic_medical_pro', 'mh_treatment', 'discuss_mh_with_employer_negative', 'current_mh_disorder']].values

# Specify Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', random_state=0)

# Fit the model
gmm.fit(X)

# Extract the cluster predictions according to the highest probability
labels = gmm.predict(X)

# Add the cluster labels to the original DataFrame
df['Cluster_Label'] = labels

# Save the updated DataFrame to a new CSV file
df.to_csv('clustered_data.csv', index=False)