import pandas as pd
import matplotlib.pyplot as plt
from cleaning_dataset import drop_columns, correct_age

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Clean the dataset
df_column_cleaned = drop_columns(df)
df_age_cleaned = correct_age(df_column_cleaned)

# Display the percentage of self-employed and employees
percentage_self_employed = df_column_cleaned['self_employed'].value_counts(normalize=True) * 100
print(f"Percentage of self-employed: {percentage_self_employed[1]:.2f}%")
print(f"Percentage of employees: {percentage_self_employed[0]:.2f}%")

# Define the relevant columns for analysis
columns = ['family_history', 'past_mh_disorder', 'current_mh_disorder', 'mh_diagnostic_medical_pro']

# Create a dictionary to store the percentage of Yes/No for each question
yes_no_percentages = {}

for col in columns:
    yes_no_percentages[col] = df[col].value_counts(normalize=True) * 100  # Convert to percentages

# Create a DataFrame for plotting
yes_no_df = pd.DataFrame(yes_no_percentages)

# Transpose the DataFrame so the questions are on the x-axis
yes_no_df = yes_no_df.T

# Plot the grouped bar chart
yes_no_df.plot(kind='bar', figsize=(10, 6))

# Add labels and title
plt.title('Percentage of Yes/No Responses for Mental Health Questions', fontsize=16)
plt.ylabel('Percentage of Responses', fontsize=12)
plt.xlabel('Mental Health Questions', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.legend(title='Responses')
plt.tight_layout()
plt.show()