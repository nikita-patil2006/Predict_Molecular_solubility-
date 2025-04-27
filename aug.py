import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('/Users/allenpeter/Desktop/Dataset augmentation/delaney_solubility_with_descriptors.csv')

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Function to add Gaussian noise
def add_gaussian_noise(df, columns, noise_level=0.02):
    """Add Gaussian noise to specified columns to generate synthetic data."""
    noisy_df = df.copy()
    for column in columns:
        std_dev = noise_level * noisy_df[column].std()
        noisy_df[column] += np.random.normal(0, std_dev, size=df.shape[0])
    return noisy_df

# Choose numeric columns (excluding target variable)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('logS')

# Define the required number of additional rows
n_samples_needed = 100000

# Estimate how many times we need to augment the dataset
num_iterations = max(1, -(-n_samples_needed // df.shape[0]))  # Ceiling division

# Generate augmented data multiple times
augmented_data_list = []
for i in range(num_iterations):
    noise_factor = 0.02 + (i * 0.002)  # Gradually increase noise for diversity
    augmented_part = add_gaussian_noise(df, numeric_columns, noise_level=noise_factor)
    augmented_data_list.append(augmented_part)

# Combine all augmented versions
augmented_data = pd.concat(augmented_data_list, ignore_index=True)

# Ensure we have exactly 100,000 additional samples
if augmented_data.shape[0] >= n_samples_needed:
    augmented_data = augmented_data.sample(n_samples_needed, replace=False).reset_index(drop=True)
else:
    print(f"Warning: Only generated {augmented_data.shape[0]} rows instead of {n_samples_needed}. Using all available.")

# Combine original data with augmented data
df_augmented = pd.concat([df, augmented_data]).drop_duplicates().reset_index(drop=True)

# Save final dataset
df_augmented.to_csv('/Users/allenpeter/Desktop/Dataset augmentation/augmented_data.csv', index=False)

# Print dataset summary
print(f"Final dataset shape: {df_augmented.shape}")
print("Sample augmented data:")
print(df_augmented.head())
