import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load your dataset
dataset_filepath = "/Users/allenpeter/Desktop/overfit/augmented_data.csv"
dataset = pd.read_csv(dataset_filepath)

# Assuming 'logS' is your target column
y = dataset['logS']
X = dataset.drop(columns=['logS'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
mod2 = RandomForestRegressor(n_estimators=150, random_state=42)
mod2.fit(X_train, y_train)

# Make predictions on both the training and test data
y_train_pred = mod2.predict(X_train)
y_test_pred = mod2.predict(X_test)

# Evaluate performance on the training data
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Evaluate performance on the test data
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print performance metrics
print("Training Performance:")
print(f"R²: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}\n")

print("Testing Performance:")
print(f"R²: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")
