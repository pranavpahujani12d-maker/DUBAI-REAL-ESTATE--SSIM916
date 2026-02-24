# DUBAI REAL ESTATE PRICE PREDICTION - REGRESSION ANALYSIS
# This script builds machine learning models to predict
# property prices in Dubai using regression techniques

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import data manipulation libraries
import pandas as pd 
# For numerical computations and arrays
import numpy as np   

# Import machine learning tools
 # Split data into train/test
from sklearn.model_selection import train_test_split 
# Normalize feature values
from sklearn.preprocessing import StandardScaler  
 # Tree-based regression model
from sklearn.ensemble import RandomForestRegressor  
# Linear regression model
from sklearn.linear_model import LinearRegression     

# Import evaluation metrics to measure model performance
from sklearn.metrics import (
    mean_absolute_error,   
    mean_squared_error,   
    r2_score               
)

# Utilities
import warnings
warnings.filterwarnings("ignore")  # Hide deprecation warnings
import os  

# STEP 1: Load the Data
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file containing the cleaned Dubai real estate data
data_path = os.path.join(script_dir, "DATA", "cleaned_dubai_realestate.csv")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

# Display how many rows (samples) and columns (features) we have
print(f"[OK] Loaded dataset: {df.shape}\n")

# STEP 2: Prepare Data for Regression
print("="*60)
print("         REGRESSION (Predicting log_price)")
print("="*60)

# Step 2a: Define which columns to exclude from features
# We remove the target variable and related price columns
drop_cols_reg = ["price", "log_price", "price_class"]

# Step 2b: Create feature matrix X by dropping the specified columns
# The X will be used as input to train the models
X_reg = df.drop(columns=[c for c in drop_cols_reg if c in df.columns])

# Step 2c: Create target variable y 
# predicting log_price 
y_reg = df["log_price"]

# Step 2d: Split data into training (80%) and testing (20%) sets
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
# STEP 3: Normalize Features (Feature Scaling)
# Step 3a: Create a StandardScaler object
# This will rescale all features to have mean=0 and std=1
scaler_r = StandardScaler()

# Step 3b: Learn scaling parameters from training data and apply the transformation
# This normalizes the training features
X_train_r_scaled = scaler_r.fit_transform(X_train_r)

# Step 3c: Apply the same scaling to test data using the training statistics
X_test_r_scaled = scaler_r.transform(X_test_r)

# STEP 4: Train Machine Learning Models
# Model 1: Random Forest Regressor
# Step 4a: Create a Random Forest model with 100 trees
rf_reg = RandomForestRegressor(
    n_estimators=100,    # Number of trees in the forest
    random_state=42,     # For reproducible results
    n_jobs=-1            # Use all CPU cores for parallel processing
)
# Step 4b: Train the Random Forest model on training data
rf_reg.fit(X_train_r, y_train_r)

# Step 4c: Make predictions on test data
y_pred_rf_log = rf_reg.predict(X_test_r)

# Model 2: Linear Regression
# Step 4d: Create a Linear Regression model
lin_reg = LinearRegression()

# Step 4e: Train Linear Regression on scaled training data
# (Linear Regression performs better with scaled features)
lin_reg.fit(X_train_r_scaled, y_train_r)

# Step 4f: Make predictions on scaled test data
y_pred_lin_log = lin_reg.predict(X_test_r_scaled)

# STEP 5: Define Function to Evaluate Model Performance
def print_reg_metrics(name, y_log_true, y_log_pred):
    """Calculate and display regression metrics.
    Parameters:
    - name: Model name to display
    - y_log_true: Actual log-transformed prices
    - y_log_pred: Predicted log-transformed prices"""
    
    # Step 5a: Convert log predictions back to original price scale
    # We use np.expm1 because prices were log-transformed using log(price + 1)
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    
    # Step 5b: Calculate error metrics in actual price units (AED)
    # MAE: Average absolute difference between actual and predicted prices
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE: Square root of average squared differences (penalizes large errors)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # R² Score: How well the model explains price variation (0=bad, 1=perfect)
    r2 = r2_score(y_true, y_pred)

    # Step 5c: Display results in a readable format
    print(f"--- {name} ---")
    print(f"  R² Score   : {r2:.4f}")
    print(f"  MAE        : {mae:,.0f} AED")
    print(f"  RMSE       : {rmse:,.0f} AED\n")

# STEP 6: Evaluate Both Models and Display Results
# Evaluate the Random Forest model's performance
print_reg_metrics("Random Forest Regressor", y_test_r, y_pred_rf_log)
# Evaluate the Linear Regression model's performance
print_reg_metrics("Linear Regression", y_test_r, y_pred_lin_log)
# Print footer line to mark end of analysis
print("="*60)
