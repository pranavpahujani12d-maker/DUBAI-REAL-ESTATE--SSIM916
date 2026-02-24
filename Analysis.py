# DUBAI REAL ESTATE PRICE PREDICTION - REGRESSION ANALYSIS
# This script builds machine learning models to predict
# property prices in Dubai using regression techniques
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import data manipulation libraries
import pandas as pd                                    # Data loading and manipulation
import numpy as np                                     # Numerical computations and arrays

# Import machine learning tools
# Split data into train/test sets
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

# Construct the path to the cleaned Dubai real estate dataset
data_path = os.path.join("cleaned_dubai_realestate.csv")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

# Display how many rows (samples) and columns (features) we have
print(f"[OK] Loaded dataset: {df.shape}\n")

# STEP 2: Prepare Data for Regression
print("="*60)
print("         REGRESSION (Predicting log_price)")
print("="*60)

# Step 2a: Define columns to exclude — remove targets to avoid data leakage
drop_cols_reg = ["price", "log_price", "price_class"]

# Step 2b: Create feature matrix X by dropping the excluded columns
X_reg = df.drop(columns=[c for c in drop_cols_reg if c in df.columns])

# Step 2c: Create target variable y (log-transformed price)
# Log scale reduces the effect of extreme price outliers
y_reg = df["log_price"]

# Step 2d: Split into training (80%) and testing (20%) sets
# random_state=42 ensures reproducible results
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
# STEP 3: Normalize Features (Feature Scaling)
# Step 3a: Create a StandardScaler — rescales features to mean=0, std=1
scaler_r = StandardScaler()

# Step 3b: Fit on training data only and transform it
# (fitting on test data would cause data leakage)
X_train_r_scaled = scaler_r.fit_transform(X_train_r)

# Step 3c: Apply the same learned scaling to the test data
X_test_r_scaled = scaler_r.transform(X_test_r)

# STEP 4: Train Machine Learning Models
# Model 1: Random Forest Regressor
# Ensemble of 100 decision trees — captures non-linear price patterns
rf_reg = RandomForestRegressor(
    n_estimators=100,    # Number of trees in the forest
    random_state=42,     # Fixed seed for reproducible results
    n_jobs=1             # Set to 1 to avoid Windows multiprocessing issues
)
# Train on raw (unscaled) features — RF is scale-invariant
rf_reg.fit(X_train_r, y_train_r)

# Generate predictions on the test set
y_pred_rf_log = rf_reg.predict(X_test_r)

# Model 2: Linear Regression
# Simple interpretable baseline — requires scaled features
lin_reg = LinearRegression()

# Train on scaled training data
lin_reg.fit(X_train_r_scaled, y_train_r)

# Generate predictions on the scaled test set
y_pred_lin_log = lin_reg.predict(X_test_r_scaled)

# STEP 5: Define Function to Evaluate Model Performance
def print_reg_metrics(name, y_log_true, y_log_pred):
    """Calculate and display regression metrics.
    Parameters:
    - name       : Model name to display
    - y_log_true : Actual log-transformed prices (ground truth)
    - y_log_pred : Predicted log-transformed prices from the model"""

    # Step 5a: Convert log-scale predictions back to original AED price scale
    # np.expm1 reverses log1p (i.e., expm1(x) = e^x - 1)
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)

    # Step 5b: Calculate error metrics in actual price units (AED)
    # MAE: Average absolute difference between actual and predicted prices
    mae  = mean_absolute_error(y_true, y_pred)
    # RMSE: Square root of average squared differences (penalises large errors)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # R²: How well the model explains price variation (0 = bad, 1 = perfect)
    r2   = r2_score(y_true, y_pred)

    # Step 5c: Display results in a clear, readable format
    print(f"--- {name} ---")
    print(f"  R\u00b2 Score   : {r2:.4f}")        # Proportion of variance explained
    print(f"  MAE        : {mae:,.0f} AED")   # Average error in AED
    print(f"  RMSE       : {rmse:,.0f} AED\n")  # Root mean square error in AED

# STEP 6: Evaluate Both Models and Display Results
# Evaluate the Random Forest model's performance
print_reg_metrics("Random Forest Regressor", y_test_r, y_pred_rf_log)

# Evaluate the Linear Regression model's performance
print_reg_metrics("Linear Regression", y_test_r, y_pred_lin_log)

# Print footer line to mark end of analysis
print("="*60)
