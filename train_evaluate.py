"""
Phase 2-4: Modeling and Evaluation Script
Training Linear Regression and Random Forest on Dubai Real Estate Data.
Target: log_price (log-transformed price)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join("cleaned_dubai_realestate.csv"))

# 2. Define Features and Target
# We use 'log_price' as target for training to handle skewness
# We must drop 'price' (original target) and 'log_price' from features
X = df.drop(columns=["price", "log_price"])
y = df["log_price"]

print(f"Dataset Shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Target: log_price")

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train sizes: X={X_train.shape}, y={y_train.shape}")
print(f"Test sizes:  X={X_test.shape}, y={y_test.shape}")

# 4. Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 5. Model 2: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 6. Evaluation Function
def evaluate_model(name, y_true, y_pred, y_true_orig=None):
    # Metrics on LOG scale (for model performance)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Metrics on ORIGINAL scale (for business interpretation)
    # We invert the log transformation: exp(y) - 1
    y_pred_orig = np.expm1(y_pred)
    if y_true_orig is None:
        y_true_orig = np.expm1(y_true)
    
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    
    print(f"\n--- {name} Results ---")
    print(f"Log-Scale RMSE: {rmse_log:.4f}")
    print(f"Log-Scale MAE:  {mae_log:.4f}")
    print(f"R² Score:       {r2:.4f}")
    print(f"Orig-Scale RMSE: {rmse_orig:,.0f} AED")
    print(f"Orig-Scale MAE:  {mae_orig:,.0f} AED")
    
    return {
        "Model": name, 
        "RMSE (Log)": rmse_log, "MAE (Log)": mae_log, "R²": r2,
        "RMSE (AED)": rmse_orig, "MAE (AED)": mae_orig
    }

metrics = []
metrics.append(evaluate_model("Linear Regression", y_test, y_pred_lr))
metrics.append(evaluate_model("Random Forest", y_test, y_pred_rf))

# 7. Comparison Table
results_df = pd.DataFrame(metrics)
print("\n=== Model Comparison Table ===")
print(results_df.to_string(index=False))

# 8. Feature Importance (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False).head(10)

print("\n Top 10 Important Features (Random Forest) ")
print(feature_imp_df.to_string(index=False))

# Save results for report
with open("results_summary.txt", "w", encoding="utf-8") as f:
    f.write(" Model Comparison \n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n Top 10 Features \n")
    f.write(feature_imp_df.to_string(index=False))

print("\nResults saved to results_summary.txt")


