# DUBAI-REAL-ESTATE--SSIM916
## 1. Introduction
This project aims to predict property prices in Dubai and identify the most critical factors influencing real estate value. Using a dataset of residential properties, we develop and compare regression models to provide actionable insights for buyers and investors.
## 2. Data Description
The analysis utilizes the "Dubai Real Estate" dataset, cleaned and prepared for regression.
- **Source:** `realestatedata.csv` (7,825 initial records)
- **Target Variable:** `price` (Transformed to `log_price` for modeling)
- **Key Features:** Property size (`size.sq.meter`), number of beds (`no.beds`), and location/neighborhood (`area.name`).
- **Data Cleaning:** Duplicates removed, ID columns dropped, numeric/categorical missing values imputed with median/mode, extreme price outliers removed via IQR method, and categorical features one-hot encoded.
- ## 3. Methodology
Two regression models were trained and evaluated on an 80/20 train-test split (`random_state=42`):
1.  **Linear Regression:** Serves as an interpretable baseline to establish linear relationships.
2.  **Random Forest Regressor:** A non-linear ensemble method chosen for its ability to capture complex feature interactions and robust performance.
**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** Measures average prediction error magnitude (lower is better).
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual prices.
- **R² Score:** Proportion of variance in the target explained by the model (higher is better).
- ## 4. Results
### Model Performance Comparison 
| **Random Forest** | **0.1873** | **0.8863** | **17,528 AED** | **10,877 AED** |
| Linear Regression | 0.2796 | 0.7466 | 22,852 AED | 15,854 AED |
**Selected Model:** The **Random Forest Regressor** is the superior model, explaining **88.6%** of the variance in property prices compared to 74.7% for the baseline. It reduces the average error (MAE) by approximately 5,000 AED per property.
## 5. Discussion
### Feature Importance Analysis
Using the Random Forest model, we identified the top drivers of property value:
1.  **Property Size (`size.sq.meter`):** The dominant factor, accounting for **50.2%** of the model's predictive power. Larger properties command higher prices, aligning with market intuition.
2.  **Location - Downtown Dubai:** The most influential neighborhood feature. Being in the city center significantly impacts value.
3.  **Location - Dubai Marina:** Another premium location with high predictive weight.
4.  **Number of Beds (`no.beds`):** A significant but secondary factor compared to size and location.
  ## 6. Limitations
- **Geographic Scope:** The model is specific to Dubai and may not generalize to other Emirates or international markets.
- **Data Constraints:** We assume the dataset is representative of the current market, but real estate prices are sensitive to external economic factors not captured here (e.g., interest rates, global events).
- ## 7. Future Directions
- **Advanced Modeling:** Explore Gradient Boosting (XGBoost/LightGBM) for potentially higher accuracy.
- **Explainability:** Implement SHAP (SHapley Additive exPlanations) values to provide granular, instance-level explanations for price predictions.
- **Temporal Analysis:** Incorporate time-series data to model price trends over seasons and years.
  ## 8. Replication
### Requirements
- Python 3.x
- Dependencies: pandas, numpy, scikit-learn
