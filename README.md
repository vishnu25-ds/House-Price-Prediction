# 🏡 House Price Prediction using Machine Learning Regression Models  

An end-to-end machine learning project that predicts house sale prices using advanced regression models. This project (Team REG-02) integrates data cleaning, preprocessing, feature engineering, model training, evaluation, and prediction generation.  

---

## 🔬 Background & Objectives  

The **primary objective** of this project is to predict residential property sale prices based on diverse property characteristics.  

- **Inputs:** Lot size, area, number of bedrooms, neighborhood, year built, garage type, etc.  
- **Output:** Predicted sale price.  

### Why is this important?  
- 🏠 Helps **sellers** determine competitive prices  
- 👩‍👩‍👧 Helps **buyers** estimate fair values  
- 🏢 Assists **real estate agencies** with data-driven recommendations  

This project uses the **Kaggle Housing Dataset**:  
- **Training Data:** 1,460 rows × 81 columns  
- **Feature Types:**  
  - 38 numerical  
  - 43 categorical:contentReference[oaicite:3]{index=3}  

---

## 📊 Dataset Details  

- **Source:** Kaggle House Prices – Advanced Regression Techniques  
- **Target Variable:** `SalePrice`  
- **Dropped Columns (due to missingness):** `Alley`, `PoolQC`, `Fence`, `MiscFeature`, `Id`:contentReference[oaicite:4]{index=4}  
- **Handling Missing Values:**  
  - Numerical → median imputation  
  - Categorical → mode imputation  
- **Feature Transformations:**  
  - Converted `MSSubClass` to categorical  
  - One-hot encoding applied to categorical features  
- **Outlier Handling:**  
  - Features like `MasVnrArea`, `BsmtUnfSF` treated using IQR thresholds  
  - Outliers replaced with **median (BsmtUnfSF)** or **mean (MasVnrArea)**:contentReference[oaicite:5]{index=5}  

---
## 🏠 Features  

The dataset contains **80 features** (43 categorical + 38 numerical) that describe properties, their condition, and sale information.  

### Property & Lot Information  
- `MSSubClass` – Type of dwelling  
- `MSZoning` – Zoning classification  
- `LotFrontage` – Linear feet of street connected to property  
- `LotArea` – Lot size in square feet  
- `LotShape` – Property shape (Regular/Irregular)  
- `Neighborhood` – Physical location within Ames city limits  

### Building & Interior Features  
- `YearBuilt` – Year house was built  
- `OverallQual` – Overall material and finish quality (1–10)  
- `OverallCond` – Overall condition rating  
- `GrLivArea` – Above-ground living area (sq ft)  
- `BedroomAbvGr` – Bedrooms above ground level  
- `KitchenQual` – Kitchen quality  
- `TotRmsAbvGrd` – Total rooms above ground  

### Basement & Garage Features  
- `TotalBsmtSF` – Total basement area (sq ft)  
- `BsmtFinSF1` – Finished basement area (Type 1)  
- `BsmtUnfSF` – Unfinished basement area  
- `GarageType` – Garage location (attached, detached, etc.)  
- `GarageCars` – Garage capacity (number of cars)  
- `GarageArea` – Garage size (sq ft)  

### Exterior Features  
- `RoofStyle`, `RoofMatl` – Roof type & material  
- `Exterior1st`, `Exterior2nd` – Exterior covering on house  
- `MasVnrArea` – Masonry veneer area (sq ft)  
- `ExterQual` – Exterior material quality  

### Sale Information  
- `MoSold` – Month house was sold  
- `YrSold` – Year house was sold  
- `SaleType` – Type of sale (Normal, Auction, etc.)  
- `SaleCondition` – Condition of sale  

📌 **Dropped features** due to excessive missing values:  
- `Alley`, `PoolQC`, `Fence`, `MiscFeature`, and `Id`.  
---
## 📈 Exploratory Data Analysis (EDA)  

Performed with **Seaborn, Matplotlib, and FastEDA**.  

- Countplots for categorical features (`GarageType`, `LotShape`, `Neighborhood`)  
- Boxplots to detect outliers (`MasVnrArea`, `BsmtUnfSF`)  
- Correlation heatmap to detect strong feature relationships  
- Distribution plots for key numeric features  

📌 Example:  
![Neighborhood Distribution](images/neighborhood_dist.png)  
*Property distribution across neighborhoods*  

📌 Example:  
![Correlation Heatmap](images/correlation_heatmap.png)  
*Correlation of features with SalePrice*  

---

## 🛠️ Data Preprocessing & Feature Engineering  

### Steps Taken  
1. **Data Cleaning**  
   - Handled missing values  
   - Dropped sparse columns  
   - Removed duplicates  

2. **Outlier Handling**  
   - Applied IQR method  
   - Replaced extreme values with median/mean:contentReference[oaicite:6]{index=6}  

3. **Feature Engineering**  
   - Converted numerical → categorical (e.g., `MSSubClass`)  
   - One-hot encoded categorical variables  
   - Interaction features (`GrLivArea × OverallQual`)  

4. **Feature Selection with RFE**  
   - Recursive Feature Elimination chosen  
   - Retained **83 features** as optimal  
   - Reduced multicollinearity & improved interpretability:contentReference[oaicite:7]{index=7}  

📌 Example — Outliers Before & After:  
![Outliers Before](images/outliers_before.png)  
![Outliers After](images/outliers_after.png)  

---

## 🤖 Models Trained  

We experimented with a wide range of models:  

| Model | Advantages | Limitations |
|-------|------------|-------------|
| **Linear Regression** | Easy to interpret | Sensitive to outliers, poor with non-linear data |
| **Decision Tree** | Handles non-linearity, interpretable | Overfitting, unstable |
| **Random Forest** | Robust, reduces overfitting | Slower, less interpretable |
| **SVR** | Handles linear & non-linear | High computation, tuning required |
| **XGBoost** | High accuracy, handles missing values | Complex tuning, memory usage |
| **LightGBM** | Fast, efficient | Overfitting, memory consumption |
| **CatBoost** | Handles categorical features well | Training time, memory usage |
| **KNN** | Intuitive, non-parametric | Poor in high dimensions:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}  

📌 Example — Workflow:  
![Workflow](images/workflow.png)  

---

## 📏 Evaluation Metrics  

We measured performance using:  

- **MAE (Mean Absolute Error)** – average error  
- **MSE (Mean Squared Error)** – penalizes large errors  
- **RMSE (Root Mean Squared Error)** – standard deviation of prediction errors  
- **R² Score** – variance explained:contentReference[oaicite:10]{index=10}  

---

## 📊 Results  

| Model | MAE | MSE (×10⁸) | RMSE | R² |
|-------|-----|------------|------|----|
| Linear Regression | 21,881 | 12.2 | 34,934 | 0.84 |
| SVR (Default) | 59,548 | 78.6 | 88,644 | -0.02 |
| SVR (Tuned) | 24,317 | 17.1 | 41,419 | 0.78 |
| Decision Tree | 27,470 | 17.5 | 41,811 | 0.77 |
| Random Forest | 18,404 | 8.8 | 29,691 | 0.89 |
| XGBoost | 18,628 | 8.4 | 28,982 | 0.90 |
| LightGBM | 16,633 | 8.1 | 28,374 | 0.90 |
| **CatBoost** | **15,765** | **7.5** | **27,313** | **0.91** |
| KNN (k=5) | 27,706 | 19.2 | 43,826 | 0.75 |:contentReference[oaicite:11]{index=11}  

📌 Example — Model Comparison Plot  
![Model Comparison](images/model_comparison.png)  

---

## 📉 Visualizations  

📌 **Actual vs Predicted Scatter Plots**  

- Linear Regression  
![LR](images/actual_vs_pred_lr.png)  

- Random Forest  
![RF](images/actual_vs_pred_rf.png)  

- CatBoost (Best Model)  
![CatBoost](images/actual_vs_pred_cat.png)  

📌 **KNN Hyperparameter Tuning**  
![KNN](images/knn_best_k.png)  

---

## 🏆 Key Takeaways  

- Feature selection (RFE) improved model interpretability & accuracy  
- Boosting models (CatBoost, XGBoost, LightGBM) performed best  
- CatBoost achieved the **highest R² (0.91)**  
- Cross-validation stabilized performance:contentReference[oaicite:12]{index=12}  

---

## 🚀 Future Work  

- Hyperparameter tuning for further accuracy  
- Ensemble stacking (XGBoost + CatBoost + LightGBM)  
- Integrate external features (economic/market trends)  
- Deploy via **Flask / Streamlit** for real-time predictions:contentReference[oaicite:13]{index=13}  

---

## 📂 Project Structure  

