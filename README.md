# House-Price-Prediction

# Housing Price Prediction using Machine Learning

This repository contains a machine learning project for predicting house prices using various regression models. The dataset used is the `train.csv` file, and the goal is to predict the `SalePrice` of houses based on a variety of features.

## Libraries Used

The following libraries are used in this project:

- `matplotlib` - For data visualization and plotting graphs.
- `seaborn` - For statistical data visualization.
- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical operations.
- `lightgbm` - For LightGBM regression model.
- `xgboost` - For XGBoost regression model.
- `sklearn` - For machine learning tools such as regression models and metrics.
- `catboost` - For CatBoost regression model.
- `colorama` - For colored output in terminal.
- `datacleaner` - For data cleaning tasks (handling missing values and duplicates).
- `fasteda` - For performing exploratory data analysis (EDA).
- `tabulate` - For creating formatted tables for output.

## Overview

This project aims to predict house prices based on various features. Below is a detailed explanation of the steps taken throughout the project.

### 1. **Loading and Preprocessing the Dataset**
- **Dataset Loading**: The dataset (`train.csv`) is loaded using `pandas`, a Python library for data manipulation.
- **Shape of Dataset**: The shape of the dataset is printed, providing the number of rows (entries) and columns (features).
- **Data Type Identification**: The data types of each column are displayed to ensure they align with the type of analysis we wish to perform (e.g., numerical columns for continuous variables and categorical columns for categories).
- **Converting Numerical to Categorical**: The `MSSubClass` column is converted to a categorical data type because it represents different property types, not a continuous variable.

### 2. **Exploratory Data Analysis (EDA)**
- **Descriptive Statistics**: We use `df.describe()` to generate basic statistics of the dataset, including mean, standard deviation, and range, to understand the distribution of data.
- **Missing Values Analysis**: Missing data is identified and displayed using `df.isnull().sum()`, which helps determine the next steps for data cleaning.
- **Visualizations**:
  - **Count Plots**: Various categorical features (such as `GarageType`, `LotShape`, `BldgType`) are visualized using `seaborn.countplot`.
  - **Box Plots**: Box plots for numerical features like `MasVnrArea` and `BsmtUnfSF` are used to detect outliers.
  
### 3. **Handling Missing Values and Outliers**
- **Data Cleaning**: The `autoclean` function from the `datacleaner` package is used to handle missing values and duplicates.
- **Outlier Handling**: Outliers in the `MasVnrArea` and `BsmtUnfSF` columns are handled using the IQR method. The extreme values are replaced by the mean or median, which helps reduce the impact of outliers on model performance.

### 4. **Feature Engineering**
- **One-Hot Encoding**: Categorical features are encoded using one-hot encoding with `pd.get_dummies()`, making them suitable for machine learning models.
- **Feature Selection**: Recursive Feature Elimination (RFE) is used to select the most important features based on their contribution to the target variable `SalePrice`. This helps to reduce the feature space and improve model performance.

### 5. **Model Training and Evaluation**
- **Training Multiple Models**: Various regression models are trained to predict house prices:
  - `LinearRegression`
  - `SVR` (Support Vector Regression)
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `XGBoost`
  - `LGBMRegressor`
  - `CatBoostRegressor`
  - `KNeighborsRegressor`
  
- **Model Evaluation**: For each model, performance is evaluated using several metrics:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² (R-squared)**
  - **Mean Absolute Error (MAE)**

- **Model Comparison**: The performance of each model is compared by visualizing the **Actual vs Predicted** values for each model and calculating the error metrics for model selection.

### 6. **Hyperparameter Tuning for KNN**
- **Finding Best `k` for KNN**: The optimal number of neighbors (`k`) for the KNN model is found by plotting R² scores for different values of `k` (1, 3, 5, 7, 9). This is crucial to improving KNN's performance.

### 7. **Making Predictions on New Data**
- After training and evaluating the models, the best model (`CatBoostRegressor`) is applied to a test dataset (`test.csv`) to predict the house prices.
- **Saving Predictions**: The predicted house prices are saved to a CSV file (`REG-02-CKPT3.csv`) for further analysis or submission.

### 8. **Model Performance Summary**
- **Results Comparison**: A summary table using the `tabulate` library is generated, comparing key performance metrics (MAE, MSE, RMSE, R²) across all models. This provides a clear view of which model performed the best.

### 9. **Visualizations**
- **Actual vs Predicted Plots**: Scatter plots of actual vs predicted values for each model (Linear Regression, SVR, Decision Tree, Random Forest, etc.) are generated to visually assess model performance.
- **Error Distribution**: Plots showing how prediction errors are distributed across the models.
- **Feature Importance**: Important features are visualized for models like Random Forest, XGBoost, and CatBoost to understand which features have the greatest impact on house price prediction.

## Dataset

The dataset used for this project is a housing price dataset. It contains various features related to house attributes such as:
- MSSubClass
- GarageType
- LotShape
- Neighborhood
- YearBuilt
- TotalBsmtSF
- SalePrice (target variable)


