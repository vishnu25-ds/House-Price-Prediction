#IMPORTING LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR,LinearSVR
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
from datacleaner import autoclean
from fasteda import fast_eda
from tabulate import tabulate

#LOADING DATASET
df=pd.read_csv("train.csv")
df

#SHAPE OF DATASET
print(f"{Fore.BLUE}The Shape of Dataset is: {df.shape}.{Style.RESET_ALL}")



#IDENTIFYING DATATYPES
df.info()






#CONVERTING NUMERICAL TO CATEGORICAL VALUES
df['MSSubClass']=df['MSSubClass'].astype('category')


#IDENTIFYING DATATYPES
df.info()




df.describe()  #UNDERSTANDING THE DATA



#IDENTIFYING MISSING VALUES
missing_values = df.isnull().sum()
missing_values1 = missing_values[missing_values > 0]
print(f"{Fore.BLUE}Missing Values Count (before autoclean): {Style.RESET_ALL}")
print(missing_values1)




sns.countplot(x=df["GarageType"])
plt.title("Value Counts of Feature: GarageType")
plt.xticks(rotation=90)




sns.countplot(x=df["LotShape"])
plt.title("Value Counts of Feature: LotShape")
plt.xticks(rotation=90)




sns.countplot(x=df["BldgType"])
plt.title("Value Counts of Feature: BldgType")
plt.xticks(rotation=90)





sns.countplot(x=df["Neighborhood"])
plt.title("Value Counts of Feature: Neighborhood")
plt.xticks(rotation=90)






sns.countplot(x=df["LandSlope"])
plt.title("Value Counts of Feature: LandSlope")
plt.xticks(rotation=90)




sns.countplot(x=df["Utilities"])
plt.title("Value Counts of Feature: Utilities")
plt.xticks(rotation=90)





sns.countplot(x=df["YearBuilt"])
plt.title("Value Counts of Feature: YearBuilt")
plt.xticks(rotation=90)




sns.boxplot(x=df["MasVnrArea"])
plt.title("Outliers")






sns.boxplot(x=df["BsmtUnfSF"])
plt.title("Outliers")


df = autoclean(df)  # TO HANDLE MISSING AND DUPLICATES


#IDENTIFYING MISSING VALUES AFTER AUTOCLEAN
missing_values = df.isnull().sum()
missing_values1 = missing_values[missing_values > 0]
print(f"{Fore.BLUE}Missing Values Count (after autoclean): {Style.RESET_ALL}")
print(missing_values1)


fast_eda(df) #PERFORMING EDA FOR UNDERSTANDING THE PATTERNS




# BsmtUnfSF

# Calculate Q1, Q3, and IQR for the column 
Q1 = df['BsmtUnfSF'].quantile(0.25)
Q3 = df['BsmtUnfSF'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the median value for the column
median_value = df['BsmtUnfSF'].median()

# Replace outliers with the median
df['BsmtUnfSF'] = np.where(
    (df['BsmtUnfSF'] < (Q1 - 1.5 * IQR)) | (df['BsmtUnfSF'] > (Q3 + 1.5 * IQR)),
    median_value,
    df['BsmtUnfSF']
)




# MasVnrArea

# Calculate the mean value for the column MasVnrArea
mean_value = df['MasVnrArea'].mean()

# Calculate Q1, Q3, and IQR for the column MasVnrArea
Q1 = df['MasVnrArea'].quantile(0.25)
Q3 = df['MasVnrArea'].quantile(0.75)
IQR = Q3 - Q1

# Using the IQR method to identify outliers and replace them with the mean
df['MasVnrArea'] = np.where(
    (df['MasVnrArea'] < (Q1 - 1.5 * IQR)) | (df['MasVnrArea'] > (Q3 + 1.5 * IQR)),
    mean_value,
    df['MasVnrArea']
)





sns.boxplot(x=df["MasVnrArea"])
plt.title("After handling Outliers")



sns.boxplot(x=df["BsmtUnfSF"])
plt.title("After handling Outliers")




#USING HOT SHOT ENCODING FOR CATEGORICAL VALUES
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)



X = df.drop(columns=['Id','Alley','PoolQC','Fence','MiscFeature','SalePrice'], axis=1)  
y = df['SalePrice']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
selector = RFE(model, n_features_to_select=83)  # Select top features
selector = selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.support_]
print("Selected features:", selected_features)



include_columns=['MSZoning', 'LotFrontage', 'Street', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45',
       'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75',
       'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120',
       'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190','SalePrice']
df_new=df[include_columns] 




#USING HOT SHOT ENCODING FOR CATEGORICAL VALUES
#categorical_cols = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
#df_new = pd.get_dummies(df_new, columns=categorical_cols, drop_first=True)

#df = df.drop(columns=['Alley','PoolQC','Fence','MiscFeature'], axis=1)


X = df_new.drop(columns=['SalePrice'], axis=1)  
y = df_new['SalePrice']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





#TRAINING THE MODELS

#LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  #FITTING THE MODEL
yPredRegLR = model.predict(X_test)  #MAKING PREDICTINS
mse = round(mean_squared_error(y_test, yPredRegLR),2) #MEAN SQUARRED ERROR
rsme=round(np.sqrt(mse),2)  #ROOT MEAN SQUARRED ERROR
r2 = round(r2_score(y_test, yPredRegLR),2) #R SQARRED ERROR
print(f"\n{Fore.RED}LINEARRegressor:{Style.RESET_ALL}")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print ("Root Mean Squared Error",rsme)




#SVR
model2 = SVR()
model2.fit(X_train, y_train)
yPredRegSVR = model2.predict(X_test)
mse2 = round(mean_squared_error(y_test, yPredRegSVR),2)
rsme2=round(np.sqrt(mse2),2)
r22 = round(r2_score(y_test, yPredRegSVR),2)
print(f"\n{Fore.BLUE}SVR: {Style.RESET_ALL}")
print("Mean Squared Error:", mse2)
print("R^2 Score:", r22)
print ("Root Mean Squared Error",rsme2)






#DecisionTreeRegressor
model1 = DecisionTreeRegressor()
model1.fit(X_train, y_train)
yPredRegDTR = model1.predict(X_test)
mse1 = round(mean_squared_error(y_test, yPredRegDTR),2)
rsme1=round(np.sqrt(mse1),2)
r21 = round(r2_score(y_test, yPredRegDTR),2)
print(f"\n{Fore.BLUE}DecisionTreeRegressor: {Style.RESET_ALL}")
print("Mean Squared Error:", mse1)
print("R^2 Score:", r21)
print ("Root Mean Squared Error",rsme1)





#XGBRegressor
model5 = xgb.XGBRegressor()
model5.fit(X_train, y_train)
yPredRegXGB = model5.predict(X_test)
mse5 = round(mean_squared_error(y_test, yPredRegXGB),2)
rmse5 =round(np.sqrt(mse5),2)
r25 = round(r2_score(y_test, yPredRegXGB),2)
print(f"\n{Fore.BLUE}XGBRegressor:.{Style.RESET_ALL}")
print("Mean Squared Error:", mse5)
print("R^2 Score:", r25)
print("rmse:",rmse5)




#LGB
model6 = lgb.LGBMRegressor()
model6.fit(X_train, y_train)
yPredRegLGB = model6.predict(X_test)
mse6 = round(mean_squared_error(y_test, yPredRegLGB),2)
r26 = round(r2_score(y_test, yPredRegLGB),2)
rmse6 =round(np.sqrt(mse6),2)
print(f"\n{Fore.BLUE}LGBRegressor:{Style.RESET_ALL}")
print("Mean Squared Error:", mse6)
print("R^2 Score:", r26)
print("rmse:",rmse6)






#RANDOMFOREST
model7 = RandomForestRegressor()
model7.fit(X_train, y_train)
yPredRegRFR = model7.predict(X_test)
mse7 = round(mean_squared_error(y_test, yPredRegRFR),2)
r27 = round(r2_score(y_test, yPredRegRFR),2)
rmse7 =round(np.sqrt(mse7),2)
print(f"\n{Fore.BLUE}RANDOMFOREST:{Style.RESET_ALL}")
print("Mean Squared Error:", mse7)
print("R^2 Score:", r27)
print("rmse:",rmse7)




#CATABOOST
model8 = CatBoostRegressor()
model8.fit(X_train, y_train)
yPredRegCAT = model8.predict(X_test)
mse8 = round(mean_squared_error(y_test, yPredRegCAT),2)
r28 = round(r2_score(y_test, yPredRegCAT),2)
rmse8 =round(np.sqrt(mse8),2)
print(f"\n{Fore.BLUE}CATBOOSTREGRESSOR: {Style.RESET_ALL}")
print("Mean Squared Error:", mse8)
print("R^2 Score:", r28)
print("rmse:",rmse8)




#SVR
model2S = SVR(kernel='linear',  C=1, epsilon=0.2)
model2S.fit(X_train, y_train)
yPredRegSVRP = model2S.predict(X_test)
mse2S = round(mean_squared_error(y_test, yPredRegSVRP),2)
rsme2S=round(np.sqrt(mse2S),2)
r22S = round(r2_score(y_test, yPredRegSVRP),2)
print(f"\n{Fore.BLUE}SVR Tuned: {Style.RESET_ALL}")
print("Mean Squared Error:", mse2S)
print("R^2 Score:", r22S)
print ("Root Mean Squared Error",rsme2S)



#SVR TUNED
model2S = SVR(kernel='linear',  C=1, epsilon=0.2)
model2S.fit(X_train, y_train)
yPredRegSVRP = model2S.predict(X_test)
mse2S = round(mean_squared_error(y_test, yPredRegSVRP),2)
rsme2S=round(np.sqrt(mse2S),2)
r22S = round(r2_score(y_test, yPredRegSVRP),2)
print(f"\n{Fore.BLUE}SVR Tuned: {Style.RESET_ALL}")
print("Mean Squared Error:", mse2S)
print("R^2 Score:", r22S)
print ("Root Mean Squared Error",rsme2S)

print(f"\n{Fore.BLUE}KNN_BEST_K: {Style.RESET_ALL}")





from sklearn.neighbors import KNeighborsRegressor

#FINDING THE BEST K

#KNN
knn_r2_scores=[]
a=[1,3,5,7,9]
for i in a:
    model3 = KNeighborsRegressor(n_neighbors=i)
    model3.fit(X_train, y_train)  #FITTING THE MODEL
    yPredRegKNNT = model3.predict(X_test)  #MAKING PREDICTINS
    mse3 = round(mean_squared_error(y_test, yPredRegKNNT),2) #MEAN SQUARRED ERROR
    rsme3=round(np.sqrt(mse3),2)  #ROOT MEAN SQUARRED ERROR
    r23 = round(r2_score(y_test, yPredRegKNNT),4) #R SQARRED ERROR
    knn_r2_scores.append(r23)
    print(f'r2_for_{i}:',r23)
    
print(f"\n{Fore.BLUE}KNN_BEST_K_PLOT: {Style.RESET_ALL}")
    
#print(r2)   
plt.plot(a,knn_r2_scores,marker='*')
plt.title("r_2 TABLE for k between 1 and 9 ")
plt.xlabel("k")
plt.ylabel("r_2 ")
plt.show()


print(f"\n{Fore.BLUE}KNN: {Style.RESET_ALL}")

from sklearn.neighbors import KNeighborsRegressor

#TRAINING THE MODELS

#KNN
model3 = KNeighborsRegressor(n_neighbors=5)
model3.fit(X_train, y_train)  #FITTING THE MODEL
yPredRegKNN = model3.predict(X_test)  #MAKING PREDICTINS
mse3 = round(mean_squared_error(y_test, yPredRegKNN),2) #MEAN SQUARRED ERROR
rsme3=round(np.sqrt(mse3),2)  #ROOT MEAN SQUARRED ERROR
r23 = round(r2_score(y_test, yPredRegKNN),2) #R SQARRED ERROR
print(f"\n{Fore.RED}KNeighborsRegressor:{Style.RESET_ALL}")
print("Mean Squared Error:", mse3)
print("R^2 Score:", r23)
print ("Root Mean Squared Error",rsme3)







data=[
     ["Linear Regression",round(mean_absolute_error(y_test,yPredRegLR),2),round(mean_squared_error(y_test,yPredRegLR),2),round(np.sqrt(mean_squared_error(y_test,yPredRegLR)),2),round(r2_score(y_test,yPredRegLR),2)],
     ["SVR",round(mean_absolute_error(y_test,yPredRegSVR),2),round(mean_squared_error(y_test,yPredRegSVR),2),round(np.sqrt(mean_squared_error(y_test,yPredRegSVR)),2),round(r2_score(y_test,yPredRegSVR),2)],
     ["SVR TUNED",round(mean_absolute_error(y_test,yPredRegSVRP),2),round(mean_squared_error(y_test,yPredRegSVRP),2),round(np.sqrt(mean_squared_error(y_test,yPredRegSVRP)),2),round(r2_score(y_test,yPredRegSVRP),2)],
     ["Decision Tree Regression",round(mean_absolute_error(y_test,yPredRegDTR),2),round(mean_squared_error(y_test,yPredRegDTR),2),round(np.sqrt(mean_squared_error(y_test,yPredRegDTR)),2),round(r2_score(y_test,yPredRegDTR),2)],
     ["Random Forest Regression",round(mean_absolute_error(y_test,yPredRegRFR),2),round(mean_squared_error(y_test,yPredRegRFR),2),round(np.sqrt(mean_squared_error(y_test,yPredRegRFR)),2),round(r2_score(y_test,yPredRegRFR),2)],
     ["XGBoost",round(mean_absolute_error(y_test,yPredRegXGB),2),round(mean_squared_error(y_test,yPredRegXGB),2),round(np.sqrt(mean_squared_error(y_test,yPredRegXGB)),2),round(r2_score(y_test,yPredRegXGB),2)], 
     ["LGBoost",round(mean_absolute_error(y_test,yPredRegLGB),2),round(mean_squared_error(y_test,yPredRegLGB),2),round(np.sqrt(mean_squared_error(y_test,yPredRegLGB)),2),round(r2_score(y_test,yPredRegLGB),2)], 
     ["CAT",round(mean_absolute_error(y_test,yPredRegCAT),2),round(mean_squared_error(y_test,yPredRegCAT),2),round(np.sqrt(mean_squared_error(y_test,yPredRegCAT)),2),round(r2_score(y_test,yPredRegCAT),2)], 
     ["KNN",round(mean_absolute_error(y_test,yPredRegKNN),2),round(mean_squared_error(y_test,yPredRegKNN),2),round(np.sqrt(mean_squared_error(y_test,yPredRegKNN)),2),round(r2_score(y_test,yPredRegKNN),2)], 
    ]
columns=["Model Name","Mean Absolute Error","Mean Squared Error","Root Mean Squared Error","R Squared Error"]

print(tabulate(data, headers=columns, tablefmt="fancy_grid"))






#LR Actual vs. Predicted Values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegLR, color='black', label='LR Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for LR')
plt.legend()
plt.show()






#Actual vs. Predicted Values for SVR
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegSVR, color='black', label='SVR Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for SVR')
plt.legend()
plt.show()




# Actual vs. Predicted Values for SVR TUNED
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegSVRP, color='black', label='SVR Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for SVR TUNED')
plt.legend()
plt.show()




# Actual vs. Predicted Values for DTR
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegDTR, color='black', label='DTR Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for DTR')
plt.legend()
plt.show()






# Actual vs. Predicted Values for RFR
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegRFR, color='black', label='RFR Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for RFR')
plt.legend()
plt.show()



# Actual vs. Predicted Values for XGB
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegXGB, color='black', label='XGB Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for XGB')
plt.legend()
plt.show()




# Actual vs. Predicted Values for LGB
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegLGB, color='black', label='LGB Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for LGB')
plt.legend()
plt.show()




# Actual vs. Predicted Values for CAT
plt.figure(figsize=(12, 6))
plt.scatter(y_test, yPredRegCAT, color='black', label='CAT Predicted', alpha=0.7, marker='o')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for CAT')
plt.legend()
plt.show()



#READING PREDICTION CSV DATA AND CREATING DATAFRAME
dft=pd.read_csv("test.csv")
dft


dft['MSSubClass']=dft['MSSubClass'].astype('category')



dft = autoclean(dft)

#USING HOT SHOT ENCODING FOR CATEGORICAL VALUES
categorical_colst = dft.select_dtypes(include=['object', 'category']).columns.tolist()
dft = pd.get_dummies(dft, columns=categorical_colst, drop_first=True)

include_columns_2=['Id','MSZoning', 'LotFrontage', 'Street', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45',
       'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75',
       'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120',
       'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190']
dft_new=dft[include_columns_2] 



#MAKING PREDICTIONS
y_predt = model8.predict(dft_new)

#SAVING PREDICTIONS TO DATAFRAME
predictions_dft_new = pd.DataFrame({
    'ID': dft_new['Id'],  
    'SalePrice': y_predt  
})


#SAVING PREDICTED VALUES TO CSV
predictions_dft_new.to_csv('REG-02-CKPT3.csv', index=False)


