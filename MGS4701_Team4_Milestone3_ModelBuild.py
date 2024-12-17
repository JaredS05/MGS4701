#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Filename: MGS4701_Team4_Milestone1.ipynb
# Author: Jared Sheridan, Evan Kavanagh
# Created: 2024-12-16
# Version: 1.0
# Description: 
    Machine Learning and Model Creation for MSRP Prediction Project
"""


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')


# In[3]:


print("MGS4701 Team 4")


# In[4]:


df = pd.read_csv("CarDataset_Normalized.csv")


# In[5]:


df.head()


# In[6]:


##Determining MSE and R² Score Using Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Drop non-numeric columns ('Make', 'Model') and set 'MSRP' as the target variable
features = df.drop(columns=['Make', 'Model', 'MSRP', 'MSRP_UnLogged'])
target = df['MSRP']

#Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("The MSE value is: ", mse)
print("The R² value is: ", r2)

print("y_test:", y_test.values)
print("y_pred:", y_pred)


#The linear regression model achieved the following performance on the test set:

#Mean Squared Error (MSE): 0.035 (lower is better)
#R² Score: 0.8533 (closer to 1 indicates better fit)
#This suggests the model explains approximately 85.33% of the variance in the target variable using the given features. 


# In[7]:


##Linear Regression Visualization

import matplotlib.pyplot as plt

#y_test = y_test
#y_pred = y_pred

#Calculate the range of the data
min_value = min(min(y_test), min(y_pred))
max_value = max(max(y_test), max(y_pred))

#Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')

#Ideal fit line spanning the full range of the model
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label='Ideal Fit')

#Add titles and labels
plt.title('Actual vs Predicted Values for MSRP')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')

#Set axis limits to ensure the ideal fit line spans the entire range
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

#Add legend and grid
plt.legend()
plt.grid(True)

#Show the plot
plt.show()


# In[8]:


##Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

#Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

#Train the model on the training data
rf_model.fit(X_train, y_train)

#Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)

#Evaluate the model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

#Print results
print("The MSE value is: ", rf_mse)
print("The R² value is: ", rf_r2)

#This suggests that the Random Forest Regressor captures the data's patterns much better than the linear regression model.


# In[9]:


##Random Forest Regressor Visualization

import matplotlib.pyplot as plt

#Calculate the range of the data
min_value = min(min(y_test), min(rf_y_pred))
max_value = max(max(y_test), max(rf_y_pred))

#Scatter plot of actual vs predicted values for Random Forest Regressor
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_y_pred, alpha=0.6, color='seagreen', label='Predicted vs Actual')

#Ideal fit line spanning the full range of the data
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label='Ideal Fit')

#Add title and axis labels
plt.title('Random Forest Regressor: Actual vs Predicted MSRP')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')

#Set axis limits to ensure the ideal fit line spans the entire range
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

#Add legend and grid
plt.legend()
plt.grid(True)

#Show the plot
plt.show()



# In[10]:


##Feature Importance from Randome Forest Model


#Extract feature importances from the Random Forest model
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)


#Group features by prefix and summing their importances
prefix_groups = {
    'Market_Category': feature_importances[feature_importances['Feature'].str.startswith('Market_Category_')]['Importance'].sum(),
    'Transmission_Type': feature_importances[feature_importances['Feature'].str.startswith('Transmission_')]['Importance'].sum(),
    'Drive_Type': feature_importances[feature_importances['Feature'].str.startswith('Drive_')]['Importance'].sum(),
    'Fuel_Type': feature_importances[feature_importances['Feature'].str.startswith('FuelType_')]['Importance'].sum()
}

#Add specific features to compare
prefix_groups['Car_Age'] = feature_importances[feature_importances['Feature'] == 'Car_Age']['Importance'].values[0]
prefix_groups['Engine_HP'] = feature_importances[feature_importances['Feature'] == 'Engine_HP']['Importance'].values[0]
prefix_groups['Popularity'] = feature_importances[feature_importances['Feature'] == 'Popularity']['Importance'].values[0]
prefix_groups['average_MPG'] = feature_importances[feature_importances['Feature'] == 'average_MPG']['Importance'].values[0]

#Prepare the data for visualization
group_comparison = pd.DataFrame({
    'Category': list(prefix_groups.keys()),
    'Total Importance': list(prefix_groups.values())
}).sort_values(by='Total Importance', ascending=False)

#Visualize of grouped importances
plt.figure(figsize=(10, 6))
plt.barh(group_comparison['Category'], group_comparison['Total Importance'], align='center')
plt.xlabel('Total Importance')
plt.ylabel('Category / Feature')
plt.title('Combined Feature Importance vs Individual Features')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# In[11]:


##Better Visualization of Feature Importance from Random Forest Model

import matplotlib.cm as cm
import matplotlib.colors as mcolors

#Normalize the values to [0, 1] for colormap
norm = mcolors.Normalize(vmin=group_comparison['Total Importance'].min(), vmax=group_comparison['Total Importance'].max())
cmap = cm.get_cmap('Oranges')  

#Get colors for each bar based on importance
colors = [cmap(norm(value)) for value in group_comparison['Total Importance']]

#Visualization of grouped importances with colormap
plt.figure(figsize=(10, 6))
plt.barh(group_comparison['Category'], group_comparison['Total Importance'], align='center', color=colors)
plt.xlabel('Total Importance')
plt.ylabel('Category / Feature')
plt.title('Combined Feature Importance vs Individual Features (with Colormap)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Total Importance')
plt.show()


# In[12]:


##XGBoost Model

from xgboost import XGBRegressor

#Initialize the XGBoost Regressor with default parameters
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

#Train the model on the training data
xgb_model.fit(X_train, y_train)

#Make predictions on the test set
xgb_y_pred = xgb_model.predict(X_test)

#Evaluate the model
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

#Print the results
print("The MSE value is: ", xgb_mse)
print("The R² value is: ", xgb_r2)


# In[13]:


##XGBoost Model using Bayesian Optimization

from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

#Define the function to optimize
def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    model = XGBRegressor(
        n_estimators=int(n_estimators), 
        max_depth=int(max_depth), 
        learning_rate=learning_rate, 
        subsample=subsample, 
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return -mean_squared_error(y_test, preds)  # Negative because BayesianOptimization maximizes

#Define the parameter bounds
param_bounds = {
    'n_estimators': (50, 250),
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0)
}

#Initialize the optimizer
optimizer = BayesianOptimization(f=xgb_cv, pbounds=param_bounds, random_state=42)

#Perform optimization
optimizer.maximize(init_points=5, n_iter=20)

#Extract the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])  # Convert to integer
best_params['max_depth'] = int(best_params['max_depth'])

#Train a new model with the best parameters
best_model = XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

#Make predictions and evaluate
xgb_y_pred = best_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

print("Best Parameters:", best_params)
print("The MSE value is: ", xgb_mse)
print("The R² value is: ", xgb_r2)




# In[14]:


##Feature Importance Visualization based on XGBoost Model

#Feature importance visualization
xgb_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

#Bar chart for top features

#Normalize the values to [0, 1] for colormap
norm = mcolors.Normalize(vmin=group_comparison['Total Importance'].min(), vmax=group_comparison['Total Importance'].max())
cmap = cm.get_cmap('Oranges')  
#Get colors for each bar based on importance
colors = [cmap(norm(value)) for value in group_comparison['Total Importance']]
#Visualization of grouped importances with colormap
plt.figure(figsize=(10, 6))
plt.barh(group_comparison['Category'], group_comparison['Total Importance'], align='center', color=colors)
plt.xlabel('Total Importance')
plt.ylabel('Category / Feature')
plt.title('Combined Feature Importance vs Individual Features (with Colormap)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Total Importance')  # Add a colorbar
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Example y_test and xgb_y_pred data
# y_test = [...]  # Replace with actual test values
# xgb_y_pred = [...]  # Replace with predicted values from the XGBoost Regressor

# Generate example data
# y_test = np.random.rand(100) * 100
# xgb_y_pred = y_test + np.random.normal(0, 10, 100)

#Calculate the range of the data
min_value = min(min(y_test), min(xgb_y_pred))
max_value = max(max(y_test), max(xgb_y_pred))

#Scatter plot of actual vs predicted values with colormap
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    y_test, 
    xgb_y_pred, 
    c=xgb_y_pred,  
    cmap='Oranges',  
    alpha=0.6, 
    label='Predicted vs Actual'
)

#Add a colorbar
plt.colorbar(scatter, label='Predicted Value Intensity')

#Ideal fit line spanning the full range of the data
plt.plot([min_value, max_value], [min_value, max_value], color='blue', linestyle='--', linewidth=2, label='Ideal Fit')

#Add title and axis labels
plt.title('XGBoost Regressor: Actual vs Predicted MSRP')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')

#Set axis limits to ensure the ideal fit line spans the entire range
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

#Add legend and grid
plt.legend()
plt.grid(True)

#Show the plot
plt.show()



# In[15]:


offset_values = df['MSRP'] - np.log(df['MSRP_UnLogged'])
offset = offset_values.mean()

print("The offset value for MSRP and MSRP_UnLogged is:", offset)


# In[ ]:


from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

# Load dataset
df = pd.read_csv('CarDataset_Normalized.csv')

# Prepare features and target
features = df.drop(columns=['Popularity', 'Make', 'Model', 'MSRP', 'MSRP_UnLogged'])  # Drop non-numeric and target columns
target = df['MSRP']

# Categorical feature groups
categorical_features = {
    'Transmission': [col for col in features.columns if col.startswith('Transmission_')],
    'Drive': [col for col in features.columns if col.startswith('Drive_')],
    'Market_Category': [col for col in features.columns if col.startswith('Market_Category_')],
    'VehicleStyle': [col for col in features.columns if col.startswith('VehicleStyle_')],
    'Fuel_Type': [col for col in features.columns if col.startswith('Fuel_Type_')],
}

# Function to initialize, optimize, and train the model
def reset_and_train():
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define Bayesian Optimization function
    def xgb_cv(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        model = XGBRegressor(
            n_estimators=int(n_estimators), 
            max_depth=int(max_depth), 
            learning_rate=learning_rate, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return -mean_squared_error(y_test, preds)

    # Parameter bounds
    param_bounds = {
        'n_estimators': (50, 250),
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.2),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0)
    }

    # Perform Bayesian Optimization
    optimizer = BayesianOptimization(f=xgb_cv, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=20)

    # Extract best parameters
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    # Train the final model
    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    return final_model, X_train, X_test, y_train, y_test

# Function to handle categorical input
def get_categorical_input(options, feature_name):
    print(f"\nChoose one option for {feature_name}:")
    for idx, option in enumerate(options):
        print(f"{idx + 1}. {option}")
    while True:
        choice = input(f"Enter the number of your choice for {feature_name}: ")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                selected = options[choice_idx]
                return {opt: 1 if opt == selected else 0 for opt in options}
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# User input function
def get_user_input(feature_names):
    user_data = []
    print("\nPlease provide the following details about the vehicle:")
    for feature in feature_names:
        if feature not in sum(categorical_features.values(), []):  # Exclude categorical
            value = input(f"{feature}: ")
            try:
                user_data.append(float(value))
            except ValueError:
                print(f"Invalid input for {feature}. Please provide a numeric value.")
                return get_user_input(feature_names)  # Retry
    # Handle categorical features
    for group_name, options in categorical_features.items():
        encoded = get_categorical_input(options, group_name)
        user_data.extend(encoded[opt] for opt in options)
    return np.array(user_data).reshape(1, -1)

# Predict MSRP based on user input
def predict_msrp():
    # Reset and train the model
    final_model, X_train, X_test, y_train, y_test = reset_and_train()

    # Get user input
    user_features = get_user_input(features.columns)
    
    # Predict in log scale
    predicted_log_msrp = final_model.predict(user_features)
    
    # Reverse the log transformation
    offset = 5.7213
    predicted_msrp_unlogged = np.exp(predicted_log_msrp + offset)  # Apply transformation
    
    # Calculate the range to the nearest thousands
    predicted_value = predicted_msrp_unlogged[0]
    lower_bound = math.floor(predicted_value / 1000) * 1000  # Round down to nearest 1000
    upper_bound = lower_bound + 1000                        # Upper bound

    print(f"\nThe predicted MSRP for the vehicle is: ${lower_bound:,} - ${upper_bound:,} (Values may not be exact)")

# Run prediction (resets and trains the model each time)
predict_msrp()

