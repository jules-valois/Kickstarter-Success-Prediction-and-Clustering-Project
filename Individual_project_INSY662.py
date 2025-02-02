#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:11:44 2024

@author: julesvalois
"""
------------------------------------------------------------------------------
##  GRADING FORMAT CODE IS AFTER 1.CLASSIFICATION IN 2.GRDAING CODE SECTION ##
------------------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
import random

random.seed(42)
np.random.seed(42)


# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  # if you want to see all rows as well
pd.set_option('display.width', 1000)  # adjust the width as needed


# Load the dataset
df = pd.read_excel('/Users/julesvalois/Downloads/Kickstarter.xlsx')

------------------------------------------------------------------------------
################################################################## 
################## 1.Classification ##############################
##################################################################
------------------------------------------------------------------------------


################################################################## 
################## DATA PROCESSING ###############################
##################################################################

# Filter the dataset to keep only rows where 'state' is 'successful' or 'failed'
df = df[df['state'].isin(['successful', 'failed'])]

# Convert goal to USD using static_usd_rate
df['goal_usd'] = df['goal'] * df['static_usd_rate']

# Calculate project duration in days
df['project_duration'] = (df['deadline'] - df['launched_at']).dt.days

# Check for and display columns with missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Columns with missing values:")
print(missing_values)

# Drop rows with missing values in the 'main_category' column
df = df.dropna(subset=['main_category'])


################################################################## 
################## FEATURE SELECTION #############################
##################################################################

# Define the selected variables for modeling
variables = ['state', 'category', 'country', 'goal_usd', 'deadline_month', 'deadline_weekday',
             'launched_at_month', 'launched_at_weekday', 'video', 'project_duration', 'name_len_clean',
             'blurb_len_clean']

# Select the relevant columns
new_df = df[variables]

# Assign "Others" to null values in the 'category' column
new_df['category'].fillna('Others', inplace=True)

# Drop duplicate rows
new_df = new_df.drop_duplicates()

# Drop remaining rows with any missing values
new_df = new_df.dropna()

# Encode the target variable
new_df['state'] = new_df['state'].replace({'successful': 1, 'failed': 0})

#Encoding Categorical Variables
new_df = pd.get_dummies(new_df, columns=['category', 'country', 'video', 'deadline_weekday', 'launched_at_weekday'], drop_first=True)

################################################################## 
################## OUTLIER DETECTION #############################
##################################################################

# Initialize Isolation Forest model for outlier detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the Isolation Forest model and predict outliers
outliers = iso_forest.fit_predict(new_df)

# Keep only the inliers
new_df = new_df[outliers == 1]

print("Dataset shape after outlier removal:", new_df.shape)

################################################################## 
################## Correlation and VIF ###########################
##################################################################
# Correlation
correlation_with_state = new_df.corr()['state'].sort_values(ascending=False)
print("Correlation of 'state' with all other variables:")
print(correlation_with_state)

# VIF
numeric_df = new_df.select_dtypes(include=['float64', 'int64'])
vif_data = add_constant(numeric_df)
vif_results = pd.DataFrame()
vif_results["feature"] = vif_data.columns
vif_results["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
print(vif_results)

################################################################## 
################## Classification Model  #########################
##################################################################
# Define the target variable 'y' and predictors 'X'
y = new_df['state']
X = new_df.drop(columns=['state'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the numerical features
numeric = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])

# Define the models to test
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),  # No random state needed for KNN
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
}


# Dictionary to store model scores
model_scores = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store scores
    model_scores[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Display model performance
for model_name, scores in model_scores.items():
    print(f"Model: {model_name}")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    print("\n")
    

################################################################## 
################## feature hyperparameters  ######################
##################################################################
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

##################################################################
# Step 1: Define Hyperparameter Grid for Random Forest
##################################################################
param_grid_rf = {
    'n_estimators': [100, 150, 200, 250],     # Number of trees
    'max_depth': [10, 20, 30, None],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20],      # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4, 8]          # Minimum samples at a leaf node
}

##################################################################
# Step 2: Perform GridSearchCV for Hyperparameter Tuning
##################################################################
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,                                     # 5-fold cross-validation
    verbose=1,
    scoring='accuracy',
    n_jobs=-1                                 # Use all CPU cores for faster computation
)


# Fit GridSearchCV on the training data
grid_search_rf.fit(X_train, y_train)

##################################################################
# Step 3: Retrieve the Best Model and Evaluate on the Test Set
##################################################################
# Retrieve the best estimator and parameters
optimized_rf = grid_search_rf.best_estimator_
best_params = grid_search_rf.best_params_
print("Best Parameters (GridSearchCV):", best_params)

# Make predictions on the test set
y_pred_rf = optimized_rf.predict(X_test)

# Evaluate the optimized model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='binary')
recall_rf = recall_score(y_test, y_pred_rf, average='binary')
f1_rf = f1_score(y_test, y_pred_rf, average='binary')

# Print performance metrics
print("Optimized Random Forest Performance:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

##################################################################
# Step 4: Predicted Probabilities and MSE
##################################################################
y_pred_probs = optimized_rf.predict_proba(X_test)[:, 1]
mse_rf = mean_squared_error(y_test, y_pred_probs)
print(f"MSE (Mean Squared Error): {mse_rf:.4f}")

##################################################################
# Step 5: Visualize Model Performance Metrics
##################################################################
# Create a bar chart for performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy_rf, precision_rf, recall_rf, f1_rf]

plt.figure(figsize=(8, 6))
plt.bar(metrics, scores, color='skyblue')
plt.title("Performance Metrics for Optimized Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

##################################################################
# Step 6: Feature Importance Analysis
##################################################################
# Retrieve feature importances
feature_importances = optimized_rf.feature_importances_

# Select and plot the top 20 features
top_indices = np.argsort(feature_importances)[::-1][:20]
top_features = X_train.columns[top_indices]
top_importances = feature_importances[top_indices]

plt.figure(figsize=(10, 6))
plt.bar(range(len(top_features)), top_importances, align='center')
plt.xticks(range(len(top_features)), top_features, rotation=90)
plt.title("Top 20 Feature Importances for Optimized Random Forest")
plt.ylabel("Feature Importance")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()



------------------------------------------------------------------------------
################################################################## 
###########        2. Grading CODE         #######################
##################################################################
------------------------------------------------------------------------------

# Load the dataset
grading_df = pd.read_excel('Kickstarter-Grading.xlsx')

################################################################## 
################## DATA PROCESSING ###############################
##################################################################

# Filter the dataset to keep only rows where 'state' is 'successful' or 'failed'
grading_df = grading_df[grading_df['state'].isin(['successful', 'failed'])]

# Convert goal to USD using static_usd_rate
grading_df['goal_usd'] = grading_df['goal'] * grading_df['static_usd_rate']

# Calculate project duration in days
grading_df['project_duration'] = (grading_df['deadline'] - grading_df['launched_at']).dt.days

# Check for and display columns with missing values
missing_values = grading_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Columns with missing values:")
print(missing_values)

# Drop rows with missing values in the 'main_category' column
grading_df = grading_df.dropna(subset=['main_category'])


################################################################## 
################## FEATURE SELECTION #############################
##################################################################

# Define the selected variables for modeling
variables = ['state', 'category', 'country', 'goal_usd', 'deadline_month', 'deadline_weekday',
             'launched_at_month', 'launched_at_weekday', 'video', 'project_duration', 'name_len_clean',
             'blurb_len_clean']

# Select the relevant columns
grading_df = grading_df[variables]

# Assign "Others" to null values in the 'category' column
grading_df['category'].fillna('Others', inplace=True)

# Drop duplicate rows
grading_df = grading_df.drop_duplicates()

# Drop remaining rows with any missing values
grading_df = grading_df.dropna()

# Encode the target variable
grading_df['state'] = grading_df['state'].replace({'successful': 1, 'failed': 0})

#Encoding Categorical Variables
grading_df = pd.get_dummies(grading_df, columns=['category', 'country', 'video', 'deadline_weekday', 'launched_at_weekday'], drop_first=True)

################################################################## 
################## OUTLIER DETECTION #############################
##################################################################

# Initialize Isolation Forest model for outlier detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(grading_df)
grading_df = grading_df[outliers == 1]
print("Dataset shape after outlier removal:", grading_df.shape)

################################################################## 
################## Model Prediction #############################
##################################################################

# Separate features and target variable
X_grading = grading_df.drop(columns=['state'], errors='ignore')  
y_grading = grading_df['state']

# Ensure the columns match the model's training features
X_grading = X_grading.reindex(columns=X_train.columns, fill_value=0)

# Apply the same scaler used during training
scaler = StandardScaler()
X_grading[numeric] = scaler.fit_transform(X_grading[numeric])

# Make predictions using the pre-trained model
y_pred_grading = optimized_rf.predict(X_grading)

# Calculate accuracy
grading_accuracy = accuracy_score(y_grading, y_pred_grading)
print(f"Accuracy on the grading dataset: {grading_accuracy:.4f}")








------------------------------------------------------------------------------
################################################################## 
################## 3. Cluster #######################################
##################################################################
------------------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Load the dataset
file_path = "/Users/julesvalois/Downloads/Kickstarter.xlsx"
df = pd.read_excel(file_path)

################################################################## 
################## DATA PROCESSING ###############################
##################################################################

# Convert goal to USD using static_usd_rate
df['goal_usd'] = df['goal'] * df['static_usd_rate']

# Drop the original goal and static_usd_rate columns
df = df.drop(columns=['goal', 'static_usd_rate'])

# Calculate project duration in days
df['project_duration'] = (df['deadline'] - df['launched_at']).dt.days

# Check for and display columns with missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Columns with missing values:")
print(missing_values)

# Drop rows with missing values in the 'main_category' column
df = df.dropna(subset=['main_category'])

# Exclude 'state' from clustering
state_column = df.pop('state')  # Save 'state' for later use in classification, but exclude it from clustering

################################################################## 
###################### PRE-PROCESSING ############################
##################################################################

# Separate numeric and categorical variables
numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Ensure categorical variables are correctly encoded as 'category'
for col in categorical_vars:
    df[col] = df[col].astype('category')

# Encode categorical variables
for col in categorical_vars:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Combine all numeric and encoded categorical variables
X = df[numeric_vars + categorical_vars]

# Scale numeric variables
scaler = StandardScaler()
X[numeric_vars] = scaler.fit_transform(X[numeric_vars])

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso_forest.fit_predict(X)  # -1 indicates outliers, 1 indicates inliers
df_cleaned = df[df['outlier'] == 1].drop(columns=['outlier'])


################################################################## 
###################### APPLY K-PROTOTYPES ########################
##################################################################

# Sample the dataset (20% of data)
df_sample = df_cleaned.sample(frac=0.2, random_state=42)
X_sample = df_sample[numeric_vars + categorical_vars].values
categorical_indices = list(range(len(numeric_vars), len(numeric_vars) + len(categorical_vars)))

# Test Silhouette Scores for a limited range of k
k_range = range(2, 7)  # Smaller range to reduce computation
silhouette_scores = []

for k in k_range:
    kproto = KPrototypes(n_clusters=k, random_state=42, init='Cao')
    labels = kproto.fit_predict(X_sample, categorical=categorical_indices)
    score = silhouette_score(X_sample, labels, metric='euclidean')
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title("Silhouette Score for K-Prototypes (Sampled Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# Define the number of clusters
k = 4  

# Combine numeric and categorical variables
X_combined = df_cleaned[numeric_vars + categorical_vars].values
# Determine indices of categorical variables
categorical_indices = list(range(len(numeric_vars), len(numeric_vars) + len(categorical_vars)))

# Apply K-Prototypes clustering
kproto = KPrototypes(n_clusters=k, random_state=42, init='Cao')  # Use 'Cao' initialization
cluster_labels = kproto.fit_predict(X_combined, categorical=categorical_indices)

# Add the cluster labels to the dataset
df_cleaned['cluster'] = cluster_labels
print("Cluster Centroids:", kproto.cluster_centroids_)
cluster_analysis = df_cleaned.groupby('cluster').mean()
print(cluster_analysis)




