import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
import os

# Set working directory
os.chdir('C:/Vedant/Int-data/Phone Pricing/Mobile Phone Pricing')

# Load the dataset
phone = pd.read_csv("dataset.csv")

# Split dataset into features (X) and target variable (y)
X = phone.drop('price_range', axis=1)  # Feature variables
y = phone['price_range']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, test_size=0.2)

# Initialize a RandomForestClassifier with basic parameters
rf = RandomForestClassifier(random_state=23, n_estimators=15)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Evaluate the model's accuracy
print("Accuracy Score:", accuracy_score(y_test, y_pred))

##### Hyperparameter Tuning #####

# Define a KFold cross-validator with 5 splits
kfold = KFold(n_splits=5, shuffle=True, random_state=23)

# Define a new RandomForest model
rf = RandomForestClassifier(random_state=23, n_estimators=15)

# Define a dictionary of parameters for GridSearchCV
params = {
    'max_features': [3, 4, 5, 6, 7, 8, 9, 10],  # Number of features to consider at each split
    'max_depth': [None, 2, 3, 5, 6],  # Maximum depth of the tree
    'random_state': [22, 23, 24, 25, 26]  # Different random states for variability
}

# Perform GridSearchCV to find the best hyperparameters
gcv = GridSearchCV(rf, param_grid=params, n_jobs=-1, cv=kfold, scoring='accuracy', verbose=3)

# Fit the model on the entire dataset for best parameter selection
gcv.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", gcv.best_params_)
print("Best Score:", gcv.best_score_)

# Retrieve the best model from GridSearchCV
best_model = gcv.best_estimator_

######### Save the Best Model #########

import pickle

# Save the trained model using Pickle
file_model = open("C:/Vedant/Dockar_int/Dockar_Mobile/mobile.pkl", "wb")
pickle.dump(best_model, file_model)
file_model.close()

print("Model saved successfully.")
