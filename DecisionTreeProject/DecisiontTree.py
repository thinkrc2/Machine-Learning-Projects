# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:43:15 2025

@author: think
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_csv("USA_cars_datasets.csv")

## Identify Target and Features
TARGET = "price"
feature_cols = ["brand", "model", "year", "title_status", "mileage", "color"]

#separate num and cat features
numeric_features = ["year", "mileage"]
categorical_features = ["brand", "model", "title_status", "color"]

y = df[TARGET]
x = df[feature_cols]

#Train/test split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#preprocess

num_pre = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

cat_pre = Pipeline(steps=[("imputer", SimpleImputer(strategy = "most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer(
    transformers=[("num", num_pre, numeric_features),
                  ("cat", cat_pre, categorical_features)],
    remainder="drop")

#Build Model (you can adjust the values on the line below (default num 8,5,42)
reg = DecisionTreeRegressor(max_depth=15, min_samples_leaf=15, random_state=42)


model = Pipeline(steps=[("prep", preprocess), ("reg", reg)])


# Define parameter grid
param_grid = {
    "reg__max_depth": [5, 8, 10, 12, 15, None],
    "reg__min_samples_leaf": [1, 5, 10, 20],
    "reg__min_samples_split": [2, 5, 10, 20]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",  # you can also use 'r2' or 'neg_root_mean_squared_error'
    cv=5,                # 5-fold cross-validation
    n_jobs=-1,           # use all CPU cores
    verbose=2            # print progress
)

# Fit the grid search on training data
grid_search.fit(x_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validated MAE:", -grid_search.best_score_)

#evalute and fit model
best_model = grid_search.best_estimator_

pred = best_model.predict(x_test)
print("\nModel Evaluation with Best Parameters:")
print("MAE:", mean_absolute_error(y_test, pred))
print("r2:", r2_score(y_test, pred))

plt.figure(figsize=(25, 15))
encoder = best_model.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
encoded_features = encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(encoded_features)

plot_tree(best_model.named_steps["reg"], feature_names=all_features, filled=True, max_depth=3, rounded=True)
plt.rcParams.update({'font.size': 7})
plt.show()



#Feature Import (get column names)
ohe = best_model.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
cat_out = list(ohe.get_feature_names_out(categorical_features)) if len(categorical_features) else [] 
feature_names = numeric_features + cat_out

importances = best_model.named_steps["reg"].feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
print("\nTop Features: ")
for i in top_idx:
    print(f"{feature_names[i]:30s} {importances[i]:.4f}")


#print example and prediction

example = pd.DataFrame([{
    "brand": "ford",
    "model": "door",
    "year": 2013,
    "title_status": "clean vehicle",
    "mileage": 52000,
    "color": "black"   
    }])


print('\nExample prediction:', best_model.predict(example)[0])
