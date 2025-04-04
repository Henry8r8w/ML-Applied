## Gradient Boosting
### The Idea
**Ensemble method**: combine the predictions of several models (e.g., several trees, in the case of random forests)
- Iteratively adding models into sequential ensemble (aka. boosting), attempting to minimize error from previous models

### Workflow
1. **Initialize** with a simple model (even if predictions are inaccurate)
2. **Perform Iterative Cycle**:
   - Generate predictions using current ensemble (sum of all model predictions)
   - Calculate loss function (e.g., mean squared error)
   - Fit a new model focused on reducing the loss function
   - Add the new model to the ensemble
   - Repeat

Note: 
- The "gradient" in gradient boosting refers to gradient descent

## Implementation with XGBoost
```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

## Simple Implementation 
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# Slightly better Implementation
  # n_estimator - define number of ensemble (# of boosting rounds)
  # early_stopping_rounds - stop training when validation set does not improve metrics by n rounds
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False) 
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```

## CheatSheet
One Hot Encoding with pandas
```python
# Obtain dummy columns in your dataframe; additional columns by n-cardinal x col columns are added; each cardinal forms its own column 
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid) 
X_test = pd.get_dummies(X_test)

# Ensure validation and test data sets have aligned columns with training data set
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
```
