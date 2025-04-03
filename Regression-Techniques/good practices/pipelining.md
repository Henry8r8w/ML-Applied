## ML is an Iteraitve Process
Therefore, there are benefits of making your ML worflow faster and reproducible for testing, validation, and debug

The benefits of pipelining your ML workflow 
- Cleaner Looking Code (easy for debug and tracking)
- Usability (quick test to production)
- Simple workflow: easy to switch model, quicker parameter tuning, cascade change 

## Creating a Pipeline with Categorical and Numerical Preprocessing
### Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split
categorical_cols = [cname for cname in X_train_full.columns if 
                   X_train_full[cname].nunique() < 10 and 
                   X_train_full[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train_full.columns if 
                 X_train_full[cname].dtype in ['int64', 'float64']]

# Keep only selected columns
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy() # .copy() is a good practice; it allow changes on X_train to be not interefering with X_train_full
X_valid = X_valid_full[my_cols].copy()
```

### Building the Preprocessing Pipeline
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),  # replace missing values with constant
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # convert unkonw to 0 by default
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

### Model Selection

```python
from sklearn.ensemble import RandomForestRegressor

# Define / Choose model
model = RandomForestRegressor(n_estimators=100, random_state=0)
```

### Creating and Using the Full Pipeline

```python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

## CheatSheet
### ColumnTransformer
The `ColumnTransformer` applies different transformations to different columns:
- First parameter: list of tuples in format `(name, transformer, columns)`
- Each transformer processes the specified columns

### Pipeline Configuration
- Numerical preprocessing: using `SimpleImputer` with `strategy='constant'` to fill missing values
- Categorical preprocessing: Two-step pipeline:
  1. Fill missing values with `SimpleImputer(strategy='constant')`
  2. Apply one-hot encoding with `OneHotEncoder(handle_unknown='ignore')`

### Validation vs. Test Data
- Validation data: used during model development to tune hyperparameters and assess model performance
- Test data: used only once at the very end to get an unbiased estimate of model performance
-  Keep the test data untouched until submission

### Appeared Functions
- `df.copy()`: create an operational indepdent copy of df
